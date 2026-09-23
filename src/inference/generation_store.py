import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

from src.artifacts import (
    ArtifactProvenance,
    RunIdentity,
    read_activity_vocabulary,
    read_provenance_metadata,
    with_activity_vocabulary,
    with_provenance_metadata,
)
from src.inference.generation import DecodedEvents, Draws, Generation

# Which prefix a row answers, and so what the rows of two runs of one log are matched on. A cut is
# a case and a length, and the pair is unique within a file.
type PrefixKey = tuple[str, int]

# How the activity head was sampled. The checkpoint hash does not settle this because the sampler
# is chosen after training and can be changed without the weights moving.
_SAMPLING = b'sampling'

# One sequence of activities, one character each. A suffix is a string rather than a list of
# names. Activity names live once in the vocabulary metadata; edit distance reads the string.
_SUFFIX = pa.string()

# One model's inter-event time before each of its activities. Timestamps are these accumulated, so
# they are not written a second time. The model emits float32; `denormalize` widens to
# float64 on the way out, and storing that width would double the largest column of the file to
# carry digits the decoder never produced.
_INTER_EVENT_TIMES = pa.list_(pa.field(name='element', type=pa.float32()))

# The schema of the Parquet file that holds a model's generations. One row per prefix: the samples
# nest inside it, so nothing describing the prefix is written once per sample.
_SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        # The events before the cut, which a constraint over the whole trace is checked against.
        ('prefix_activities', _SUFFIX),
        # The distinct suffixes drawn for this prefix, each written once however many draws landed
        # on it, and which of them each draw took, in the order they were drawn. A mean over the
        # draws is the weighted mean over the distinct suffixes.
        ('generated_suffixes', pa.list_(pa.field(name='element', type=_SUFFIX))),
        ('generated_draws', pa.list_(pa.field(name='element', type=pa.int16()))),
        # Still one entry per draw, in draw order. Times do not fold with activities because two
        # draws of one activity suffix can still produce different time values.
        (
            'generated_inter_event_time_minutes',
            pa.list_(pa.field(name='element', type=_INTER_EVENT_TIMES)),
        ),
        (
            'generated_remaining_time_minutes',
            pa.list_(pa.field(name='element', type=pa.float32())),
        ),
        (
            'generated_used_eot_sentinel',
            pa.list_(pa.field(name='element', type=pa.bool_())),
        ),
        ('true_activities', _SUFFIX),
        ('true_inter_event_time_minutes', _INTER_EVENT_TIMES),
        ('true_remaining_time_minutes', pa.float32()),
    ]
)

# The two columns that identify a prefix, which is all `prefix_keys` reads.
_KEY_COLUMNS = ['case_id', 'prefix_len']

_COMPRESSION = 'zstd'

# Worth the write time: generation is GPU-bound over hours, where the whole file costs under a
# minute more to compress at this level and comes out a tenth smaller than at the default.
_COMPRESSION_LEVEL = 9

# The float columns, named as Parquet names their leaves. Byte-stream-split splits a float into its
# four byte planes before compressing, so the exponents of a column line up and zstd has something
# repetitive to find; on the inter-event times, which are continuous and share nothing as whole
# values, it is the difference between compressing and not.
_FLOAT_LEAVES = [
    'generated_inter_event_time_minutes.list.element.list.element',
    'generated_remaining_time_minutes.list.element',
    'true_inter_event_time_minutes.list.element',
    'true_remaining_time_minutes',
]


class GenerationWriter:
    """A generations file, open for writing, one block per batch.

    Used as a context manager: the file's footer is written when it closes, so a model that dies
    mid-generation leaves nothing readable rather than a file that lies about its length.
    """

    def __init__(
        self,
        path: Path,
        provenance: ArtifactProvenance,
        *,
        vocabulary: Sequence[str],
        sampling: DictConfig | None,
    ) -> None:
        """
        Args:
            path: Destination file inside the active Hydra output directory.
            provenance: Training run and source checkpoint that produced the artifact.
            vocabulary: The activity names the suffixes are spelled on, in code order, from
                `ActivityCodec.vocabulary`. Written into the file so it says what its own
                characters mean.
            sampling: How the activity head was sampled. Written so the checkpoint hash alone does
                not need to identify the inference setting.
        """
        schema = with_activity_vocabulary(_SCHEMA, vocabulary)
        if sampling is not None:
            schema = schema.with_metadata(
                (schema.metadata or {})
                | {_SAMPLING: json.dumps(OmegaConf.to_container(sampling, resolve=True))}
            )
        schema = with_provenance_metadata(schema, provenance.as_metadata())
        self._path = path
        self._writer = pq.ParquetWriter(
            where=self._path,
            schema=schema,
            compression=_COMPRESSION,
            compression_level=_COMPRESSION_LEVEL,
            # Dictionary encoding takes precedence over byte-stream-split wherever it is left on,
            # and a column of continuous inter-event times has no dictionary worth building, so
            # the two are set together.
            use_dictionary=False,
            use_byte_stream_split=_FLOAT_LEAVES,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._writer.close()
        except BaseException:
            self._path.unlink(missing_ok=True)
            raise
        if exception_type is not None:
            self._path.unlink(missing_ok=True)

    def write(self, generations: list[Generation]) -> None:
        """Write one batch's generations as one block of the file, one row per prefix.

        Args:
            generations: The model's answers, in the order `generate_batch` returned them.
        """
        # Dicts here because they are what arrow's constructor takes, keyed by the schema's own
        # names.
        rows = [
            {
                'case_id': generation.case_id,
                'prefix_len': generation.prefix_len,
                'prefix_activities': generation.prefix_activities,
                'generated_suffixes': list(generation.samples.suffixes),
                'generated_draws': list(generation.samples.taken),
                'generated_inter_event_time_minutes': [
                    events.inter_event_time_minutes for events in generation.samples.events
                ],
                'generated_remaining_time_minutes': [
                    events.remaining_time_minutes for events in generation.samples.events
                ],
                'generated_used_eot_sentinel': [
                    events.used_eot_sentinel for events in generation.samples.events
                ],
                'true_activities': generation.truth.activities,
                'true_inter_event_time_minutes': generation.truth.inter_event_time_minutes,
                'true_remaining_time_minutes': generation.truth.remaining_time_minutes,
            }
            for generation in generations
        ]
        self._writer.write_table(table=pa.Table.from_pylist(mapping=rows, schema=_SCHEMA))


class Generations:
    """A generations file, open for reading.

    Used as a context manager, and opened once per process that reads it: the scoring pool gives
    each of its workers its own, since a block is decoded where it is scored and nothing but the
    handful of floats it reduces to comes back.
    """

    def __init__(self, path: Path) -> None:
        """
        Args:
            path: The generations file to read, from `python -m pipelines.generate`.
        """
        self._parquet = pq.ParquetFile(path)
        actual = self._parquet.schema_arrow.remove_metadata()
        if not actual.equals(_SCHEMA):
            self._parquet.close()
            raise ValueError(
                f'{path} uses an incompatible generations schema. Regenerate it with '
                '`python -m pipelines.generate`.'
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._parquet.close()

    @property
    def metadata(self) -> dict[str, str]:
        """Stable run identity, dataset fingerprint, and source checkpoint hash.

        Raises:
            ValueError: If the file has no run identity.
        """
        return self.provenance.as_metadata()

    @property
    def provenance(self) -> ArtifactProvenance:
        """Validated training run and checkpoint that produced this file."""
        return ArtifactProvenance.from_metadata(read_provenance_metadata(self._parquet))

    @property
    def run(self) -> RunIdentity:
        """The training run that produced this file."""
        return self.provenance.run

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """The activity names this file spells its suffixes on, in code order.

        What `ActivityCodec.from_vocabulary` uses to recreate a codebook, and what a reader
        compares against the
        continuation index's own before it scores a single prefix.

        Raises:
            ValueError: If the file has no activity vocabulary.
        """
        return read_activity_vocabulary(self._parquet.schema_arrow)

    @property
    def blocks(self) -> int:
        """How many blocks the file holds, which is how many units of work scoring it splits into.

        A block is what one call to `GenerationWriter.write` wrote, held as a Parquet row group. A
        prefix cannot cross one, since a row holds one, which is what makes a block an independent
        unit: what it costs to read and to score is set by the batch a model wrote rather than
        by the
        size of the split.
        """
        return self._parquet.num_row_groups

    @property
    def prefixes(self) -> int:
        """How many prefixes the file answers, one per row."""
        return self._parquet.metadata.num_rows

    def prefix_keys(self) -> list[PrefixKey]:
        """Which prefixes this file answers, without decoding a single suffix.

        Returns:
            The key of every row, in the order the file holds them, which is the order a walk of
            its blocks scores them in. Only the two columns that identify a prefix are read, which
            is cheap even on a file of a quarter of a million rows.
        """
        table = self._parquet.read(columns=_KEY_COLUMNS)
        return list(
            zip(
                table.column('case_id').to_pylist(),
                table.column('prefix_len').to_pylist(),
                strict=True,
            )
        )

    def block(self, block: int) -> list[Generation]:
        """Read one block of the file back.

        Args:
            block: Which block to decode, in `range(self.blocks)`.
        Returns:
            The generation for each prefix of the block, in the order they were written, exactly as
            `generate_batch` produced them. Read a column at a time: Arrow converts a whole column
            to Python in one call, and the lists that come back are the plain lists `DecodedEvents`
            promises, so nothing is copied a second time and the block itself is dropped as soon as
            this returns.
        """
        table = self._parquet.read_row_group(block)
        columns = {name: table.column(name).to_pylist() for name in table.schema.names}

        generations = []
        for position in range(table.num_rows):
            suffixes = columns['generated_suffixes'][position]
            taken = columns['generated_draws'][position]
            generations.append(
                Generation(
                    case_id=columns['case_id'][position],
                    prefix_activities=columns['prefix_activities'][position],
                    samples=Draws(
                        suffixes=tuple(suffixes),
                        taken=tuple(taken),
                        events=[
                            DecodedEvents(
                                activities=suffixes[index],
                                inter_event_time_minutes=inter_event_time_minutes,
                                remaining_time_minutes=remaining_time_minutes,
                                used_eot_sentinel=used_eot_sentinel,
                            )
                            for (
                                index,
                                inter_event_time_minutes,
                                remaining_time_minutes,
                                used_eot_sentinel,
                            ) in zip(
                                taken,
                                columns['generated_inter_event_time_minutes'][position],
                                columns['generated_remaining_time_minutes'][position],
                                columns['generated_used_eot_sentinel'][position],
                                strict=True,
                            )
                        ],
                    ),
                    truth=DecodedEvents(
                        activities=columns['true_activities'][position],
                        inter_event_time_minutes=columns['true_inter_event_time_minutes'][position],
                        remaining_time_minutes=columns['true_remaining_time_minutes'][position],
                    ),
                )
            )
        return generations
