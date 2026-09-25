import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

from src.artifacts import (
    Provenance,
    RunIdentity,
)
from src.inference.generation import DecodedEvents, Draws, Generation

# Which prefix a row answers, and so what the rows of two runs of one log are matched on. A cut is
# a case and a length, and the pair is unique within a file.
type PrefixKey = tuple[str, int]

# The sampler configuration used for generation. The checkpoint hash does not alone describe
# inference settings when a sampler is selected after training.
_SAMPLING = b'sampling'
_ACTIVITIES = b'activities'


def with_activity_vocabulary(schema: pa.Schema, vocabulary: Sequence[str]) -> pa.Schema:
    """Attach the activity vocabulary used to encode generations."""
    return schema.with_metadata(
        (schema.metadata or {}) | {_ACTIVITIES: json.dumps(list(vocabulary)).encode()}
    )


def read_activity_vocabulary(schema: pa.Schema) -> tuple[str, ...]:
    """Read the activity vocabulary from a generations schema."""
    raw = (schema.metadata or {}).get(_ACTIVITIES)
    if raw is None:
        raise ValueError('Missing activity vocabulary; regenerate this file.')
    try:
        vocabulary = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid activity vocabulary') from error
    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or any(not isinstance(name, str) or not name for name in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise ValueError('Invalid activity vocabulary')
    return tuple(vocabulary)


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
_EVENTS = pa.struct(
    [
        ('activities', _SUFFIX),
        ('inter_event_time_minutes', _INTER_EVENT_TIMES),
        ('remaining_time_minutes', pa.float32()),
    ]
)
_DRAW = pa.struct(
    [
        ('suffix_index', pa.int32()),
        ('inter_event_time_minutes', _INTER_EVENT_TIMES),
        ('remaining_time_minutes', pa.float32()),
        ('used_eot_sentinel', pa.bool_()),
    ]
)
_SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        ('prefix_activities', _SUFFIX),
        ('generated_suffixes', pa.list_(pa.field(name='element', type=_SUFFIX))),
        ('generated_draws', pa.list_(pa.field(name='element', type=_DRAW))),
        ('truth', _EVENTS),
    ]
)

# The two columns that identify a prefix, which is all `prefix_keys` reads.
_KEY_COLUMNS = ['case_id', 'prefix_len']
_FLOAT_LEAVES = [
    'generated_draws.list.element.inter_event_time_minutes.list.element',
    'generated_draws.list.element.remaining_time_minutes',
    'truth.inter_event_time_minutes.list.element',
    'truth.remaining_time_minutes',
]


def _generation_row(generation: Generation) -> dict:
    return {
        'case_id': generation.case_id,
        'prefix_len': generation.prefix_len,
        'prefix_activities': generation.prefix_activities,
        'generated_suffixes': list(generation.samples.suffixes),
        'generated_draws': [
            {
                'suffix_index': index,
                'inter_event_time_minutes': events.inter_event_time_minutes,
                'remaining_time_minutes': events.remaining_time_minutes,
                'used_eot_sentinel': events.used_eot_sentinel,
            }
            for index, events in zip(
                generation.samples.taken, generation.samples.events, strict=True
            )
        ],
        'truth': {
            'activities': generation.truth.activities,
            'inter_event_time_minutes': generation.truth.inter_event_time_minutes,
            'remaining_time_minutes': generation.truth.remaining_time_minutes,
        },
    }


def _generation_from_row(row: dict) -> Generation:
    suffixes = tuple(row['generated_suffixes'])
    draws = row['generated_draws']
    taken = tuple(draw['suffix_index'] for draw in draws)
    if any(index < 0 or index >= len(suffixes) for index in taken):
        raise ValueError('Generation draw references an unknown suffix')
    if row['prefix_len'] != len(row['prefix_activities']):
        raise ValueError('Generation prefix length does not match its activities')
    return Generation(
        case_id=row['case_id'],
        prefix_activities=row['prefix_activities'],
        samples=Draws(
            suffixes=suffixes,
            taken=taken,
            events=[
                DecodedEvents(
                    activities=suffixes[index],
                    inter_event_time_minutes=draw['inter_event_time_minutes'],
                    remaining_time_minutes=draw['remaining_time_minutes'],
                    used_eot_sentinel=draw['used_eot_sentinel'],
                )
                for index, draw in zip(taken, draws, strict=True)
            ],
        ),
        truth=DecodedEvents(**row['truth']),
    )


class GenerationWriter:
    """A generations file, open for writing, one block per batch.

    Used as a context manager: the file's footer is written when it closes, so a model that dies
    mid-generation leaves nothing readable rather than a file that lies about its length.
    """

    def __init__(
        self,
        path: Path,
        provenance: Provenance,
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
            sampling: The activity-head controls or diffusion schedules and sampler settings.
                Written so the checkpoint hash alone need not identify inference settings.
        """
        provenance.require_checkpoint_source()
        schema = with_activity_vocabulary(_SCHEMA, vocabulary)
        if sampling is not None:
            schema = schema.with_metadata(
                (schema.metadata or {})
                | {_SAMPLING: json.dumps(OmegaConf.to_container(sampling, resolve=True))}
            )
        schema = provenance.attach_to_schema(schema)
        self._path = path
        self._writer = pq.ParquetWriter(
            where=self._path,
            schema=schema,
            compression='zstd',
            compression_level=9,
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
        rows = [_generation_row(generation) for generation in generations]
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
        try:
            self._provenance = Provenance.from_parquet(self._parquet)
            self._provenance.require_checkpoint_source()
            self._vocabulary = read_activity_vocabulary(self._parquet.schema_arrow)
        except Exception:
            self._parquet.close()
            raise

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
    def provenance(self) -> Provenance:
        """Validated training run and checkpoint that produced this file."""
        return self._provenance

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
        return self._vocabulary

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
            `generate_batch` produced them.
        """
        return [
            _generation_from_row(row) for row in self._parquet.read_row_group(block).to_pylist()
        ]
