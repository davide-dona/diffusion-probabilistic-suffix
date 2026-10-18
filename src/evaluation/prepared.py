import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from fcfdeclare import Checker

from src.evaluation.signature import SignatureKernel
from src.inference.generation import Generation

# Seeds the uniform that breaks PIT ties, together with the prefix key.
PIT_SEED = 42


@dataclass(frozen=True, slots=True)
class PreparedPrefix:
    """Decoded values and constraint checks shared by every metric for one prefix.

    Sample conformance arrays have shape [U], one entry per distinct suffix in suffix order;
    generation sample counts give their draw multiplicities. A conformance share is the fraction
    of constraints the full trace satisfies, and full conformance is 1.0 when it satisfies them
    all. Suffix lengths and remaining times have shape [S], one entry per draw, and are compared
    with the scalar truth. Times are in minutes. The signature Gram matrix has shape
    [S + 1, S + 1]: the normalized signature kernel between every pair of draws, with the
    observed suffix last. The PIT noise is one uniform on [0, 1] drawn
    from the prefix key alone, so it does not depend on the worker or batch scoring the prefix.
    """

    generation: Generation
    suffix_lengths: np.ndarray
    remaining_times: np.ndarray
    true_suffix_length: float
    true_remaining_time: float
    pit_noise: float
    signature_gram: np.ndarray
    sample_conformance_share: np.ndarray
    sample_full_conformance: np.ndarray
    observed_conformance_share: float
    observed_full_conformance: float

    @classmethod
    def of_batch(
        cls, generations: Sequence[Generation], *, checker: Checker, kernel: SignatureKernel
    ) -> list[Self]:
        """Prepare the shared draw and observation arrays for a batch of prefixes.

        Every full trace of the batch is checked in one call, since the checker's fixed cost per
        call outweighs its work per trace. Every signature kernel of the batch is solved in one
        call for the same reason.

        Args:
            generations: Decoded prefixes, sampled suffixes, and observed continuations.
            checker: Declare checker applied to each prefix plus each of its suffixes.
            kernel: Signature kernel comparing the draws and observed suffix of each prefix.

        Returns:
            Shared metric inputs in input order, with numeric arrays stored in float64. Empty
            input gives an empty list.
        """
        if not generations:
            return []
        # Each prefix contributes its distinct sampled traces followed by its observed one
        conformance = checker.check(
            [
                generation.prefix_activities + suffix
                for generation in generations
                for suffix in (*generation.samples.suffixes, generation.truth.activities)
            ]
        )
        share, full = conformance.share, conformance.full
        grams = kernel.grams(_suffix_paths(generations, kernel=kernel))
        prepared = []
        start = 0
        for generation, gram in zip(generations, grams, strict=True):
            stop = start + len(generation.samples.suffixes) + 1
            prepared.append(
                cls._of(generation, share=share[start:stop], full=full[start:stop], gram=gram)
            )
            start = stop
        return prepared

    @classmethod
    def _of(
        cls, generation: Generation, *, share: np.ndarray, full: np.ndarray, gram: np.ndarray
    ) -> Self:
        """Prepare one prefix from the conformance of its sampled traces, then its observed one."""
        samples = generation.samples
        truth = generation.truth
        noise = np.random.default_rng(
            (PIT_SEED, zlib.crc32(generation.case_id.encode()), generation.prefix_len)
        ).random()
        return cls(
            generation=generation,
            suffix_lengths=samples.lengths.astype(np.float64),
            remaining_times=samples.remaining_time_minutes,
            true_suffix_length=float(len(truth)),
            true_remaining_time=truth.remaining_time_minutes,
            pit_noise=float(noise),
            signature_gram=gram,
            sample_conformance_share=share[:-1],
            sample_full_conformance=full[:-1],
            observed_conformance_share=float(share[-1]),
            observed_full_conformance=float(full[-1]),
        )


def _suffix_paths(
    generations: Sequence[Generation], *, kernel: SignatureKernel
) -> list[list[np.ndarray]]:
    """Embed each prefix's draws, in draw order, followed by its observed suffix."""
    codes = {
        code
        for generation in generations
        for suffix in (*generation.samples.suffixes, generation.truth.activities)
        for code in suffix
    }
    offset = min(map(ord, codes), default=0)
    alphabet = max(map(ord, codes), default=offset - 1) - offset + 1
    groups = []
    for generation in generations:
        samples, truth = generation.samples, generation.truth
        times = samples.inter_event_time_minutes
        offsets = samples.offsets
        groups.append(
            [
                *(
                    kernel.embed(
                        samples.suffixes[taken],
                        times[offsets[draw] : offsets[draw + 1]],
                        offset=offset,
                        alphabet=alphabet,
                    )
                    for draw, taken in enumerate(samples.taken)
                ),
                kernel.embed(
                    truth.activities,
                    truth.inter_event_time_minutes,
                    offset=offset,
                    alphabet=alphabet,
                ),
            ]
        )
    return groups
