from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.models.architectures.shared_components.sutran.cache import LayerCache
from src.models.contracts import GeneratedSuffix

if TYPE_CHECKING:
    from src.models.architectures.shared_components.sutran.decoder import CausalDecoder


def generate_autoregressive(
    *,
    decoder: CausalDecoder,
    prefix_encoded: torch.Tensor,
    prefix_pad_mask: torch.Tensor,
    max_steps: int,
) -> GeneratedSuffix:
    """Decode independent rows from prefix memory `[R, P, D]` using one cached step at a time.

    Returns raw activities and standardized durations `[R, T]`, with lengths excluding EOT.
    Values at and after EOT are retained; each model owns its output padding policy.
    Remaining time is a placeholder until the model applies its codec conversion.
    """
    batch_size = prefix_encoded.size(dim=0)
    device = prefix_encoded.device
    # What the decoder reads at each step: SOS first, exactly how `_teacher_forced_input`
    # opens, then the activity the previous step predicted.
    next_input = torch.full(
        size=(batch_size, 1),
        fill_value=decoder.sos_activity_index,
        dtype=torch.long,
        device=device,
    )

    generated_activities = torch.zeros(
        size=(batch_size, max_steps), dtype=torch.long, device=device
    )
    generated_inter_event_times = torch.zeros(
        size=(batch_size, max_steps), dtype=prefix_encoded.dtype, device=device
    )
    # A row that never emits EOT ran to the cap, so that is the length it keeps.
    lengths = torch.full(size=(batch_size,), fill_value=max_steps, dtype=torch.long, device=device)
    finished = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)

    steps_taken = max_steps
    # Seeded before the loop rather than left None for its first iteration: every layer's
    # suffix cache is preallocated to `max_steps` right away, so even the first step writes
    # into it in place instead of starting the cache off at its exact size.
    caches: list[LayerCache] = [
        layer.init_cache(prefix_encoded=prefix_encoded, max_steps=max_steps)
        for layer in decoder.layers
    ]
    for position in range(max_steps):
        # Only this one position is new; everything before it is in `caches`.
        hidden, caches = decoder._run_layers(
            activities=next_input,
            prefix_encoded=prefix_encoded,
            prefix_pad_mask=prefix_pad_mask,
            start_position=position,
            caches=caches,
        )
        features = decoder.shared_layer(hidden[:, 0])  # [batch_size, head_hidden_dim]
        activities, times = decoder.sample(features)
        generated_activities[:, position] = activities
        generated_inter_event_times[:, position] = times
        next_input = activities.unsqueeze(dim=1)  # [batch_size, 1]

        # A suffix ends at its first EOT, so a later one cannot move the length back.
        just_finished = ~finished & (activities == decoder.eot_activity_index)
        lengths = lengths.masked_fill(mask=just_finished, value=position)
        finished |= just_finished
        # Reading this stalls the device queue once per step, but suffixes are far shorter
        # than `max_steps` on every log here, so most of the loop is skipped outright.
        if bool(finished.all()):
            steps_taken = position + 1
            break

    return GeneratedSuffix(
        activities=generated_activities[:, :steps_taken],  # [batch_size, steps]
        lengths=lengths,
        inter_event_times=generated_inter_event_times[:, :steps_taken],  # [batch_size, steps]
        remaining_time=generated_inter_event_times.new_zeros(size=(batch_size,)),
    )
