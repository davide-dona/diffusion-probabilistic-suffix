# Dataset Runtime Guide

This package turns preprocessed event logs into immutable tensor records. The model boundary is
`TraceCut`: one observed prefix, its held-out suffix, and identifiers needed to decode and evaluate
the prediction.

## Tensor Contracts

`Events` contains aligned `[T]` fields for one item or `[B, T]` fields after collation:

| Field | Type and meaning |
| --- | --- |
| `activities` | Integer activity vocabulary indices. |
| `resources` | Integer resource vocabulary indices. |
| `inter_event_times` | Standardized minutes since the preceding event. |
| `categorical_attributes` | Integer feature indices, shape `[..., C]`. |
| `numeric_attributes` | Standardized numeric features, shape `[..., N]`. |
| `numeric_attributes_present` | Float presence indicators paired with numeric features. |
| `length` | Unpadded sequence length, shape `[B]` after collation. |

`Events.pad_mask()` is true at PAD activity positions. `Events.padded(to)` pads every channel to
the same width and rejects truncation. `Events.cut(index)` and `Events.to(device)` must preserve
channel alignment.

`TraceCut` contains `case_id`, `prefix`, `suffix`, `inter_event_times`, and `remaining_times`.
The separate time targets are standardized values used by model losses. A collated batch has
prefix and suffix events padded independently.

## Prefix and Suffix Enumeration

`TraceDataset` reads one preprocessed split through `DatasetCodec`, groups it by case, and exposes
every permitted cut. A cut at position `i` yields events before `i` as the prefix and events from
`i` onward as the suffix. The suffix appends EOT, so `suffix.length` includes EOT while real timed
events number `suffix.length - 1`.

Respect `_prefix_start` and `_prefix_end`. They encode leak-proof cut bounds, including cases that
cross chronological split boundaries. Do not recreate cut ranges from raw case lengths.

`length_sorted_indices()` orders cuts by true suffix length for efficient generation. It does not
change prefix identity. `fixed_subset()` selects a reproducible random subset from an explicit
`torch.Generator` and returns the whole dataset when the requested size is at least its length.

## Codec Contracts

`DatasetCodec.fit` receives the training split only. It stores:

- Activity and resource vocabularies with stable special token indices.
- Categorical feature vocabularies and non-overlapping offsets into one shared embedding table.
- Mean, standard deviation, and optional log scaling for numeric features, inter-event time, and
  remaining time.
- The maximum retained trace length and the dataset configuration used to locate split artifacts.

Activity special tokens are PAD, UNK, EOT, and SOS. The activity codec used by evaluation maps
activity names to Unicode private-use characters so sequence metrics can operate on compact strings.
Never change special token order or fitted offsets without an explicit artifact migration.

Numeric encoding replaces non-finite normalized values with zero and supplies a separate presence
channel. Decoding inter-event and remaining times must use the exact fitted column, including its
log transform.

## Changes and Tests

- Preserve channel shapes and dtype conventions across padding, collation, and device moves.
- Fit any new learned transform on train only, serialize it in `codec.json`, and use it unchanged
  for validation and test.
- Update shared embeddings when adding a model-visible event channel.
- Test empty or missing features, PAD masks, cut boundaries, special tokens, codec round trips, and
  deterministic subsets as relevant.
- Run model contract tests after changing `Events`, `TraceCut`, or `DatasetCodec` because every
  architecture consumes them.

See [`data/AGENTS.md`](../../data/AGENTS.md) for stored artifact ownership.
