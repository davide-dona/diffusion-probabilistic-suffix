import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from src.artifacts import Provenance, RunIdentity
from src.evaluation.scoring import EvaluationSummary, LengthSummary


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation results and source artifact provenance.

    provenance identifies the dataset, model, training run, dataset fingerprint, and checkpoint
    hash. summary contains overall scores and both observed-length breakdowns.
    """

    provenance: Provenance
    summary: EvaluationSummary

    def __post_init__(self) -> None:
        self.provenance.require_generations_source()

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read and validate a JSON evaluation report.

        Args:
            path: JSON report containing an evaluation summary and artifact provenance.

        Returns:
            The validated report, including its run and checkpoint metadata.

        Raises:
            ValidationError: If the report does not match the expected structure.
            ValueError: If the JSON or artifact provenance is invalid.
        """
        path = Path(path)
        payload = json.loads(path.read_bytes())
        if not isinstance(payload, dict) or set(payload) != {'provenance', 'summary'}:
            raise ValueError(f'{path} is not an evaluation report')
        provenance = Provenance.from_dict(payload['provenance'])
        report = _REPORT_ADAPTER.validate_python(payload)
        return cls(provenance=provenance, summary=report.summary)

    def write(self, path: str | Path) -> Path:
        """Write the report as JSON.

        Args:
            path: Destination file in an existing directory, replacing any existing contents.

        Returns:
            The destination path.
        """
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=4))
        return path


_REPORT_ADAPTER = TypeAdapter(EvaluationReport)


class Axis(StrEnum):
    """Metric aggregation levels."""

    OVERALL = 'overall'
    PREFIX = 'prefix'
    SUFFIX = 'suffix'


REPORT_COLUMNS = (
    'dataset',
    'model',
    'run_id',
    'axis',
    'length',
    'prefixes',
    'metric',
    'value',
)
SUMMARY_COLUMNS = (
    'dataset',
    'model',
    'axis',
    'length',
    'prefixes',
    'metric',
    'mean',
    'std',
    'runs',
)


def group_runs(
    reports: Iterable[tuple[Provenance, Path]],
) -> dict[str, dict[str, tuple[Path, ...]]]:
    """Group evaluation artifacts by dataset and model, keeping every run in input order.

    Runs of one model on one dataset are treated as seeds of one configuration.

    Args:
        reports: Artifact provenance and path for each evaluated run.

    Returns:
        Dataset names mapped to model names and the paths of their runs.

    Raises:
        ValueError: If one training run is given twice.
    """
    grouped: dict[str, dict[str, list[Path]]] = {}
    seen: dict[RunIdentity, Path] = {}
    for provenance, path in reports:
        run = provenance.run
        if run in seen:
            raise ValueError(f'{run} is given twice: {seen[run]}, {path}')
        seen[run] = path
        grouped.setdefault(run.dataset, {}).setdefault(run.model, []).append(path)
    return {
        dataset: {model: tuple(paths) for model, paths in models.items()}
        for dataset, models in grouped.items()
    }


def _summary_rows(
    summary: EvaluationSummary | LengthSummary,
    *,
    identity: dict[str, str],
    axis: Axis,
    length: int | None,
) -> Iterator[dict[str, object]]:
    for metric, value in summary.scores.flatten().items():
        yield identity | {
            'axis': axis,
            'length': length,
            'prefixes': summary.prefixes,
            'metric': metric,
            'value': value,
        }


def _report_rows(report: EvaluationReport) -> Iterator[dict[str, object]]:
    """Yield overall scores followed by prefix-length and suffix-length breakdowns."""
    run = report.provenance.run
    identity = {'dataset': run.dataset, 'model': run.model, 'run_id': run.run_id}
    summary = report.summary
    yield from _summary_rows(summary, identity=identity, axis=Axis.OVERALL, length=None)
    for entry in summary.by_prefix_length:
        yield from _summary_rows(entry, identity=identity, axis=Axis.PREFIX, length=entry.length)
    for entry in summary.by_suffix_length:
        yield from _summary_rows(entry, identity=identity, axis=Axis.SUFFIX, length=entry.length)


def read_reports(files: Sequence[Path]) -> pd.DataFrame:
    """Load reports into the dataframe consumed by tables and figures.

    Args:
        files: JSON evaluation reports, each from a distinct training run.

    Returns:
        Long-form scores with REPORT_COLUMNS, one row set per run, ordered by report, aggregation
        level, bucket, and registered metric. Overall rows have a missing length; empty input
        retains the schema.

    Raises:
        ValueError: If a report or its provenance is invalid, or a training run is given twice.
    """
    reports: list[tuple[Path, EvaluationReport]] = []
    for file in files:
        try:
            reports.append((file, EvaluationReport.read(file)))
        except ValidationError as error:
            raise ValueError(f'{file} is not an evaluation report: {error}') from error
    group_runs((report.provenance, file) for file, report in reports)
    rows = [row for _, report in reports for row in _report_rows(report)]
    frame = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    return frame.astype({'length': 'Int64', 'prefixes': 'Int64', 'value': 'float64'})


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize the runs of each model into seed means and sample standard deviations.

    Args:
        frame: Long-form scores from `read_reports`.

    Returns:
        Long-form scores with SUMMARY_COLUMNS, in the order each row first appears in `frame`.
        `std` uses one delta degree of freedom and is NaN for a single run; `runs` counts the
        runs behind each row.

    Raises:
        ValueError: If the runs of one model do not report the same rows with the same prefix
            counts, so they did not score the same prefix population.
    """
    keys = ['dataset', 'model', 'axis', 'length', 'prefixes', 'metric']
    expected = frame.groupby(['dataset', 'model'])['run_id'].nunique().rename('expected')
    rows = frame.groupby(keys, sort=False, dropna=False)['value']
    summary = rows.agg(mean='mean', std='std', runs='count').reset_index()
    mismatched = summary.join(expected, on=['dataset', 'model']).query('runs != expected')
    if not mismatched.empty:
        first = mismatched.iloc[0]
        raise ValueError(
            f'The runs of {first["model"]} on {first["dataset"]} do not score the same prefixes.'
        )
    summary = summary[list(SUMMARY_COLUMNS)]
    return summary.astype({'mean': 'float64', 'std': 'float64', 'runs': 'int64'})
