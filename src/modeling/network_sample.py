"""Add within-quarter network-feature ranks to the modeling sample."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .feature_sets import (
    NETWORK_FEATURES as _NETWORK_FEATURES,
    RANKED_NETWORK_FEATURES as _RANKED_FEATURES,
)

_KEY = ("report_period", "information_date", "permno")
_STATIC_RANKED_FEATURES = tuple(
    feature
    for feature in _RANKED_FEATURES
    if not feature.endswith("_change_rank")
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NetworkSampleBuildResult:
    """Result of building or validating the network modeling sample."""

    created: bool
    output_path: Path
    rows: int
    quarters: int
    ranked_features: int


def _load_inputs(
    sample_path: Path,
    feature_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the modeling sample and raw network features."""
    if not sample_path.is_file():
        raise FileNotFoundError(f"Modeling sample not found: {sample_path}")

    if not feature_path.is_file():
        raise FileNotFoundError(f"Network features not found: {feature_path}")

    sample = pd.read_parquet(sample_path)
    required_sample_columns = {*_KEY, "sample_split"}
    missing_sample_columns = required_sample_columns - set(sample.columns)

    if missing_sample_columns:
        raise RuntimeError(
            f"Missing modeling-sample columns: {sorted(missing_sample_columns)}"
        )

    if set(_RANKED_FEATURES).intersection(sample.columns):
        raise RuntimeError("The modeling sample already contains ranked network features.")

    features = pd.read_parquet(
        feature_path,
        columns=[*_KEY, *_NETWORK_FEATURES],
    )

    if sample.duplicated(list(_KEY)).any():
        raise RuntimeError("The modeling sample contains duplicate stock-quarters.")

    if features.duplicated(list(_KEY)).any():
        raise RuntimeError("The network features contain duplicate stock-quarters.")

    return sample, features


def _rank_network_features(features: pd.DataFrame) -> pd.DataFrame:
    """Rank each network feature within its reporting quarter."""
    grouped = features.groupby("report_period")[list(_NETWORK_FEATURES)]
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    percentile_ranks = (ranks - 1) / (counts - 1)
    percentile_ranks = percentile_ranks.where(counts.ne(1), 0.5)
    percentile_ranks.columns = _RANKED_FEATURES

    return pd.concat(
        [features.loc[:, list(_KEY)], percentile_ranks],
        axis="columns",
    )


def _validate_output(
    output: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[int, int]:
    """Validate the ranked sample and return its rows and quarters."""
    expected_columns = [*sample.columns, *_RANKED_FEATURES]

    if list(output.columns) != expected_columns:
        raise RuntimeError("The network modeling sample has unexpected columns.")

    if len(output) != len(sample) or output.duplicated(list(_KEY)).any():
        raise RuntimeError("The network modeling sample has inconsistent keys or rows.")

    if not output.loc[:, list(_KEY)].equals(sample.loc[:, list(_KEY)]):
        raise RuntimeError("The network modeling sample changed the original row order.")

    if output[list(_STATIC_RANKED_FEATURES)].isna().any().any():
        raise RuntimeError("Static network ranks contain missing values.")

    ranked_values = output[list(_RANKED_FEATURES)].stack()

    if not ranked_values.between(0, 1).all():
        raise RuntimeError("Network-feature ranks must be between 0 and 1.")

    return len(output), output["report_period"].nunique()


def build_network_modeling_sample(
    sample_path: Path,
    feature_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> NetworkSampleBuildResult:
    """Build or validate the modeling sample with ranked network features."""
    sample_path = Path(sample_path)
    feature_path = Path(feature_path)
    output_path = Path(output_path)
    sample, features = _load_inputs(sample_path, feature_path)

    if output_path.exists() and not overwrite:
        input_paths = (
            sample_path,
            feature_path,
            Path(__file__),
            Path(__file__).with_name("feature_sets.py"),
        )

        if max(path.stat().st_mtime for path in input_paths) > output_path.stat().st_mtime:
            raise RuntimeError(
                "An input or the network-sample code is newer than the "
                "network modeling sample. Rebuild with overwrite=True."
            )

        output = pd.read_parquet(output_path)
        rows, quarters = _validate_output(output, sample)

        return NetworkSampleBuildResult(
            created=False,
            output_path=output_path,
            rows=rows,
            quarters=quarters,
            ranked_features=len(_RANKED_FEATURES),
        )

    _LOGGER.info("Ranking network features within each reporting quarter.")
    ranked_features = _rank_network_features(features)
    _LOGGER.info("Adding ranked network features to the modeling sample.")
    output = sample.merge(
        ranked_features,
        on=list(_KEY),
        how="left",
        sort=False,
        validate="one_to_one",
    )
    rows, quarters = _validate_output(output, sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False, compression="zstd")
    saved_output = pd.read_parquet(output_path)
    rows, quarters = _validate_output(saved_output, sample)

    return NetworkSampleBuildResult(
        created=True,
        output_path=output_path,
        rows=rows,
        quarters=quarters,
        ranked_features=len(_RANKED_FEATURES),
    )


__all__ = ["NetworkSampleBuildResult", "build_network_modeling_sample"]
