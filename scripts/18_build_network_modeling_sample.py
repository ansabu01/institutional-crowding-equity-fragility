"""Build the modeling sample with within-quarter network-feature ranks."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.modeling.network_sample import build_network_modeling_sample  # noqa: E402
from src.utils.configuration import get_paths_config  # noqa: E402


_KEY = ["report_period", "information_date", "permno"]
_FEATURE_GROUPS = {
    "Market": (
        ("market_cap_usd", "Market capitalization"),
        ("average_daily_dollar_volume_63d", "Average daily dollar volume"),
        ("return_21d", "21-day return"),
        ("momentum_252_21d", "Skip-month momentum"),
        ("volatility_63d", "Volatility"),
        ("downside_volatility_63d", "Downside volatility"),
        ("market_beta_252d", "Market beta"),
        ("maximum_drawdown_252d", "Past maximum drawdown"),
        ("negative_return_share_63d", "Negative-return share"),
    ),
    "Ownership": (
        ("institutional_holder_count", "Institutional holder count"),
        ("total_reported_value_usd", "Total reported institutional value"),
        ("reported_ownership_hhi", "Ownership HHI"),
        ("top_five_reported_ownership_share", "Top-five ownership share"),
        ("mean_owner_portfolio_weight", "Mean owner portfolio weight"),
        ("maximum_owner_portfolio_weight", "Maximum owner portfolio weight"),
        ("manager_holder_share", "Manager-holder share"),
        (
            "institutional_holder_count_change_percent",
            "Change in institutional holder count",
        ),
        (
            "total_reported_value_change_percent",
            "Change in total reported institutional value",
        ),
        ("reported_ownership_hhi_change", "Change in ownership HHI"),
        (
            "top_five_reported_ownership_share_change",
            "Change in top-five ownership share",
        ),
    ),
    "Network": (
        ("owner_similarity_score", "Owner similarity"),
        (
            "value_weighted_owner_degree_centrality",
            "Owner degree centrality",
        ),
        (
            "value_weighted_owner_neighbor_similarity",
            "Owner-neighbour similarity",
        ),
        ("owner_community_count", "Owner community count"),
        ("owner_community_hhi", "Owner-community HHI"),
        ("largest_owner_community_share", "Largest owner-community share"),
        ("owner_similarity_score_change", "Change in owner similarity"),
        ("owner_community_hhi_change", "Change in owner-community HHI"),
    ),
}
_SPLITS = ("train", "validation", "test")


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build the modeling sample with ranked network features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace the existing network modeling sample.",
    )

    return parser.parse_args()


def _write_feature_missingness_table(
    sample_path: Path,
    network_feature_path: Path,
    output_path: Path,
) -> None:
    """Write missing shares for every candidate predictor and sample split."""
    traditional_features = [
        feature
        for group in ("Market", "Ownership")
        for feature, _ in _FEATURE_GROUPS[group]
    ]
    network_features = [
        feature for feature, _ in _FEATURE_GROUPS["Network"]
    ]
    sample = pd.read_parquet(
        sample_path,
        columns=[*_KEY, "sample_split", *traditional_features],
    )
    network = pd.read_parquet(
        network_feature_path,
        columns=[*_KEY, *network_features],
    )
    data = sample.merge(
        network,
        on=_KEY,
        how="left",
        sort=False,
        validate="one_to_one",
    )

    observed_splits = set(data["sample_split"].dropna().unique())
    missing_splits = set(_SPLITS) - observed_splits
    if missing_splits:
        raise RuntimeError(
            f"Missing sample splits for the feature audit: {sorted(missing_splits)}"
        )

    missing_shares = {
        split: data.loc[data["sample_split"].eq(split)].isna().mean()
        for split in _SPLITS
    }
    row_ending = r"\\"
    rows = []
    for group, features in _FEATURE_GROUPS.items():
        for feature, label in features:
            shares = [100 * missing_shares[split][feature] for split in _SPLITS]
            rows.append(
                f"        {group} & {label} & "
                f"{shares[0]:.2f}\\% & {shares[1]:.2f}\\% & "
                f"{shares[2]:.2f}\\% {row_ending}"
            )

    table = "\n".join(
        [
            r"\begingroup",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\begin{longtable}{p{0.14\textwidth}p{0.40\textwidth}rrr}",
            r"    \caption{Missingness across candidate predictors and modeling samples}",
            r"    \label{tab:feature-missingness} \\",
            r"    \toprule",
            r"    Feature group & Feature & \multicolumn{3}{c}{Missing share} \\",
            r"    \cmidrule(lr){3-5}",
            r"                  &         & Training & Validation & Test \\",
            r"    \midrule",
            r"    \endfirsthead",
            r"    \multicolumn{5}{l}{\footnotesize\itshape Table~\thetable\ continued} \\",
            r"    \toprule",
            r"    Feature group & Feature & \multicolumn{3}{c}{Missing share} \\",
            r"    \cmidrule(lr){3-5}",
            r"                  &         & Training & Validation & Test \\",
            r"    \midrule",
            r"    \endhead",
            r"    \midrule",
            r"    \multicolumn{5}{r}{\footnotesize\itshape Continued on next page} \\",
            r"    \endfoot",
            r"    \bottomrule",
            r"    \endlastfoot",
            *rows,
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")


def main(*, overwrite: bool = False) -> None:
    """Create or validate the network modeling sample."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    modeling_directory = paths.processed / "modeling"
    sample_path = modeling_directory / "modeling_sample.parquet"
    network_feature_path = (
        paths.interim / "networks" / "stock_network_features.parquet"
    )
    result = build_network_modeling_sample(
        sample_path=sample_path,
        feature_path=network_feature_path,
        output_path=modeling_directory / "network_modeling_sample.parquet",
        overwrite=overwrite,
    )
    report_table_path = paths.tables / "A_06_feature_missingness.tex"
    _write_feature_missingness_table(
        sample_path,
        network_feature_path,
        report_table_path,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Network modeling sample: {status}")
    print(f"Rows: {result.rows:,}")
    print(f"Quarters: {result.quarters:,}")
    print(f"Ranked network features: {result.ranked_features}")
    print(f"Missingness table: {report_table_path}")
    print("Network modeling sample: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
