"""Build report-ready Chapter 4 tables from saved empirical outputs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / "results" / "tables" / "modeling"
EVALUATION = ROOT / "results" / "tables" / "evaluation"
ROBUSTNESS = ROOT / "results" / "tables" / "robustness"
OUTPUT = ROOT / "results" / "tables"


def _write_table(
    frame: pd.DataFrame,
    filename: str,
    *,
    caption: str,
    label: str,
    column_format: str,
    tabcolsep: str | None = None,
) -> None:
    """Write one complete LaTeX table environment."""
    latex = frame.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        position="!htbp",
        column_format=column_format,
    )
    latex = latex.replace(
        "\\begin{table}[!htbp]\n",
        "\\begin{table}[!htbp]\n\\centering\n\\footnotesize\n",
        1,
    )
    if tabcolsep is not None:
        latex = latex.replace(
            "\\footnotesize\n",
            f"\\footnotesize\n\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n",
            1,
        )
    (OUTPUT / filename).write_text(latex, encoding="utf-8")


def _yes_no(value: object) -> str:
    return "Yes" if bool(value) else "No"


def _prediction_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Feature set": frame["feature_set"],
            "Validation MAE": frame["validation_mae"].map(lambda x: f"{x:.4f}"),
            "$R^2$": frame["validation_r2"].map(lambda x: f"{x:.3f}"),
            "Spearman": frame["mean_quarterly_spearman"].map(lambda x: f"{x:.3f}"),
            "Retained": frame["feature_set"].eq("Market").map(_yes_no),
        }
    )


def build_feature_set_tables() -> None:
    ridge = pd.read_csv(MODELING / "ridge_feature_group_validation_comparison.csv")
    finalist = pd.read_csv(MODELING / "selected_models_validation_comparison.csv")
    market_rmse = finalist.loc[finalist["model"].eq("Ridge"), "rmse"].item()
    market_r2 = ridge.loc[ridge["feature_set"].eq("Market"), "validation_r2"].item()
    ridge["validation_rmse"] = market_rmse * (
        (1 - ridge["validation_r2"]) / (1 - market_r2)
    ) ** 0.5
    ridge_labels = {
        "Market": "MKT",
        "Market + ownership": "MKT + OWN",
        "Market + ownership + network": "MKT + OWN + NTW",
    }
    ridge_table = pd.DataFrame(
        {
            "Specification": ridge["feature_set"].map(ridge_labels),
            "$\\mathrm{MAE}_{\\mathrm{train}}$": ridge["training_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{MAE}_{\\mathrm{val}}$": ridge["validation_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{RMSE}$": ridge["validation_rmse"].map(
                lambda x: f"{x:.4f}"
            ),
            "$R^2$": ridge["validation_r2"].map(
                lambda x: f"{x:.3f}"
            ),
            "$\\overline{\\rho}$": ridge["mean_quarterly_spearman"].map(
                lambda x: f"{x:.3f}"
            ),
            "AIC": ridge["AIC"].map(lambda x: f"{x:,.0f}"),
            "BIC": ridge["BIC"].map(lambda x: f"{x:,.0f}"),
        }
    )
    selected = ridge_table["Specification"].eq("MKT")
    ridge_table.loc[selected] = ridge_table.loc[selected].map(
        lambda value: f"\\textbf{{{value}}}"
    )
    _write_table(
        ridge_table,
        "04_01_ridge_feature_set_validation.tex",
        caption=(
            "Ridge validation performance and training-sample complexity "
            "diagnostics across feature sets."
        ),
        label="tab:ridge-feature-set-validation",
        column_format="lrrrrrrr",
        tabcolsep="4pt",
    )

    xgboost = pd.read_csv(
        MODELING / "xgboost_feature_group_validation_comparison.csv"
    )
    xgboost_table = pd.DataFrame(
        {
            "Specification": xgboost["feature_set"].map(ridge_labels),
            "Best round": xgboost["best_round"].map(lambda x: f"{int(x):,}"),
            "$\\mathrm{MAE}_{\\mathrm{train}}$": xgboost["training_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{MAE}_{\\mathrm{val}}$": xgboost["validation_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{RMSE}$": xgboost["validation_rmse"].map(
                lambda x: f"{x:.4f}"
            ),
            "$R^2$": xgboost["validation_r2"].map(
                lambda x: f"{x:.3f}"
            ),
            "$\\overline{\\rho}$": xgboost["mean_quarterly_spearman"].map(
                lambda x: f"{x:.3f}"
            ),
        }
    )
    selected = xgboost_table["Specification"].eq("MKT")
    xgboost_table.loc[selected] = xgboost_table.loc[selected].map(
        lambda value: f"\\textbf{{{value}}}"
    )
    _write_table(
        xgboost_table,
        "04_02_xgboost_feature_set_validation.tex",
        caption="XGBoost validation performance across feature sets.",
        label="tab:xgboost-feature-set-validation",
        column_format="lrrrrrr",
        tabcolsep="4pt",
    )


def build_tuning_tables() -> None:
    xgboost = pd.read_csv(
        MODELING / "xgboost_controlled_configuration_comparison.csv"
    )
    xgboost_comparison = pd.DataFrame(
        {
            "$\\eta$": xgboost["learning_rate"].map(lambda x: f"{x:.2f}"),
            "Depth": xgboost["max_depth"].map(lambda x: f"{int(x)}"),
            "Best round": xgboost["best_round"].map(lambda x: f"{int(x):,}"),
            "$\\mathrm{MAE}_{\\mathrm{train}}$": xgboost["training_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{MAE}_{\\mathrm{val}}$": xgboost["validation_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{RMSE}$": xgboost["validation_rmse"].map(
                lambda x: f"{x:.4f}"
            ),
            "$R^2$": xgboost["validation_r2"].map(
                lambda x: f"{x:.3f}"
            ),
            "$\\overline{\\rho}$": xgboost["mean_quarterly_spearman"].map(
                lambda x: f"{x:.3f}"
            ),
        }
    )
    selected = xgboost["selected"].astype(bool)
    xgboost_comparison.loc[selected] = xgboost_comparison.loc[selected].map(
        lambda value: f"\\textbf{{{value}}}"
    )
    _write_table(
        xgboost_comparison,
        "04_03_xgboost_tuning_comparison.tex",
        caption="Validation performance of the controlled XGBoost configurations.",
        label="tab:xgboost-tuning-comparison",
        column_format="rrrrrrrr",
        tabcolsep="3.5pt",
    )

    selected_xgboost = xgboost.loc[xgboost["selected"]].copy()
    selected_xgboost = selected_xgboost.iloc[0]
    selected_xgboost = pd.DataFrame(
        {
            "Parameter": [
                "Learning rate",
                "Maximum depth",
                "Minimum child weight",
                "Row subsample",
                "Column subsample",
                "Maximum boosting rounds",
                "Early-stopping patience",
                "Training objective",
                "Validation metric",
                "Selected boosting round",
                "Training MAE",
                "Validation MAE",
            ],
            "Value": [
                f"{selected_xgboost['learning_rate']:.2f}",
                f"{int(selected_xgboost['max_depth'])}",
                "1",
                "0.80",
                "0.80",
                "1,000",
                "30 rounds",
                "Squared error",
                "MAE",
                f"{int(selected_xgboost['best_round'])}",
                f"{selected_xgboost['training_mae']:.4f}",
                f"{selected_xgboost['validation_mae']:.4f}",
            ],
        }
    )
    _write_table(
        selected_xgboost,
        "04_03_selected_xgboost_configuration.tex",
        caption="Selected market-only XGBoost configuration.",
        label="tab:selected-xgboost-configuration",
        column_format="lr",
    )

    architectures = pd.read_csv(
        MODELING / "gnn_architecture_validation_comparison.csv"
    )
    architecture_table = pd.DataFrame(
        {
            "Architecture": architectures["model"],
            "Best epoch": architectures["best_epoch"].map(lambda x: f"{int(x)}"),
            "$\\mathrm{MAE}_{\\mathrm{train}}$": architectures["training_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{MAE}_{\\mathrm{val}}$": architectures["validation_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{RMSE}$": architectures[
                "validation_rmse"
            ].map(lambda x: f"{x:.4f}"),
            "$R^2$": architectures["validation_r2"].map(
                lambda x: f"{x:.3f}"
            ),
            "$\\overline{\\rho}$": architectures[
                "mean_quarterly_spearman"
            ].map(lambda x: f"{x:.3f}"),
        }
    )
    selected = architectures["selected"].astype(bool)
    architecture_table.loc[selected] = architecture_table.loc[selected].map(
        lambda value: f"\\textbf{{{value}}}"
    )
    _write_table(
        architecture_table,
        "04_04_gnn_architecture_comparison.tex",
        caption="Validation performance of the three GNN architectures.",
        label="tab:gnn-architecture-comparison",
        column_format="lrrrrrr",
        tabcolsep="4pt",
    )

    temporal = pd.read_csv(
        MODELING / "temporal_graphsage_configuration_comparison.csv"
    )
    temporal_table = pd.DataFrame(
        {
            "Hidden": temporal["hidden_dimension"].map(lambda x: f"{int(x)}"),
            "$\\eta$": temporal["learning_rate"].map(lambda x: f"{x:.3f}"),
            "Dropout": temporal["dropout"].map(lambda x: f"{x:.2f}"),
            "Best epoch": temporal["best_epoch"].map(lambda x: f"{int(x)}"),
            "$\\mathrm{MAE}_{\\mathrm{train}}$": temporal["training_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{MAE}_{\\mathrm{val}}$": temporal["validation_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{RMSE}$": temporal["validation_rmse"].map(
                lambda x: f"{x:.4f}"
            ),
            "$R^2$": temporal["validation_r2"].map(
                lambda x: f"{x:.3f}"
            ),
            "$\\overline{\\rho}$": temporal[
                "mean_quarterly_spearman"
            ].map(lambda x: f"{x:.3f}"),
        }
    )
    selected = temporal["selected"].astype(bool)
    temporal_table.loc[selected] = temporal_table.loc[selected].map(
        lambda value: f"\\textbf{{{value}}}"
    )
    _write_table(
        temporal_table,
        "04_05_temporal_graphsage_tuning_comparison.tex",
        caption="Validation performance of the Temporal GraphSAGE configurations.",
        label="tab:temporal-graphsage-tuning-comparison",
        column_format="rrrrrrrrr",
        tabcolsep="2.5pt",
    )

    selected_temporal = temporal.loc[
        (temporal["hidden_dimension"] == 32)
        & (temporal["learning_rate"] == 0.001)
        & (temporal["dropout"] == 0.20)
    ].copy()
    selected_temporal = selected_temporal.iloc[0]
    selected_temporal = pd.DataFrame(
        {
            "Parameter": [
                "Architecture",
                "Hidden dimension",
                "Learning rate",
                "Dropout",
                "Weight decay",
                "Optimizer",
                "Training loss",
                "Maximum epochs",
                "Early-stopping patience",
                "Minimum improvement",
                "Selected epoch",
                "Training MAE",
                "Validation MAE",
            ],
            "Value": [
                "Temporal GraphSAGE",
                f"{int(selected_temporal['hidden_dimension'])}",
                f"{selected_temporal['learning_rate']:.3f}",
                f"{selected_temporal['dropout']:.2f}",
                "0.0001",
                "Adam",
                "MAE",
                "50",
                "5 epochs",
                "0.0001",
                f"{int(selected_temporal['best_epoch'])}",
                f"{selected_temporal['training_mae']:.4f}",
                f"{selected_temporal['validation_mae']:.4f}",
            ],
        }
    )
    _write_table(
        selected_temporal,
        "04_04_selected_temporal_graphsage_configuration.tex",
        caption="Selected Temporal GraphSAGE configuration.",
        label="tab:selected-temporal-graphsage-configuration",
        column_format="lr",
    )


def build_model_comparison_tables() -> None:
    validation = pd.read_csv(MODELING / "selected_models_validation_comparison.csv")
    validation = pd.DataFrame(
        {
            "Model": validation["model"],
            "$\\mathrm{MAE}_{\\mathrm{train}}$": validation["training_mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{MAE}_{\\mathrm{val}}$": validation["mae"].map(
                lambda x: f"{x:.4f}"
            ),
            "$\\mathrm{RMSE}$": validation["rmse"].map(lambda x: f"{x:.4f}"),
            "$R^2$": validation["r2"].map(lambda x: f"{x:.3f}"),
            "$\\overline{\\rho}$": validation["mean_quarterly_spearman"].map(
                lambda x: f"{x:.3f}"
            ),
        }
    )
    selected = validation["Model"].eq("XGBoost")
    validation.loc[selected] = validation.loc[selected].map(
        lambda value: f"\\textbf{{{value}}}"
    )
    _write_table(
        validation,
        "04_05_finalist_validation_performance.tex",
        caption="Validation performance of the three finalist models.",
        label="tab:finalist-validation-performance",
        column_format="lrrrrr",
    )

    test = pd.read_csv(MODELING / "final_models_test_performance.csv")
    benchmark = pd.read_csv(MODELING / "predictive_benchmark_performance.csv")
    benchmark = benchmark.loc[benchmark["sample_split"].eq("test")].copy()
    benchmark["model"] = "No-information"
    benchmark["mean_quarterly_spearman"] = float("nan")
    benchmark["selected_model"] = False
    test = pd.concat(
        [benchmark[test.columns], test],
        ignore_index=True,
    )
    model_order = ["No-information", "Ridge", "XGBoost", "Temporal GraphSAGE"]
    test["model"] = pd.Categorical(
        test["model"], categories=model_order, ordered=True
    )
    test = test.sort_values("model")
    test = pd.DataFrame(
        {
            "Model": test["model"].astype(str),
            "$\\mathrm{MAE}$": test["mae"].map(lambda x: f"{x:.4f}"),
            "$\\mathrm{RMSE}$": test["rmse"].map(lambda x: f"{x:.4f}"),
            "$R^2$": test["r2"].map(lambda x: f"{x:.3f}"),
            "$\\overline{\\rho}$": test["mean_quarterly_spearman"].map(
                lambda x: "---" if pd.isna(x) else f"{x:.3f}"
            ),
        }
    )
    _write_table(
        test,
        "04_06_finalist_test_performance.tex",
        caption=(
            "Held-out test performance of the no-information benchmark and "
            "frozen finalist models."
        ),
        label="tab:finalist-test-performance",
        column_format="lrrrr",
    )


def build_diagnostic_tables() -> None:
    robustness = pd.read_csv(
        MODELING / "selected_xgboost_prediction_robustness_report.csv"
    )
    robustness = pd.DataFrame(
        {
            "Target": robustness["target"],
            "Quarters": robustness["test_quarters"].map(lambda x: f"{int(x)}"),
            "$\\mathrm{MAE}$": robustness["mae"].map(lambda x: f"{x:.4f}"),
            "$\\mathrm{RMSE}$": robustness["rmse"].map(lambda x: f"{x:.4f}"),
            "$R^2$": robustness["r2"].map(lambda x: f"{x:.3f}"),
            "$\\overline{\\rho}$": robustness["mean_quarterly_spearman"].map(
                lambda x: f"{x:.3f}"
            ),
            "D10--D1 spread": robustness["top_bottom_spread"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
        }
    )
    _write_table(
        robustness,
        "04_07_predictive_robustness.tex",
        caption="Held-out predictive robustness of the selected XGBoost specification.",
        label="tab:predictive-robustness",
        column_format="lrrrrrr",
    )

    ranks = pd.read_csv(EVALUATION / "fragility_rank_correlations.csv")
    summary = pd.DataFrame(
        {
            "Statistic": [
                "Mean stock-level $\\rho$: score vs. realized drawdown",
                "Mean decile-level $\\rho$: decile vs. realized drawdown",
                "Mean realized-drawdown spread: Decile 10 minus Decile 1",
            ],
            "Value": [
                f"{ranks['stock_spearman'].mean():.3f}",
                f"{ranks['decile_spearman'].mean():.3f}",
                f"{100 * ranks['top_minus_bottom_mean_drawdown'].mean():.1f}\\%",
            ],
        }
    )
    _write_table(
        summary,
        "04_08_fragility_score_summary.tex",
        caption="Held-out Fragility Score ranking performance.",
        label="tab:fragility-score-performance",
        column_format="lr",
    )


def build_portfolio_tables() -> None:
    event_windows = pd.read_csv(EVALUATION / "fragility_portfolio_quarterly.csv")
    event_windows = (
        event_windows.groupby("portfolio", as_index=False)
        .agg(
            mean_maximum_drawdown=("maximum_drawdown", "mean"),
            mean_downside_volatility=("downside_volatility", "mean"),
            mean_worst_five_day_loss=("worst_five_day_loss", "mean"),
            mean_cumulative_return=("cumulative_return", "mean"),
        )
    )
    labels = {
        "all_stocks": "All eligible stocks",
        "exclude_most_fragile_decile": "Exclude Decile 10",
        "least_fragile_decile": "Decile 1",
        "most_fragile_decile": "Decile 10",
    }
    event_windows = pd.DataFrame(
        {
            "Portfolio": event_windows["portfolio"].map(labels),
            "Max. drawdown": event_windows["mean_maximum_drawdown"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Downside vol.": event_windows["mean_downside_volatility"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Worst 5-day loss": event_windows["mean_worst_five_day_loss"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Cum. return": event_windows["mean_cumulative_return"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
        }
    )
    _write_table(
        event_windows,
        "04_09_portfolio_event_windows.tex",
        caption="Mean performance across quarterly 63-day portfolio windows.",
        label="tab:portfolio-event-windows",
        column_format="lrrrr",
    )

    performance = pd.read_csv(EVALUATION / "fragility_strategy_performance.csv")
    strategy_labels = {
        "all_stocks": "All eligible stocks",
        "exclude_most_fragile_decile": "Exclude Decile 10",
        "crsp_total_market": "CRSP total market",
    }
    performance = pd.DataFrame(
        {
            "Strategy": performance["strategy"].map(strategy_labels),
            "Cum. return": performance["cumulative_return"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Ann. return": performance["annualized_return"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Ann. vol.": performance["annualized_volatility"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Sharpe": performance["sharpe_ratio"].map(lambda x: f"{x:.2f}"),
            "Max. drawdown": performance["maximum_drawdown"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
        }
    )
    _write_table(
        performance,
        "04_10_portfolio_performance.tex",
        caption="Continuous Fragility Score portfolio performance.",
        label="tab:portfolio-performance",
        column_format="lrrrrr",
    )

    alpha = pd.read_csv(EVALUATION / "fragility_strategy_alpha.csv")
    alpha = alpha.loc[alpha["model"].eq("ff5_momentum")].copy()
    alpha_labels = {
        "all_stocks": "All eligible stocks",
        "exclude_most_fragile_decile": "Exclude Decile 10",
        "screened_minus_all_stocks": "Screened minus all stocks",
    }
    alpha = pd.DataFrame(
        {
            "Portfolio": alpha["portfolio"].map(alpha_labels),
            "Annualized alpha": alpha["annualized_alpha"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "$t$-statistic": alpha["alpha_t_statistic"].map(
                lambda x: f"{x:.2f}"
            ),
            "$p$-value": alpha["alpha_p_value"].map(lambda x: f"{x:.3f}"),
            "Adjusted $R^2$": alpha["adjusted_r_squared"].map(
                lambda x: f"{x:.3f}"
            ),
        }
    )
    _write_table(
        alpha,
        "04_11_portfolio_alpha.tex",
        caption="FF5-plus-momentum alpha for the primary portfolio comparison.",
        label="tab:portfolio-alpha",
        column_format="lrrrr",
    )

    robustness = pd.read_csv(
        ROBUSTNESS / "investment_robustness_performance.csv"
    )
    robustness_labels = {
        "equal_all": "Equal weight: all stocks",
        "equal_exclude_d10": "Equal weight: exclude Decile 10",
        "equal_exclude_top20": "Equal weight: exclude top 20\\%",
        "value_all": "Value weight: all stocks",
        "value_exclude_d10": "Value weight: exclude Decile 10",
    }
    robustness = pd.DataFrame(
        {
            "Strategy": robustness["strategy"].map(robustness_labels),
            "Ann. return": robustness["annualized_return"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Ann. vol.": robustness["annualized_volatility"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Sharpe": robustness["sharpe_ratio"].map(lambda x: f"{x:.2f}"),
            "Max. drawdown": robustness["maximum_drawdown"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
        }
    )
    _write_table(
        robustness,
        "04_12_portfolio_robustness.tex",
        caption="Long-only portfolio robustness results.",
        label="tab:portfolio-robustness",
        column_format="lrrrr",
    )

    long_short = pd.read_csv(
        ROBUSTNESS / "fragility_factor_risk_management_performance.csv"
    )
    long_short = long_short.loc[
        long_short["strategy"].isin(
            ["original", "dfrag_20", "volatility_managed"]
        )
    ].copy()
    long_short_labels = {
        "original": "DFRAG-10",
        "dfrag_20": "DFRAG-20",
        "volatility_managed": "DFRAG-10: volatility managed",
    }
    long_short = pd.DataFrame(
        {
            "Strategy": long_short["strategy"].map(long_short_labels),
            "Ann. return": long_short["annualized_return"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Ann. volatility": long_short["annualized_volatility"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Sharpe": long_short["sharpe_ratio"].map(lambda x: f"{x:.2f}"),
            "Max. drawdown": long_short["maximum_drawdown"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
        }
    )
    _write_table(
        long_short,
        "04_13_long_short_performance.tex",
        caption="Exploratory long--short Fragility Score strategies.",
        label="tab:long-short-performance",
        column_format="lrrrr",
    )

    turnover = pd.read_csv(ROBUSTNESS / "investment_robustness_turnover.csv")
    turnover = (
        turnover.loc[
            turnover["strategy"].isin(
                ["dfrag_10", "dfrag_20", "volatility_managed"]
            )
        ]
        .groupby("strategy", as_index=False)
        .agg(mean_turnover=("one_way_turnover", "mean"))
    )
    all_costs = pd.read_csv(ROBUSTNESS / "investment_robustness_costs.csv")
    reported_strategies = ["dfrag_10", "dfrag_20", "volatility_managed"]
    costs = all_costs.loc[
        all_costs["strategy"].isin(reported_strategies)
        & all_costs["cost_bps"].eq(50),
        ["strategy", "annualized_return"],
    ].rename(columns={"annualized_return": "return_after_50_bps"})
    gross = pd.read_csv(
        ROBUSTNESS / "fragility_factor_risk_management_performance.csv"
    ).loc[:, ["strategy", "annualized_return"]]
    gross["strategy"] = gross["strategy"].replace({"original": "dfrag_10"})
    gross = gross.loc[gross["strategy"].isin(reported_strategies)]
    gross = gross.rename(columns={"annualized_return": "gross_return"})
    implementation = turnover.merge(gross, on="strategy").merge(costs, on="strategy")
    implementation_labels = {
        "dfrag_10": "DFRAG-10",
        "dfrag_20": "DFRAG-20",
        "volatility_managed": "DFRAG-10: volatility managed",
    }
    implementation = pd.DataFrame(
        {
            "Strategy": implementation["strategy"].map(implementation_labels),
            "Mean turnover": implementation["mean_turnover"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Gross return": implementation["gross_return"].map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "Net return (50 bps)": implementation[
                "return_after_50_bps"
            ].map(lambda x: f"{100 * x:.1f}\\%"),
        }
    )
    _write_table(
        implementation,
        "04_14_long_short_implementation.tex",
        caption=(
            "Mean one-way turnover and annualized gross and net returns of the "
            "long--short strategies."
        ),
        label="tab:long-short-implementation",
        column_format="lrrr",
    )

    cost_sensitivity = (
        all_costs.loc[all_costs["strategy"].isin(reported_strategies)]
        .pivot(index="strategy", columns="cost_bps", values="annualized_return")
        .reindex(reported_strategies)
        .reset_index()
    )
    gross_lookup = gross.set_index("strategy")["gross_return"]
    cost_sensitivity = pd.DataFrame(
        {
            "Strategy": cost_sensitivity["strategy"].map(implementation_labels),
            "Gross": cost_sensitivity["strategy"].map(gross_lookup).map(
                lambda x: f"{100 * x:.1f}\\%"
            ),
            "10 bps": cost_sensitivity[10].map(lambda x: f"{100 * x:.1f}\\%"),
            "25 bps": cost_sensitivity[25].map(lambda x: f"{100 * x:.1f}\\%"),
            "50 bps": cost_sensitivity[50].map(lambda x: f"{100 * x:.1f}\\%"),
        }
    )
    _write_table(
        cost_sensitivity,
        "04_15_long_short_cost_sensitivity.tex",
        caption=(
            "Annualized returns of the long--short strategies under alternative "
            "one-way transaction-cost assumptions."
        ),
        label="tab:long-short-cost-sensitivity",
        column_format="lrrrr",
    )


def build_hypothesis_table() -> None:
    hypotheses = pd.DataFrame(
        {
            "Hypothesis": ["H1", "H2", "H3", "H4", "H5"],
            "Evidence": [
                "Ownership features produce only small validation improvements.",
                "Engineered network features add no consistent incremental gain.",
                "XGBoost modestly improves on Ridge with identical market inputs.",
                "Temporal GraphSAGE is competitive, but not consistently superior across metrics.",
                "The score strongly ranks downside risk; portfolio evidence is positive but qualified.",
            ],
            "Assessment": [
                "Limited support",
                "Not supported",
                "Moderate support",
                "Mixed evidence",
                "Qualified support",
            ],
        }
    )
    _write_table(
        hypotheses,
        "04_15_hypothesis_summary.tex",
        caption="Summary of evidence for the research hypotheses.",
        label="tab:hypothesis-summary",
        column_format="lp{0.58\\textwidth}l",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_feature_set_tables()
    build_tuning_tables()
    build_model_comparison_tables()
    build_diagnostic_tables()
    build_portfolio_tables()
    build_hypothesis_table()


if __name__ == "__main__":
    main()
