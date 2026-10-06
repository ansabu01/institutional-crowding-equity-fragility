"""Shared feature sets after the frozen training-sample redundancy filter."""


MARKET_FEATURES = (
    "market_cap_usd",
    "return_21d",
    "momentum_252_21d",
    "downside_volatility_63d",
    "market_beta_252d",
    "maximum_drawdown_252d",
    "negative_return_share_63d",
)
OWNERSHIP_FEATURES = (
    "reported_ownership_hhi",
    "mean_owner_portfolio_weight",
    "institutional_holder_count_change_percent",
    "total_reported_value_change_percent",
    "reported_ownership_hhi_change",
)
NETWORK_FEATURES = (
    "owner_similarity_score",
    "value_weighted_owner_degree_centrality",
    "value_weighted_owner_neighbor_similarity",
    "owner_community_hhi",
    "owner_similarity_score_change",
    "owner_community_hhi_change",
)
RANKED_NETWORK_FEATURES = tuple(
    f"{feature}_rank" for feature in NETWORK_FEATURES
)
STOCK_GNN_FEATURES = MARKET_FEATURES + OWNERSHIP_FEATURES
MANAGER_GNN_FEATURES = (
    "portfolio_value_usd",
    "security_count",
    "largest_position_weight",
    "portfolio_hhi",
)
MARKET_LOG_FEATURES = frozenset(
    {
        "market_cap_usd",
    }
)
STOCK_LOG_FEATURES = MARKET_LOG_FEATURES
MANAGER_LOG_FEATURES = frozenset({"portfolio_value_usd", "security_count"})


__all__ = [
    "MANAGER_GNN_FEATURES",
    "MANAGER_LOG_FEATURES",
    "MARKET_FEATURES",
    "MARKET_LOG_FEATURES",
    "NETWORK_FEATURES",
    "OWNERSHIP_FEATURES",
    "RANKED_NETWORK_FEATURES",
    "STOCK_GNN_FEATURES",
    "STOCK_LOG_FEATURES",
]
