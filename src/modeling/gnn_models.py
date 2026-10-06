"""Small bipartite graph models for stock-level downside prediction."""

import torch
from torch import nn
from torch_geometric.nn import GATConv


class _WeightedSageLayer(nn.Module):
    """Combine a node representation with a weighted neighbor mean."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_size, hidden_size)
        self.neighbor_linear = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Return updated destination-node representations."""
        messages = source[source_index] * weight.unsqueeze(1)
        aggregated = source.new_zeros((len(destination), source.shape[1]))
        aggregated.index_add_(0, destination_index, messages)
        return self.self_linear(destination) + self.neighbor_linear(aggregated)


class BipartiteGraphSageEncoder(nn.Module):
    """Encode stocks through weighted stock--manager--stock messages."""

    def __init__(
        self,
        stock_features: int,
        manager_features: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stock_input = nn.Linear(stock_features, hidden_size)
        self.manager_input = nn.Linear(manager_features, hidden_size)
        self.to_manager = _WeightedSageLayer(hidden_size)
        self.to_stock = _WeightedSageLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        stock_features: torch.Tensor,
        manager_features: torch.Tensor,
        edge_index: torch.Tensor,
        portfolio_weight: torch.Tensor,
        ownership_share: torch.Tensor,
        edge_attributes: torch.Tensor,
    ) -> torch.Tensor:
        """Return one embedding for every stock node."""
        del edge_attributes
        manager_index, stock_index = edge_index
        stocks = torch.relu(self.stock_input(stock_features))
        managers = torch.relu(self.manager_input(manager_features))
        managers = torch.relu(
            self.to_manager(
                stocks,
                managers,
                stock_index,
                manager_index,
                portfolio_weight,
            )
        )
        stocks = torch.relu(
            self.to_stock(
                managers,
                stocks,
                manager_index,
                stock_index,
                ownership_share,
            )
        )
        return self.dropout(stocks)


class BipartiteGatEncoder(nn.Module):
    """Encode stocks with two single-head bipartite attention layers."""

    def __init__(
        self,
        stock_features: int,
        manager_features: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stock_input = nn.Linear(stock_features, hidden_size)
        self.manager_input = nn.Linear(manager_features, hidden_size)
        self.to_manager = GATConv(
            (hidden_size, hidden_size),
            hidden_size,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=2,
            add_self_loops=False,
        )
        self.to_stock = GATConv(
            (hidden_size, hidden_size),
            hidden_size,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=2,
            add_self_loops=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        stock_features: torch.Tensor,
        manager_features: torch.Tensor,
        edge_index: torch.Tensor,
        portfolio_weight: torch.Tensor,
        ownership_share: torch.Tensor,
        edge_attributes: torch.Tensor,
    ) -> torch.Tensor:
        """Return one embedding for every stock node."""
        del portfolio_weight, ownership_share
        manager_index, stock_index = edge_index
        stocks = torch.relu(self.stock_input(stock_features))
        managers = torch.relu(self.manager_input(manager_features))
        reverse_edges = torch.stack((stock_index, manager_index))
        manager_messages = self.to_manager(
            (stocks, managers),
            reverse_edges,
            edge_attributes,
        )
        managers = torch.relu(managers + manager_messages)
        stock_messages = self.to_stock(
            (managers, stocks),
            edge_index,
            edge_attributes,
        )
        stocks = torch.relu(stocks + stock_messages)
        return self.dropout(stocks)


class StaticGnn(nn.Module):
    """Predict downside risk from one quarterly graph embedding."""

    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, *graph_inputs: torch.Tensor) -> torch.Tensor:
        """Return one prediction for every stock node."""
        return self.output(self.encoder(*graph_inputs)).squeeze(1)


class TemporalGraphSage(nn.Module):
    """Update stock states over quarterly GraphSAGE embeddings."""

    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.recurrence = nn.GRUCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        previous_state: torch.Tensor,
        *graph_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stock predictions and updated temporal states."""
        graph_embedding = self.encoder(*graph_inputs)
        current_state = self.recurrence(graph_embedding, previous_state)
        predictions = self.output(current_state).squeeze(1)
        return predictions, current_state


def build_gnn(
    architecture: str,
    stock_features: int,
    manager_features: int,
    hidden_size: int,
    dropout: float,
) -> nn.Module:
    """Construct one frozen graph architecture."""
    if architecture in {"graphsage", "temporal_graphsage"}:
        encoder: nn.Module = BipartiteGraphSageEncoder(
            stock_features,
            manager_features,
            hidden_size,
            dropout,
        )
    elif architecture == "gat":
        encoder = BipartiteGatEncoder(
            stock_features,
            manager_features,
            hidden_size,
            dropout,
        )
    else:
        raise ValueError(f"Unknown GNN architecture: {architecture}")

    if architecture == "temporal_graphsage":
        return TemporalGraphSage(encoder, hidden_size)
    return StaticGnn(encoder, hidden_size)


__all__ = ["build_gnn"]
