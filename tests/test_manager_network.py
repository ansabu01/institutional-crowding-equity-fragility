"""Check quarterly manager similarity and community detection."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

import pandas as pd

from src.network.build_manager_similarity import (
    build_manager_similarity_network,
)
from src.network.detect_manager_communities import (
    detect_manager_communities,
)


class ManagerNetworkTests(unittest.TestCase):
    def test_similarity_and_communities_are_quarterly(self):
        holdings, managers = self._inputs()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            holdings_path = root / "holdings.parquet"
            manager_nodes_path = root / "manager_nodes.parquet"
            similarity_directory = root / "similarity"
            community_directory = root / "communities"
            holdings.to_parquet(holdings_path, index=False)
            managers.to_parquet(manager_nodes_path, index=False)

            similarity = build_manager_similarity_network(
                holdings_path,
                manager_nodes_path,
                similarity_directory,
            )
            existing_similarity = build_manager_similarity_network(
                holdings_path,
                manager_nodes_path,
                similarity_directory,
            )
            edges_path = (
                similarity_directory / "manager_similarity_edges.parquet"
            )
            edges = pd.read_parquet(edges_path)
            similarity_audit = pd.read_parquet(
                similarity_directory / "similarity_graph_audit.parquet"
            )

            communities = detect_manager_communities(
                manager_nodes_path,
                edges_path,
                community_directory,
            )
            existing_communities = detect_manager_communities(
                manager_nodes_path,
                edges_path,
                community_directory,
            )
            assignments = pd.read_parquet(
                community_directory / "manager_communities.parquet"
            )
            similarity_output_mtime = min(
                path.stat().st_mtime_ns
                for path in similarity_directory.glob("*.parquet")
            )
            os.utime(
                holdings_path,
                ns=(
                    similarity_output_mtime + 1_000_000,
                    similarity_output_mtime + 1_000_000,
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "newer"):
                build_manager_similarity_network(
                    holdings_path,
                    manager_nodes_path,
                    similarity_directory,
                )

            community_output_mtime = min(
                path.stat().st_mtime_ns
                for path in community_directory.glob("*.parquet")
            )
            os.utime(
                manager_nodes_path,
                ns=(
                    community_output_mtime + 1_000_000,
                    community_output_mtime + 1_000_000,
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "newer"):
                detect_manager_communities(
                    manager_nodes_path,
                    edges_path,
                    community_directory,
                )

        self.assertTrue(similarity.created)
        self.assertFalse(existing_similarity.created)
        self.assertEqual(similarity.report_periods, 1)
        self.assertEqual(len(similarity_audit), 3)
        self.assertTrue(
            edges["manager_a_index"].lt(edges["manager_b_index"]).all()
        )
        self.assertTrue(edges["cosine_similarity"].between(0, 1).all())
        self.assertTrue(communities.created)
        self.assertFalse(existing_communities.created)
        self.assertEqual(len(assignments), len(managers))
        self.assertFalse(assignments["community_id"].isna().any())
        self.assertGreater(assignments["community_id"].nunique(), 1)

    @staticmethod
    def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        period = pd.Timestamp("2020-03-31")
        information_date = pd.Timestamp("2020-05-15")
        manager_rows = []
        holding_rows = []

        for manager_index in range(51):
            cik = str(manager_index + 1)
            group_permno = 10 if manager_index < 25 else 20
            manager_rows.append(
                {
                    "PERIODOFREPORT": "2020-03-31",
                    "report_period": period,
                    "information_date": information_date,
                    "manager_index": manager_index,
                    "CIK": cik,
                    "FILINGMANAGER_NAME": f"Manager {cik}",
                    "portfolio_value_usd": 100,
                    "security_count": 3,
                    "underlying_position_rows": 3,
                    "largest_position_weight": 0.5,
                    "portfolio_hhi": 0.42,
                }
            )

            for permno, weight in (
                (1, 0.1),
                (group_permno, 0.5),
                (1000 + manager_index, 0.4),
            ):
                holding_rows.append(
                    {
                        "CIK": cik,
                        "report_period": period,
                        "permno": permno,
                        "ticker": f"T{permno}",
                        "securitynm": f"Security {permno}",
                        "portfolio_weight": weight,
                    }
                )

        return pd.DataFrame(holding_rows), pd.DataFrame(manager_rows)


if __name__ == "__main__":
    unittest.main()
