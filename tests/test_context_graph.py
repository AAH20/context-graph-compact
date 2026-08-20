import unittest
from context_graph_compact.tri_tier_memory import TriTierMemoryEngine
from context_graph_compact.hebbian import HebbianSemanticGraph, SemanticTriple
from context_graph_compact.evaluator import SelfEvolutionOracle

class TestContextGraphCompact(unittest.TestCase):
    def setUp(self):
        self.engine = TriTierMemoryEngine(working_window_size=3)

    def test_lossless_variable_recall_across_30_turns(self):
        # Turn 1 declares critical port and user
        self.engine.add_turn("user", "Please configure the production PostgreSQL database on port 5433 with user readonly_app.")
        self.engine.add_turn("assistant", "Configuring PostgreSQL on port 5433.")

        # Simulate 15 turns of verbose conversation noise
        for i in range(15):
            self.engine.add_turn("user", f"Executing routine telemetry check #{i} for container pods.")
            self.engine.add_turn("assistant", f"Container health check #{i} passed with 0 errors.")

        # Compile compact context
        context_text, meta = self.engine.compile_compact_context()

        # Invariant 1: Port 5433 and readonly_app MUST be preserved in Tier 3 invariants
        self.assertIn("5433", context_text)
        self.assertIn("readonly_app", context_text)
        self.assertIn("TIER 3: PERMANENT SEMANTIC INVARIANTS", context_text)

        # Invariant 2: Token compression ratio must be positive
        self.assertGreater(meta["compression_ratio"], 0.20)

    def test_hebbian_synapse_reinforcement_and_decay(self):
        graph = HebbianSemanticGraph(learning_rate=0.2, decay_rate=0.05)
        triple = SemanticTriple(subject="PaymentGateway", predicate="timeout_ms", object_="3500")

        # Initial insertion
        edge1 = graph.insert_or_reinforce(triple, reinforcement_boost=1.0)
        self.assertEqual(edge1.weight, 1.2)

        # Reinforce twice after successful agent decisions
        graph.insert_or_reinforce(triple, reinforcement_boost=1.0)
        edge2 = graph.insert_or_reinforce(triple, reinforcement_boost=1.0)
        self.assertEqual(edge2.weight, 1.6)

        # Apply 10-day temporal decay
        graph.apply_temporal_decay(elapsed_days=10.0)
        self.assertLess(edge2.weight, 1.6)
        self.assertGreater(edge2.weight, 0.5)

    def test_self_evolution_oracle_evaluation(self):
        self.engine.add_turn("user", "Set target table to customer_invoices on port 5432.")
        oracle = SelfEvolutionOracle(self.engine)
        report = oracle.run_self_evaluation()

        # Invariant: 100% factual recall score
        self.assertEqual(report.factual_recall_score, 1.0)
        self.assertTrue(report.passed)

if __name__ == "__main__":
    unittest.main()
