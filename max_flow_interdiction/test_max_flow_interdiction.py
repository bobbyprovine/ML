import unittest

from max_flow_interdiction import (
    Edge,
    SAMPLE_EDGES,
    exact_interdiction,
    max_flow,
    train_q_learning,
)


class MaxFlowInterdictionTests(unittest.TestCase):
    def test_max_flow_on_three_parallel_routes(self):
        self.assertEqual(max_flow(SAMPLE_EDGES, "s", "t"), 6)

    def test_exact_interdiction_finds_optimal_flow(self):
        result = exact_interdiction(SAMPLE_EDGES, "s", "t", budget=2)
        self.assertEqual(result.remaining_flow, 1)
        self.assertEqual(len(result.removed_indices), 2)

    def test_q_learning_matches_exact_optimum_and_respects_budget(self):
        learned = train_q_learning(
            SAMPLE_EDGES, "s", "t", budget=2, episodes=20_000, seed=7
        )
        exact = exact_interdiction(SAMPLE_EDGES, "s", "t", budget=2)
        self.assertEqual(learned.remaining_flow, exact.remaining_flow)
        self.assertEqual(len(learned.removed_indices), 2)
        self.assertEqual(len(set(learned.removed_indices)), 2)

    def test_invalid_capacity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            max_flow((Edge("s", "t", 0),), "s", "t")

    def test_negative_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            exact_interdiction(SAMPLE_EDGES, "s", "t", budget=-1)


if __name__ == "__main__":
    unittest.main()
