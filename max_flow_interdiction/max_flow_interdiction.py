"""Q-learning demo for edge-removal interdiction on a small flow network."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    capacity: int


@dataclass(frozen=True)
class InterdictionResult:
    removed_indices: tuple[int, ...]
    remaining_flow: int


SAMPLE_EDGES = (
    Edge("s", "a", 2),
    Edge("a", "t", 2),
    Edge("s", "b", 3),
    Edge("b", "t", 3),
    Edge("s", "c", 1),
    Edge("c", "t", 1),
)


def _validate_network(
    edges: Sequence[Edge], source: str, sink: str
) -> tuple[Edge, ...]:
    edge_list = tuple(edges)
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    if not isinstance(sink, str) or not sink:
        raise ValueError("sink must be a non-empty string")
    if source == sink:
        raise ValueError("source and sink must be different")

    nodes: set[str] = set()
    for edge in edge_list:
        if not isinstance(edge, Edge):
            raise TypeError("edges must contain Edge instances")
        if not isinstance(edge.source, str) or not edge.source:
            raise ValueError("edge source must be a non-empty string")
        if not isinstance(edge.target, str) or not edge.target:
            raise ValueError("edge target must be a non-empty string")
        if (
            not isinstance(edge.capacity, int)
            or isinstance(edge.capacity, bool)
            or edge.capacity <= 0
        ):
            raise ValueError("edge capacities must be positive integers")
        nodes.update((edge.source, edge.target))

    if source not in nodes or sink not in nodes:
        raise ValueError("source and sink must both appear in the network")
    return edge_list


def _validate_budget(budget: int) -> None:
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("budget must be a non-negative integer")


def max_flow(
    edges: Sequence[Edge],
    source: str,
    sink: str,
    removed_indices: Iterable[int] = (),
) -> int:
    """Compute maximum flow using Edmonds-Karp; edge indices identify removals."""
    edge_list = _validate_network(edges, source, sink)
    removed = set(removed_indices)
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(edge_list)
        for index in removed
    ):
        raise ValueError("removed edge indices must identify edges in the network")

    nodes = sorted({source, sink, *(e.source for e in edge_list), *(e.target for e in edge_list)})
    node_index = {node: index for index, node in enumerate(nodes)}
    residual = [[0] * len(nodes) for _ in nodes]
    for index, edge in enumerate(edge_list):
        if index not in removed:
            residual[node_index[edge.source]][node_index[edge.target]] += edge.capacity

    start, finish = node_index[source], node_index[sink]
    total_flow = 0
    while True:
        parent = [-1] * len(nodes)
        parent[start] = start
        queue = [start]
        for current in queue:
            for neighbor, capacity in enumerate(residual[current]):
                if capacity > 0 and parent[neighbor] == -1:
                    parent[neighbor] = current
                    queue.append(neighbor)
                    if neighbor == finish:
                        break
            if parent[finish] != -1:
                break

        if parent[finish] == -1:
            return total_flow

        path_flow = float("inf")
        node = finish
        while node != start:
            previous = parent[node]
            path_flow = min(path_flow, residual[previous][node])
            node = previous

        node = finish
        while node != start:
            previous = parent[node]
            residual[previous][node] -= path_flow
            residual[node][previous] += path_flow
            node = previous
        total_flow += int(path_flow)


def exact_interdiction(
    edges: Sequence[Edge], source: str, sink: str, budget: int
) -> InterdictionResult:
    """Enumerate all edge sets up to budget and return the lowest-flow result."""
    edge_list = _validate_network(edges, source, sink)
    _validate_budget(budget)
    best = InterdictionResult((), max_flow(edge_list, source, sink))
    for count in range(1, min(budget, len(edge_list)) + 1):
        for removed in combinations(range(len(edge_list)), count):
            flow = max_flow(edge_list, source, sink, removed)
            if flow < best.remaining_flow:
                best = InterdictionResult(removed, flow)
    return best


def train_q_learning(
    edges: Sequence[Edge],
    source: str,
    sink: str,
    budget: int,
    *,
    episodes: int = 20_000,
    alpha: float = 0.2,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 7,
) -> InterdictionResult:
    """Train tabular Q-learning and return its greedy interdiction episode."""
    edge_list = _validate_network(edges, source, sink)
    _validate_budget(budget)
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    limit = min(budget, len(edge_list))
    if limit == 0:
        return InterdictionResult((), max_flow(edge_list, source, sink))

    rng = random.Random(seed)
    q_values: dict[tuple[frozenset[int], int], float] = {}
    flow_cache: dict[frozenset[int], int] = {frozenset(): max_flow(edge_list, source, sink)}

    def flow_for(state: frozenset[int]) -> int:
        if state not in flow_cache:
            flow_cache[state] = max_flow(edge_list, source, sink, state)
        return flow_cache[state]

    for episode in range(episodes):
        progress = episode / max(episodes - 1, 1)
        epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)
        state = frozenset()

        while len(state) < limit:
            actions = [index for index in range(len(edge_list)) if index not in state]
            if rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                values = [q_values.get((state, index), 0.0) for index in actions]
                best_value = max(values)
                action = rng.choice(
                    [index for index, value in zip(actions, values) if value == best_value]
                )

            next_state = state | {action}
            reward = flow_for(state) - flow_for(next_state)
            terminal = len(next_state) == limit
            next_actions = (
                [index for index in range(len(edge_list)) if index not in next_state]
                if not terminal
                else []
            )
            future_value = max(
                (q_values.get((next_state, index), 0.0) for index in next_actions),
                default=0.0,
            )
            key = (state, action)
            old_value = q_values.get(key, 0.0)
            q_values[key] = old_value + alpha * (reward + future_value - old_value)
            state = next_state

    state = frozenset()
    while len(state) < limit:
        actions = [index for index in range(len(edge_list)) if index not in state]
        values = [q_values.get((state, index), 0.0) for index in actions]
        best_value = max(values)
        action = min(
            index for index, value in zip(actions, values) if value == best_value
        )
        state = state | {action}

    removed = tuple(sorted(state))
    return InterdictionResult(removed, flow_for(state))


def run_demo(
    budget: int = 2, episodes: int = 20_000, seed: int = 7
) -> tuple[InterdictionResult, InterdictionResult]:
    """Run the sample network and print the learned and exact results."""
    source, sink = "s", "t"
    initial_flow = max_flow(SAMPLE_EDGES, source, sink)
    learned = train_q_learning(
        SAMPLE_EDGES, source, sink, budget, episodes=episodes, seed=seed
    )
    exact = exact_interdiction(SAMPLE_EDGES, source, sink, budget)
    print(f"Network: three parallel routes with capacities 2, 3, and 1")
    print(f"Source: {source}; sink: {sink}; budget: {budget}")
    print(f"Initial maximum flow: {initial_flow}")
    print(
        f"Q-learning: remove {learned.removed_indices}; "
        f"remaining flow = {learned.remaining_flow}"
    )
    print(
        f"Exact optimum: remove {exact.removed_indices}; "
        f"remaining flow = {exact.remaining_flow}"
    )
    print(f"Optimality gap: {learned.remaining_flow - exact.remaining_flow}")
    return learned, exact


if __name__ == "__main__":
    run_demo()
