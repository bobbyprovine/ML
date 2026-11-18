"""
Column generation for the path-based maximum flow problem.

Master problem:     max  sum_p f_p
                    s.t. sum_{p containing a} f_p <= u_a   for every arc a   (dual y_a >= 0)
                         f_p >= 0

Restricted master:  the same LP, but only over the paths found so far (P').
Pricing problem:    shortest s-t path using the duals y_a as arc lengths.
                    A path is worth adding if its length is < 1
                    (reduced cost = 1 - sum of duals on the path > 0).

Only numpy and scipy are needed (the LP is solved by scipy's HiGHS solver).
"""

import heapq
from collections import deque

import numpy as np
from scipy.optimize import linprog


# ----------------------------------------------------------------------------
# Graph helpers
# ----------------------------------------------------------------------------
def build_adjacency(arcs):
    """arcs: {(u, v): capacity}. Returns {u: [v, ...]} in the order arcs were given."""
    adj = {}
    for (u, v) in arcs:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, [])
    return adj


def bfs_path(arcs, s, t):
    """Fewest-arc s-t path, returned as a list of arcs. Used to seed the RMP."""
    adj = build_adjacency(arcs)
    parent = {s: None}
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    if t not in parent:
        raise ValueError("No s-t path exists.")
    nodes = [t]
    while nodes[-1] != s:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    return list(zip(nodes, nodes[1:]))


def dijkstra(arcs, length, s, t):
    """Shortest s-t path where length[(u, v)] >= 0 is the cost of each arc.

    Returns (path_length, path), with path as a list of arcs.
    """
    adj = build_adjacency(arcs)
    dist = {s: 0.0}
    parent = {}
    heap = [(0.0, s)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v in adj[u]:
            nd = d + length[(u, v)]
            if nd < dist.get(v, float("inf")) - 1e-12:   # strict improvement only
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    nodes = [t]
    while nodes[-1] != s:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    return dist[t], list(zip(nodes, nodes[1:]))


# ----------------------------------------------------------------------------
# Restricted master problem (RMP)
# ----------------------------------------------------------------------------
def solve_rmp(arcs, paths):
    """Solve the RMP over the given paths.

    Returns (objective, f, y):
        f : array, flow on each path in `paths`
        y : {arc: dual value}, the shadow price of each arc's capacity
    """
    arc_list = list(arcs)
    # A[a, p] = 1 if arc a is on path p, else 0 (one row per arc, one column per path)
    A = np.array([[1.0 if a in p else 0.0 for p in paths] for a in arc_list])
    u = np.array([arcs[a] for a in arc_list], dtype=float)

    # linprog minimizes, so maximize sum(f) by minimizing -sum(f).
    res = linprog(c=-np.ones(len(paths)), A_ub=A, b_ub=u,
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"RMP failed: {res.message}")

    # For a minimization, ineqlin.marginals are <= 0; flip the sign to get the
    # dual values y_a >= 0 of the original maximization problem.
    y_vals = np.clip(-res.ineqlin.marginals, 0.0, None)
    y = dict(zip(arc_list, y_vals))
    return -res.fun, res.x, y


# ----------------------------------------------------------------------------
# Column generation loop
# ----------------------------------------------------------------------------
def fmt_path(path):
    return "-".join([path[0][0]] + [v for (_, v) in path])


def column_generation(arcs, s, t, initial_path=None, tol=1e-9, verbose=True):
    """Max flow by column generation. Returns (max_flow, paths, f, y)."""
    paths = [initial_path if initial_path is not None else bfs_path(arcs, s, t)]

    it = 0
    while True:
        it += 1
        obj, f, y = solve_rmp(arcs, paths)

        # Pricing: shortest s-t path with the duals as arc lengths.
        cost, new_path = dijkstra(arcs, y, s, t)
        reduced_cost = 1.0 - cost

        if verbose:
            nonzero = {f"{u}{v}": round(float(val), 4) for (u, v), val in y.items() if val > tol}
            flows = {fmt_path(p): round(float(x), 4) + 0.0 for p, x in zip(paths, f)}
            print(f"Iteration {it}")
            print(f"  paths in RMP : {[fmt_path(p) for p in paths]}")
            print(f"  RMP flows    : {flows}   objective = {obj:.4f}")
            print(f"  nonzero duals: {nonzero}")
            print(f"  pricing      : shortest path {fmt_path(new_path)}, "
                  f"cost = {cost:.4f}, reduced cost = {reduced_cost:.4f}")

        if cost >= 1.0 - tol:        # no path has positive reduced cost -> optimal
            if verbose:
                print("  -> shortest path cost >= 1: stop.\n")
            return obj, paths, f, y

        if new_path in paths:        # safety net against numerical cycling
            raise RuntimeError("Pricing returned a path already in the RMP.")
        paths.append(new_path)
        if verbose:
            print(f"  -> cost < 1: add {fmt_path(new_path)}.\n")


# ----------------------------------------------------------------------------
# Independent check: the ordinary arc-based max flow LP (no column generation)
# ----------------------------------------------------------------------------
def max_flow_arc_lp(arcs, s, t):
    """Standard arc-flow LP: maximize flow out of s with flow conservation."""
    arc_list = list(arcs)
    nodes = sorted({n for a in arc_list for n in a})
    inner = [n for n in nodes if n not in (s, t)]

    c = np.zeros(len(arc_list))                 # objective: net flow leaving s
    for j, (u, v) in enumerate(arc_list):
        if u == s:
            c[j] -= 1.0
        if v == s:
            c[j] += 1.0

    A_eq = np.zeros((len(inner), len(arc_list)))  # conservation at every inner node
    for i, n in enumerate(inner):
        for j, (u, v) in enumerate(arc_list):
            if v == n:
                A_eq[i, j] += 1.0
            if u == n:
                A_eq[i, j] -= 1.0

    bounds = [(0, arcs[a]) for a in arc_list]
    res = linprog(c, A_eq=A_eq, b_eq=np.zeros(len(inner)), bounds=bounds, method="highs")
    return -res.fun


# ----------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------
# 4-node network (step-by-step example)
ARCS_4 = {("s", "a"): 3, ("s", "b"): 2, ("a", "b"): 1, ("a", "t"): 2, ("b", "t"): 3}

# 6-node network
ARCS_6 = {("s", "a"): 10, ("s", "b"): 10, ("a", "b"): 2, ("a", "c"): 4, ("a", "d"): 8,
          ("b", "d"): 9, ("d", "c"): 6, ("c", "t"): 10, ("d", "t"): 10}


def run_example(name, arcs, s, t, initial_path=None):
    print("=" * 70)
    print(name)
    print("=" * 70)
    value, paths, f, y = column_generation(arcs, s, t, initial_path=initial_path)
    check = max_flow_arc_lp(arcs, s, t)

    print(f"Column generation max flow : {value:.4f}")
    print(f"Arc-based LP max flow      : {check:.4f}")
    assert abs(value - check) < 1e-6, "Column generation disagrees with the arc LP!"

    dual_obj = sum(arcs[a] * y[a] for a in arcs)
    cut = [a for a in arcs if y[a] > 1e-9]
    print(f"Dual objective sum(u_a*y_a): {dual_obj:.4f}")
    print(f"Arcs with y_a > 0 (min cut): {cut}\n")


if __name__ == "__main__":
    # The starting paths below match the documents; leave initial_path out
    # to seed with a BFS path instead.
    run_example("4-node example", ARCS_4, "s", "t",
                initial_path=[("s", "a"), ("a", "t")])
    run_example("6-node example", ARCS_6, "s", "t",
                initial_path=[("s", "a"), ("a", "c"), ("c", "t")])
