"""
Column generation for max flow, with the restricted master LP solved by the
(dual) SIMPLEX method in scipy.

    Restricted master (RMP):  max  sum_p f_p
                              s.t. sum_{p containing a} f_p <= u_a   for each arc a
                                   f >= 0
    Pricing:                  shortest s-t path with the duals y_a as arc lengths;
                              add the path if its length is < 1.

Only numpy and scipy are needed:  pip install numpy scipy
"""

import heapq

import numpy as np
from scipy.optimize import linprog


def shortest_path(arcs, length, s, t):
    """Dijkstra. length[(u, v)] >= 0. Returns (distance, list of arcs)."""
    adj = {}
    for (u, v) in arcs:
        adj.setdefault(u, []).append(v)
    dist, parent, heap, done = {s: 0.0}, {}, [(0.0, s)], set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v in adj.get(u, []):
            nd = d + length[(u, v)]
            if nd < dist.get(v, float("inf")) - 1e-12:
                dist[v], parent[v] = nd, u
                heapq.heappush(heap, (nd, v))
    nodes = [t]
    while nodes[-1] != s:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    return dist[t], list(zip(nodes, nodes[1:]))


def solve_rmp_simplex(arcs, paths):
    """Solve the restricted master LP with HiGHS' dual simplex. Returns (obj, f, y)."""
    arc_list = list(arcs)
    A = np.array([[1.0 if a in p else 0.0 for p in paths] for a in arc_list])  # arcs x paths
    u = np.array([arcs[a] for a in arc_list], dtype=float)

    # linprog minimizes, so minimize -sum(f).  method="highs-ds" = dual simplex.
    res = linprog(c=-np.ones(len(paths)), A_ub=A, b_ub=u,
                  bounds=(0, None), method="highs-ds")
    # marginals of a minimization are <= 0; negate to get the duals y_a >= 0 of the max problem
    y = dict(zip(arc_list, np.clip(-res.ineqlin.marginals, 0.0, None)))
    return -res.fun, res.x, y


def name(path):
    return "-".join([path[0][0]] + [v for _, v in path])


def column_generation(arcs, s, t, first_path):
    paths = [first_path]
    for it in range(1, 100):
        obj, f, y = solve_rmp_simplex(arcs, paths)
        cost, new_path = shortest_path(arcs, y, s, t)

        print(f"Iteration {it}: objective = {obj:g}")
        print("  flows :", {name(p): round(float(x), 6) + 0.0 for p, x in zip(paths, f)})
        print("  duals :", {f"{a[0]}{a[1]}": round(float(v), 6) for a, v in y.items() if v > 1e-9})
        print(f"  pricing: shortest path {name(new_path)} has cost {cost:g}"
              f" (reduced cost {1 - cost:g})")

        if cost >= 1 - 1e-9:                       # no path with positive reduced cost
            print("  -> cost >= 1, stop.\n")
            return obj, paths, f, y
        print(f"  -> cost < 1, add {name(new_path)}\n")
        paths.append(new_path)


if __name__ == "__main__":
    # 4-node example
    arcs4 = {("s", "a"): 3, ("s", "b"): 2, ("a", "b"): 1, ("a", "t"): 2, ("b", "t"): 3}
    print("=== 4-node network ===")
    value, *_ = column_generation(arcs4, "s", "t", [("s", "a"), ("a", "t")])
    print("Max flow =", value, "\n")

    # 6-node example
    arcs6 = {("s", "a"): 10, ("s", "b"): 10, ("a", "b"): 2, ("a", "c"): 4, ("a", "d"): 8,
             ("b", "d"): 9, ("d", "c"): 6, ("c", "t"): 10, ("d", "t"): 10}
    print("=== 6-node network ===")
    value, *_ = column_generation(arcs6, "s", "t", [("s", "a"), ("a", "c"), ("c", "t")])
    print("Max flow =", value)
