# Maximum-Flow Interdiction Demo

Run the sample from this folder:

```powershell
python max_flow_interdiction.py
python -m unittest -v test_max_flow_interdiction.py
```

The sample is a directed network with three independent `s`-to-`t` routes of
capacities 2, 3, and 1. The default budget removes two edges. The objective is
to minimize the remaining maximum flow; Q-learning uses flow reduction after
each removal as its reward, and episodes end when the budget is spent.

The script uses seeded tabular Q-learning and compares the learned policy with
an exact enumeration of all allowed removal sets. Exact search is practical
only for small networks because the number of candidate sets grows
combinatorially. Change the graph, source, sink, budget, episode count, or seed
in `max_flow_interdiction.py` to try another small example.
