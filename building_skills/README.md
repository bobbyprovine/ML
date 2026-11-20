# Building Skills: ML, RL and network optimization

These Jupyter notebooks go in order, each one math first and then code. Every derivation is checked with an `assert` on real numbers (finite-difference gradients, strong duality, brute force on small instances), so a notebook that runs top to bottom without errors has checked its own math.

## Order

The ML track (01–07) and the optimization track (08–12) can be worked in parallel. Notebook 13 joins them.

| # | Notebook | Math | Code skill |
|---|----------|------|------------|
| **ML** | | | |
| 01 | `ml/01_linear_algebra_for_ml` | eigendecomposition, SVD, Eckart–Young, PCA | NumPy linear algebra, numerical verification |
| 02 | `ml/02_linear_regression_and_gradient_descent` | normal equations, step size vs. Hessian eigenvalues, condition number | vectorized gradients, GD |
| 03 | `ml/03_logistic_regression_and_newton` | convexity, gradient and Hessian, Newton/IRLS | **gradient checking**, Newton's method |
| 04 | `ml/04_neural_network_from_scratch` | backpropagation as the matrix chain rule | 2-layer MLP in NumPy, full gradient check |
| **RL** | | | |
| 05 | `rl/05_bandits_and_mdp_value_iteration` | Bellman equations, contraction mapping | ε-greedy bandits, value iteration, policy evaluation by linear solve |
| 06 | `rl/06_q_learning` | stochastic approximation of the Bellman optimality equation | tabular Q-learning checked against exact Q* |
| 07 | `rl/07_policy_gradient_reinforce` | policy-gradient theorem, baselines | REINFORCE with softmax policy |
| **Network optimization** | | | |
| 08 | `optimization/08_shortest_paths_dijkstra_and_lp` | incidence matrix, total unimodularity, LP duality | Dijkstra, `linprog`, complementary slackness |
| 09 | `optimization/09_max_flow_min_cut` | max-flow/min-cut via duality | Edmonds–Karp, LP and dual LP |
| 10 | `optimization/10_min_cost_flow_and_duals` | node potentials, reduced costs, sensitivity | min-cost flow LP, optimality certificate |
| 11 | `optimization/11_integer_programming_scipy_and_gurobi` | knapsack, fixed-charge linking, LP relaxation bounds | `scipy.optimize.milp`, brute-force validation, Gurobi |
| 12 | `optimization/12_shortest_path_interdiction` | bilevel max–min → single MILP by dualizing the inner LP | interdiction MILP, value-of-budget curve |
| **Bridge** | | | |
| 13 | `optimization/13_rl_meets_networks_q_routing` | shortest path as a deterministic MDP | Q-routing that adapts after interdiction |

Each notebook ends with exercises. That's where most of the skill building happens: do them in a copy of the notebook or a new cell below.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r building_skills/requirements.txt
jupyter lab
```

Gurobi is optional. Notebook 11 runs a Gurobi cell only if `gurobipy` is installed and licensed (`pip install gurobipy`, then `grbgetkey <your academic key>`). Everything else uses SciPy's HiGHS solvers.

## Suggested next steps after 13

- Rebuild notebook 04 in PyTorch and check autograd against your hand-written gradients.
- DQN on a small Gymnasium environment (Q-learning with a neural network, replay buffer and target network).
- Benders decomposition for interdiction (exercise 3 in notebook 12), then compare with `../max_flow_interdiction/`.
