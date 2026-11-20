# Building Skills

## Goal
Build skills in linear algebra, machine learning (emphasis on reinforcement learning), AI models, and network optimization. Approach each topic from the mathematics first, then implement it in code.

## How to work with me
- Start with the math (definitions, intuition, derivations), then show the code that implements it.
- Keep code notation consistent with the math (same symbols and variable names where practical).
- Build small from-scratch implementations (NumPy, SciPy) before reaching for libraries like PyTorch or Gurobi, so the math stays visible.
- Verify every derivation numerically: `assert` against finite differences, an exact solver, duality, or brute force on a tiny instance.
- When I'm working an exercise, give hints before full solutions unless I ask for the answer.

## Layout
- `ml/`: 01–04, linear algebra → regression → logistic/Newton → backprop
- `rl/`: 05–07, bandits/MDPs → Q-learning → policy gradients
- `optimization/`: 08–13, shortest path/LP duality → max flow → min-cost flow → MILP → interdiction → Q-routing
- Each notebook runs top to bottom without errors; keep it that way when editing.

## Tools
- Python with numpy, scipy (HiGHS `linprog`/`milp`), matplotlib, networkx.
- Gurobi (academic license) is available locally and optional in notebooks.
