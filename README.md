# Grid World Value Iteration with Hazard Modeling

This project implements a **Value Iteration** algorithm for finding the optimal policy in a grid world environment with obstacles, hazards, and a goal state. It supports both deterministic and stochastic hazard scenarios.

---

## Features

- **Grid World Setup**:
  - 5x5 grid with configurable obstacles, hazards, and a goal.
  - Obstacles block movement.
  - Hazards impose negative rewards and can have stochastic effects like teleportation or forced movement.

- **Value Iteration Algorithm**:
  - Computes the optimal value function and policy for each grid cell.
  - Supports stochastic transitions (action success probability and side effects).
  - Optionally includes stochastic hazard behavior:
    - Probability of teleporting to a random valid state.
    - Probability of being pushed back one square.

- **Visual Output**:
  - Color-coded terminal output using `colorama`:
    - Green `G` for goal.
    - Red `H` for hazards.
    - White `X` for obstacles.
    - Arrows showing optimal movement directions.
    - Numerical state values colored by type.

---

## Requirements

- Python 3.7+
- `numpy`
- `colorama`

Install dependencies via:

```bash
pip install numpy colorama
```
Run the script directly:

```bash
python3 91.py
```
---

## Usage

This runs two versions:

- **Basic Version** (Deterministic hazards)  
- **Advanced Version** (Stochastic hazards)  

The output shows:

- Iterations until convergence.  
- The optimal value function for each state.  
- The optimal policy with movement directions.  

---

## Code Overview

- **GRID_SIZE**: Size of the grid (5x5).  
- **OBSTACLES**: List of grid coordinates that cannot be entered.  
- **HAZARDS**: Grid coordinates with negative rewards and possible stochastic effects.  
- **GOAL**: The target state with positive reward.  
- **ACTIONS**: Possible moves (up, down, left, right).  
- **value_iteration()**: Main algorithm that computes values and policy.  
- **apply_stochastic_hazard()**: Applies random hazard effects in stochastic mode.  
- **print_value_function()** and **print_policy()**: Display results with colors.  

---

## Customization

- Modify grid size, obstacles, hazards, and rewards by changing the respective variables.  
- Adjust stochastic hazard probabilities (`HAZARD_TELEPORT_PROB`, `HAZARD_PUSH_PROB`).  
- Tweak discount factor (`GAMMA`) and action success probability (`ACTION_SUCCESS_PROB`).  

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).  
- Classic grid world examples in reinforcement learning.  

---

## License

This project is released under the **MIT License**.  

---

## Contact

For questions or suggestions, please contact **Afia Anjum Tamanna**:

- Email: [afiatamanna06@gmail.com](mailto:afiatamanna06@gmail.com)  
- GitHub: [https://github.com/afiatamanna06](https://github.com/afiatamanna06)  
