import numpy as np
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

# Define the grid world
GRID_SIZE = 5
OBSTACLES = [(1, 1), (2, 2), (3, 1)]
HAZARDS = [(0, 4), (4, 0)]
GOAL = (4, 4)

# Actions: up, down, left, right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ['↑', '↓', '←', '→']  # Using arrows for better visual

# Rewards
GOAL_REWARD = 10
HAZARD_REWARD = -10
STEP_REWARD = -1
OBSTACLE_REWARD = -5  # For hitting an obstacle

# Discount factor
GAMMA = 0.9

# Probability of intended action succeeding
ACTION_SUCCESS_PROB = 0.8
# Probability of other actions (split equally)
SIDE_PROB = (1 - ACTION_SUCCESS_PROB) / 2

# Stochastic hazard parameters (optional)
HAZARD_TELEPORT_PROB = 0.3  # Chance to teleport to random state
HAZARD_PUSH_PROB = 0.2      # Chance to be pushed back one square

def print_legend():
    """Print a legend explaining the symbols and colors"""
    print("\n" + "="*40)
    print(f"{'LEGEND':^40}")
    print("="*40)
    print(f"{Fore.GREEN}G{Style.RESET_ALL} - Goal state (reward = {GOAL_REWARD})")
    print(f"{Fore.RED}H{Style.RESET_ALL} - Hazard state (reward = {HAZARD_REWARD})")
    print(f"{Fore.WHITE}{Back.BLACK}X{Style.RESET_ALL} - Obstacle (cannot enter)")
    print("↑↓←→ - Optimal movement directions")
    print(f"{Fore.BLUE}Blue numbers{Style.RESET_ALL} - State values")
    print(f"{Fore.RED}Red numbers{Style.RESET_ALL} - Hazard state values")
    if HAZARD_TELEPORT_PROB > 0 or HAZARD_PUSH_PROB > 0:
        print("\nStochastic Hazards:")
        print(f"- {HAZARD_TELEPORT_PROB*100:.0f}% chance to teleport")
        print(f"- {HAZARD_PUSH_PROB*100:.0f}% chance to be pushed back")
    print("="*40 + "\n")

def is_valid_state(row, col):
    """Check if state is within grid and not an obstacle"""
    return (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE 
            and (row, col) not in OBSTACLES)

def get_next_state(row, col, action):
    """Get possible next states and their probabilities"""
    next_states = []
    
    # Intended action
    new_row, new_col = row + action[0], col + action[1]
    if is_valid_state(new_row, new_col):
        next_states.append((new_row, new_col, ACTION_SUCCESS_PROB))
    else:
        # Stay in place if hitting obstacle or boundary
        next_states.append((row, col, ACTION_SUCCESS_PROB))
    
    # Possible unintended actions (left and right of intended)
    side_actions = []
    if action[0] == 0:  # If moving left/right, unintended are up/down
        side_actions = [(-1, 0), (1, 0)]
    else:  # If moving up/down, unintended are left/right
        side_actions = [(0, -1), (0, 1)]
    
    for a in side_actions:
        new_row, new_col = row + a[0], col + a[1]
        if is_valid_state(new_row, new_col):
            next_states.append((new_row, new_col, SIDE_PROB))
        else:
            next_states.append((row, col, SIDE_PROB))
    
    return next_states

def get_reward(row, col):
    """Get reward for entering a state"""
    if (row, col) == GOAL:
        return GOAL_REWARD
    elif (row, col) in HAZARDS:
        return HAZARD_REWARD
    elif (row, col) in OBSTACLES:
        return OBSTACLE_REWARD
    else:
        return STEP_REWARD

def apply_stochastic_hazard(row, col):
    """Apply stochastic hazard effects (optional)"""
    if (row, col) not in HAZARDS:
        return [(row, col, 1.0)]  # No effect if not in hazard
    
    outcomes = []
    
    # Chance to teleport to random valid state
    if np.random.random() < HAZARD_TELEPORT_PROB:
        valid_states = [(r, c) for r in range(GRID_SIZE) 
                       for c in range(GRID_SIZE) 
                       if is_valid_state(r, c) and (r, c) != (row, col)]
        if valid_states:
            teleport_state = valid_states[np.random.randint(len(valid_states))]
            outcomes.append((teleport_state[0], teleport_state[1], HAZARD_TELEPORT_PROB))
    
    # Chance to be pushed back
    if np.random.random() < HAZARD_PUSH_PROB:
        push_row = max(0, row - 1)  # Push up/left as default
        push_col = max(0, col - 1)
        if is_valid_state(push_row, push_col):
            outcomes.append((push_row, push_col, HAZARD_PUSH_PROB))
        else:
            outcomes.append((row, col, HAZARD_PUSH_PROB))
    
    # Remaining probability stays in current state
    remaining_prob = 1.0 - sum(p for _, _, p in outcomes)
    if remaining_prob > 0:
        outcomes.append((row, col, remaining_prob))
    
    return outcomes

def value_iteration(theta=0.01, use_stochastic_hazards=False, verbose=True):
    """Perform value iteration until convergence"""
    # Initialize value function
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    iteration_count = 0
    
    while True:
        iteration_count += 1
        delta = 0
        new_V = np.copy(V)
        
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if (row, col) in OBSTACLES:
                    continue  # Skip obstacles
                
                if (row, col) == GOAL:
                    new_V[row, col] = GOAL_REWARD
                    continue
                
                max_value = -np.inf
                
                for action in ACTIONS:
                    next_states = get_next_state(row, col, action)
                    
                    # Apply stochastic hazards if enabled
                    if use_stochastic_hazards:
                        updated_states = []
                        for (n_row, n_col, prob) in next_states:
                            if (n_row, n_col) in HAZARDS:
                                hazard_outcomes = apply_stochastic_hazard(n_row, n_col)
                                for h_row, h_col, h_prob in hazard_outcomes:
                                    updated_states.append((h_row, h_col, prob * h_prob))
                            else:
                                updated_states.append((n_row, n_col, prob))
                        next_states = updated_states
                    
                    action_value = 0
                    for (n_row, n_col, prob) in next_states:
                        reward = get_reward(n_row, n_col)
                        action_value += prob * (reward + GAMMA * V[n_row, n_col])
                    
                    if action_value > max_value:
                        max_value = action_value
                
                delta = max(delta, abs(V[row, col] - max_value))
                new_V[row, col] = max_value
        
        V = new_V
        
        if verbose and iteration_count % 10 == 0:
            print(f"Iteration {iteration_count}: Max delta = {delta:.4f}")
        
        if delta < theta:
            break
    
    if verbose:
        print(f"\nConverged in {iteration_count} iterations")
    
    # Extract optimal policy with colored output
    policy = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if (row, col) in OBSTACLES:
                policy[row, col] = f"{Fore.WHITE}{Back.BLACK}X{Style.RESET_ALL}"
                continue
            
            if (row, col) == GOAL:
                policy[row, col] = f"{Fore.GREEN}{Back.BLACK}G{Style.RESET_ALL}"
                continue
            
            if (row, col) in HAZARDS:
                # Still calculate policy for hazards (you might enter them)
                pass
            
            max_value = -np.inf
            best_action = None
            
            for i, action in enumerate(ACTIONS):
                next_states = get_next_state(row, col, action)
                
                if use_stochastic_hazards:
                    updated_states = []
                    for (n_row, n_col, prob) in next_states:
                        if (n_row, n_col) in HAZARDS:
                            hazard_outcomes = apply_stochastic_hazard(n_row, n_col)
                            for h_row, h_col, h_prob in hazard_outcomes:
                                updated_states.append((h_row, h_col, prob * h_prob))
                        else:
                            updated_states.append((n_row, n_col, prob))
                    next_states = updated_states
                
                action_value = 0
                for (n_row, n_col, prob) in next_states:
                    reward = get_reward(n_row, n_col)
                    action_value += prob * (reward + GAMMA * V[n_row, n_col])
                
                if action_value > max_value:
                    max_value = action_value
                    best_action = ACTION_NAMES[i]
            
            # Color the policy based on cell type
            if (row, col) in HAZARDS:
                policy[row, col] = f"{Fore.RED}{Back.BLACK}{best_action}{Style.RESET_ALL}"
            else:
                policy[row, col] = best_action
    
    return V, policy, iteration_count

def print_value_function(V):
    """Print the value function with nice formatting"""
    print("\nOptimal Value Function:")
    print("+" + "-" * (GRID_SIZE * 9 - 1) + "+")
    for row in range(GRID_SIZE):
        print("|", end="")
        for col in range(GRID_SIZE):
            if (row, col) in OBSTACLES:
                print(f" {Fore.WHITE}{Back.BLACK}  X    {Style.RESET_ALL} |", end="")
            elif (row, col) == GOAL:
                print(f" {Fore.GREEN}{Back.BLACK}  G    {Style.RESET_ALL} |", end="")
            elif (row, col) in HAZARDS:
                print(f" {Fore.RED}{V[row, col]:6.2f}{Style.RESET_ALL} |", end="")
            else:
                print(f" {Fore.BLUE}{V[row, col]:6.2f}{Style.RESET_ALL} |", end="")
        print("\n+" + "-" * (GRID_SIZE * 9 - 1) + "+")

def print_policy(policy):
    """Print the policy with colored formatting"""
    print("\nOptimal Policy:")
    print("+" + "-" * (GRID_SIZE * 5 - 1) + "+")
    for row in range(GRID_SIZE):
        print("|", end="")
        for col in range(GRID_SIZE):
            print(f" {policy[row, col]} |", end="")
        print("\n+" + "-" * (GRID_SIZE * 5 - 1) + "+")

# Run value iteration without stochastic hazards
print(f"{Fore.YELLOW}=== Basic Version (Deterministic Hazards) ==={Style.RESET_ALL}")
optimal_values, optimal_policy, steps = value_iteration(use_stochastic_hazards=False)
print_legend()
print_value_function(optimal_values)
print_policy(optimal_policy)
print(f"\nConverged in {steps} iterations")

# Run value iteration with stochastic hazards
print(f"\n{Fore.YELLOW}=== Advanced Version (Stochastic Hazards) ==={Style.RESET_ALL}")
optimal_values_stoch, optimal_policy_stoch, steps_stoch = value_iteration(use_stochastic_hazards=True)
print_value_function(optimal_values_stoch)
print_policy(optimal_policy_stoch)
print(f"\nConverged in {steps_stoch} iterations")