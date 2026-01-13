from baseline_models.astar_utils import *
import sys
import time

def test_a_star(env, heuristic: str = "manhattan", render: bool = False, delay: float = 0.05):

    # Set up
    initial_state, _ = env.reset()
    
    # Create initial state with a new player instance to avoid mutations
    # We need to create a new Player instance for the initial state
    initial_player = Player(env.game, env.player.x, env.player.y, env.settings)
    initial_state = ThinIceState(env.reward_config, initial_player)
    goal_state = ThinIceGoal(env.end_tile.x, env.end_tile.y)
    heuristic_obj = Heuristic(heuristic)
    action_set_converter = {
            0: (1, 0),   # right
            1: (0, -1),  # up
            2: (-1, 0),  # left
            3: (0, 1),   # down
        }
    frontier = FrontierAStar(heuristic_obj)
    
    frontier.prepare(goal_state)

    # Clear the parent pointer and cost in order make sure that the initial state is a root node
    initial_state.parent = None
    initial_state.path_cost = 0

    frontier.add(initial_state)
    expanded = set()
    
    max_iterations = 1000000  # Safety limit
    iteration = 0
    
    print("Searching for solution with A* algorithm...")
    
    while iteration < max_iterations:
        iteration += 1
        
        if frontier.is_empty():
            print("-" * 40, file=sys.stderr)
            print("FAILED TO FIND SOLUTION!", file=sys.stderr)
            print("-" * 40, file=sys.stderr)
            return False, [] # Failure to find a solution
        
        # Dequeue state from frontier and add to expanded set
        new_state = frontier.pop()
        expanded.add(new_state)
        
        # Check goal state
        if goal_state.is_goal(new_state):
            plan = new_state.extract_plan()
            print(f"Solution found in {iteration} iterations with {len(plan)} steps")
            if new_state.moving_block_pos:
                print(f"Final moving block position: {new_state.moving_block_pos}")
            
            # If rendering is enabled, execute the plan step by step
            if render:
                print(f"\nExecuting solution with {len(plan)} steps...")
                print("=" * 60)
                
                # Reset environment to initial state
                env.reset()
                
                # Convert direction strings to action integers
                direction_to_action = {
                    "right": 0,
                    "up": 1,
                    "left": 2,
                    "down": 3
                }
                
                # Render initial state
                env.render()
                time.sleep(delay)
                
                # Execute plan step by step
                total_reward = 0
                for step_idx, direction in enumerate(plan, 1):
                    action = direction_to_action[direction]
                    
                    # Take step
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    # Render the step
                    env.render()
                    print(f"Step {step_idx}/{len(plan)}: {direction.upper()} - Reward: {reward:.2f}, Total: {total_reward:.2f}")
                    time.sleep(delay)
                    
                    if terminated or truncated:
                        print(f"\nEpisode completed! Final reward: {total_reward:.2f}")
                        if terminated:
                            print("✓ Successfully reached the goal!")
                        break
                
                # Keep window open for a moment to see final state
                if not terminated and not truncated:
                    print(f"\nPlan executed. Final reward: {total_reward:.2f}")
                time.sleep(delay * 2)
                print("=" * 60)
            
            return True, plan
        
        # Expand new_state (0=right, 1=up, 2=left, 3=down)
        for (action, (dx, dy)) in new_state.get_applicable_actions(action_set_converter):
            
            # CRITICAL FIX: Create a new Player instance for each child state
            # This prevents mutations from affecting parent states
            # The Player shares the game object (for collision checking) but has its own position
            child_player = Player(
                new_state.player.game, 
                new_state.player.x, 
                new_state.player.y, 
                new_state.player.settings
            )
            
            # Create child state with new player instance
            child = ThinIceState(
                new_state.reward_config,  # Dict is immutable enough for our purposes
                child_player,
                action,
                new_state,
                new_state.has_key,  # These are primitives, no need to copy
                new_state.can_teleport,
                new_state.reset_once,
                new_state.moved,
                new_state.keyhole_unlocked,  # Track keyhole unlock status
                new_state.moving_block_pos  # Track moving block position
            )
            
            # Apply the action
            child = child.result(dx, dy)
            
            if child not in expanded and not frontier.contains(child):
                frontier.add(child)
                
            # Update priority if we found a better path to this state
            elif hasattr(frontier, "priority_queue") and frontier.contains(child):
                current_priority = frontier.priority_queue.get_priority(child)
                new_priority = frontier.f(child, goal_state)
                if current_priority > new_priority:
                    frontier.priority_queue.change_priority(child, new_priority)
    
    print("-" * 40, file=sys.stderr)
    print(f"MAX ITERATIONS ({max_iterations}) REACHED!", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    return False, []
