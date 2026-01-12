import numpy as np
import time

def test_q_learning(env, num_episodes=100, learning_rate=0.1, discount=0.95, epsilon=0.1, render=False, delay=0.05):
    """Test with a simple Q-learning agent"""
    print("=" * 60)
    print("Testing with Q-Learning Agent")
    print("=" * 60)
    
    # Initialize Q-table
    # State space is large, so we'll use a simplified representation
    # We'll use a hash of the observation as the state key
    q_table = {}
    
    def get_state_key(obs):
        """Convert observation to a hashable state key"""
        # Use a simplified state: player position + exit position + has_key
        # This is a simplified representation for demonstration
        # In practice, you might want to use a neural network instead
        return hash(obs.tobytes())
    
    def get_q_value(state_key, action):
        """Get Q-value for state-action pair"""
        if state_key not in q_table:
            q_table[state_key] = np.zeros(env.action_space.n)
        return q_table[state_key][action]
    
    def update_q_value(state_key, action, reward, next_state_key, terminated):
        """Update Q-value using Q-learning"""
        current_q = get_q_value(state_key, action)
        if terminated:
            target_q = reward
        else:
            next_q_values = [get_q_value(next_state_key, a) for a in range(env.action_space.n)]
            target_q = reward + discount * max(next_q_values)
        
        q_table[state_key][action] = current_q + learning_rate * (target_q - current_q)
    
    episode_rewards = []
    episode_lengths = []
    results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        state_key = get_state_key(obs)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        episode_start_time = time.time()
        
        if render and episode == num_episodes - 1:  # Render only last episode
            env.render()
            time.sleep(delay)
        
        while not terminated and not truncated:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = [get_q_value(state_key, a) for a in range(env.action_space.n)]
                action = np.argmax(q_values)
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_key = get_state_key(next_obs)
            
            # Update Q-table
            update_q_value(state_key, action, reward, next_state_key, terminated)
            
            state_key = next_state_key
            total_reward += reward
            steps += 1
            
            if render and episode == num_episodes - 1:  # Render only last episode
                env.render()
                time.sleep(delay)
            
            if steps >= 500:  # Safety limit
                truncated = True
                break
        
        episode_time = time.time() - episode_start_time
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        episode_result = {
            "episode": episode + 1,
            "steps": steps,
            "total_reward": float(total_reward),
            "terminated": terminated,
            "truncated": truncated,
            "visited_tiles": info['visited_tiles'],
            "total_tiles": info['total_tiles'],
            "complete_tiles": info['complete_tiles'],
            "final_distance": info['distance'],
            "has_key": info['has_key'],
            "level": info['level'],
            "time_seconds": episode_time,
            "q_table_size": len(q_table)
        }
        results.append(episode_result)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Avg Reward={avg_reward:.2f}, Avg Length={avg_length:.1f}, "
                  f"Q-table size={len(q_table)}")
    
    print(f"\nTraining complete!")
    print(f"Average reward over last 10 episodes: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f}")
    print("=" * 60)
    
    return results
