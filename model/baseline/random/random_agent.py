import time
import tqdm

def train_random_agent(env, num_episodes=5, render=False, delay=0.1):
    """Test the environment with a random agent"""
    print("=" * 60)
    print("Testing with Random Agent")
    print("=" * 60)
    
    results = []
    
    for episode in tqdm.tqdm(range(num_episodes), total=num_episodes):
        _, info = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        episode_start_time = time.time()
        
        # print(f"\nEpisode {episode + 1}")
        # print(f"Initial distance to exit: {info['distance']}")
        
        if render:
            try:
                env.render()
                time.sleep(delay)
            except Exception as e:
                print(f"Error during rendering: {e}")
                import traceback
                traceback.print_exc()
        
        while not terminated and not truncated:
            # Random action
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(delay)
            
            # if steps % 100 == 0:
            #     print(f"  Step {steps}: Reward={total_reward:.2f}, Distance={info['distance']}")
            
            if steps >= 4000:  # Safety limit
                truncated = True
                break
        
        episode_time = time.time() - episode_start_time
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
            "time_seconds": episode_time
        }
        results.append(episode_result)
        
        # print(f"Episode finished: Steps={steps}, Total Reward={total_reward:.2f}, "
        #       f"Terminated={terminated}, Visited Tiles={info['visited_tiles']}/{info['total_tiles']}, "
        #       f"Time={episode_time:.2f}s")
    
    print("\n" + "=" * 60)
    return results
