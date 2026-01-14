import os
import json
from datetime import datetime
from typing import Callable, Any

import numpy as np
import gymnasium as gym

from model.base_manager import BaseModelManager
from model.baseline.astar.a_star_agent import train_a_star


class AStarManager(BaseModelManager):
    def __init__(self, heuristic: str, partial_env: Callable[[int], gym.Env], save_dir: str) -> None:
        self.heuristic = heuristic
        self.save_dir = os.path.join(save_dir, "astar")
        
        super().__init__(partial_env)

    def _save_results(self, level_results: list[dict[Any, Any]]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.save_dir
        os.makedirs(results_dir, exist_ok=True)
        filename = f"astar_all_levels_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "method": "astar",
            "heuristic": self.heuristic,
            "levels": level_results,
            "summary": {
                "total_levels": len(level_results),
                "successful_levels": sum(1 for r in level_results if r["success"]),
                "failed_levels": sum(1 for r in level_results if not r["success"]),
                "avg_plan_length": float(np.mean([r["plan_length"] for r in level_results if r["success"]])) if any(r["success"] for r in level_results) else 0.0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        print(f"Summary:")
        print(f"  Successful: {output['summary']['successful_levels']}/{output['summary']['total_levels']} levels")
        if output['summary']['avg_plan_length'] > 0:
            print(f"  Average plan length: {output['summary']['avg_plan_length']:.1f} steps")

    def train(self, levels: list[int], render: bool = False, delay: float = 0.05):
        level_results = []
        for level_num in levels:
            print(f"\n{'='*60}")
            print(f"Level {level_num}")
            print(f"{'='*60}")
            level_env = self.partial_env(level_num)
            try:
                success, plan = train_a_star(level_env, self.heuristic, render, delay)
                level_result = {
                    "level": level_num,
                    "success": success,
                    "plan_length": len(plan) if plan else 0,
                    "plan": plan if plan else []
                }
                level_results.append(level_result)

                if success:
                    print(f"✓ Level {level_num}: Solution found with {len(plan)} steps")
                else:
                    print(f"✗ Level {level_num}: No solution found")
            except Exception as e:
                print(f"✗ Level {level_num}: Error - {e}")
                import traceback
                traceback.print_exc()
                level_results.append({
                    "level": level_num,
                    "success": False,
                    "plan_length": 0,
                    "plan": [],
                    "error": str(e)
                })
            finally:
                level_env.close()
                
        # Print summary
        print(f"\n{'='*60}")
        print("Summary of All Levels")
        print(f"{'='*60}")
        successful = sum(1 for r in level_results if r["success"])
        print(f"Successfully solved: {successful}/{len(level_results)} levels")
        if successful > 0:
            avg_steps = np.mean([r["plan_length"] for r in level_results if r["success"]])
            print(f"Average solution length: {avg_steps:.1f} steps")

        self._save_results(level_results=level_results)
                
    def test(self, levels: list[int], render: bool, delay: float) -> None:
        print("No reason to test Astar, the direct test results came from the training.")
        # Implement if anyone has too much time on their hands with just loading and running
        #   the plan, and that's it.
        pass