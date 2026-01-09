from __future__ import annotations

import heapq
import itertools
import numpy as np
import copy
from typing import Optional, Dict, Tuple
from data.classes.settings import *

class PriorityQueue:

    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def add(self, element: np.ndarray, priority: int):
        count = next(self.counter)
        entry = [priority, -count, element]
        heapq.heappush(self.heap, entry)
        self.entry_finder[element] = entry

    def change_priority(self, element: np.ndarray, new_priority: int):
        entry = self.entry_finder.pop(element)
        entry[2] = None
        # Add new entry with new priority
        self.add(element, new_priority)

    def pop(self) -> np.ndarray:
        # Since some of the elements in the queue might have been invalidated by the 'change_priority' method, we need
        # to keep taking elements from the queue until we find a valid entry.
        while True:
            entry = heapq.heappop(self.heap)
            if entry[2] is not None:
                break
        state = entry[2]
        self.entry_finder.pop(state)
        return state

    def clear(self):
        self.heap.clear()
        self.entry_finder.clear()
        self.counter = itertools.count()

    def size(self) -> int:
        return len(self.entry_finder)

    def get_priority(self, element) -> int:
        entry = self.entry_finder.get(element)
        if entry is None:
            return None
        return entry[0]

class FrontierBestFirst:

    def __init__(self):
        self.goal_description = None
        self.priority_queue = PriorityQueue()
        self.set = set()

    def prepare(self, goal_description: np.ndarray):
        self.goal_description = goal_description
        self.priority_queue.clear()
        self.set.clear()

    def f(self, state: np.ndarray, goal_description: np.ndarray) -> int:
        raise Exception("FrontierBestFirst should not be directly used. Instead use a subclass overriding f()")

    def add(self, state: np.ndarray):
        self.priority_queue.add(state, self.f(state, self.goal_description))
        self.set.add(state)

    def pop(self) -> np.ndarray:
        state = self.priority_queue.pop()
        self.set.remove(state)
        return state

    def is_empty(self) -> bool:
        return self.priority_queue.size() == 0

    def size(self) -> int:
        return self.priority_queue.size()

    def contains(self, state: np.ndarray) -> bool:
        return state in self.set


# The FrontierAStar and FrontierGreedy classes extend the FrontierBestFirst class, that is, they are
# exact copies of the above class but where the 'f' method is replaced.

class FrontierAStar(FrontierBestFirst):

    def __init__(self, heuristic):
        super().__init__()
        self.heuristic = heuristic

    def f(self, state: ThinIceState, goal_description: ThinIceGoal) -> int:
        g = state.path_cost
        print((state.player.x, state.player.y))
        # print("h: ", self.heuristic.h(state, goal_description))
        # print("g:", g)
        print("f:", self.heuristic.h(state, goal_description) + g)
        return self.heuristic.h(state, goal_description) + g
    
class ThinIceState():
    def __init__(self, 
                 reward_config: Dict,
                 player: Player, 
                 action: int = None,
                 parent: ThinIceState = None,
                 has_key: bool = False,
                 can_teleport: bool = True,
                 reset_once: bool = False,
                 moved: bool = False):
        
        self.player = player
        self.action = action
        self.parent = parent
        self.path_cost = 0 if parent is None else parent.path_cost

        self.has_key = has_key
        self.can_teleport = can_teleport
        self.reset_once = reset_once
        self.moved = moved
        
        self.visited_tiles = set()
        self.reward_config = reward_config
        self.complete_tiles = 0
        
    def extract_plan(self) -> list[int]:
        """Extracts a plan from the search tree by walking backwards through the search tree"""
        converter = { 0: "right", 1: "up", 2: "left", 3: "down"}
        
        reverse_plan = []
        current_node = self
        while current_node.parent is not None:
            reverse_plan.append(converter[current_node.action])
            current_node = current_node.parent
        reverse_plan.reverse()
        return reverse_plan
    
    def get_applicable_actions(self, action_set):
        applicable_actions = []
        for action, (dx, dy) in action_set.items():

            if self.player.game.currentLevel > MOVINGBLOCKLEVEL and self.player.nearTile(self.player.game.movingBlock) != 0:
                locationOfPlayer = self.player.nearTile(self.player.game.movingBlock)
                self.player.game.blockIsMoving = True
                
                if locationOfPlayer == 1 and dx == -1 and dy == 0:
                    continue
                elif locationOfPlayer == 2 and dx == 1 and dy == 0:
                    continue
                elif locationOfPlayer == 3 and dx == 0 and dy == -1:
                    continue
                elif locationOfPlayer == 4 and dx == 0 and dy == 1:
                    continue
                
                # If the player is not near a moving block, just do the normal collison check                        
                else:
                    if not self.player.collideWithGroup(self.player.game.walls, dx, dy):
                        applicable_actions.append((action, (dx, dy)))           
            
            
            # When it's the earlier levels, just check if the player is colliding with a wall
            elif not self.player.collideWithGroup(self.player.game.walls, dx, dy):
                applicable_actions.append((action, (dx, dy)))
        
        return applicable_actions
    
    def result(self, dx, dy):
        # Update game state references
        import pdb
        
        # Move player
        self.player.checkAndMove(dx=dx, dy=dy)
        
        # Check if move was successful
        reward = 0.0
        
        # Track visited tiles
        new_pos = (self.player.x, self.player.y)
        if new_pos not in self.visited_tiles:
            self.visited_tiles.add(new_pos)
            self.complete_tiles += 1
            reward += self.reward_config["new_tile_reward"]
        
        # TODO: Later implementation
        # # Check if reached exit
        # if self.player.collideWithTile(self.end_tile):
        #     # Bonus reward for completing level
        #     if self.complete_tiles == self.total_tiles:
        #         reward += self.reward_config["perfect_completion_bonus"]
        #     else:
        #         reward += self.reward_config["level_completion_reward"]
        #     terminated = True
        
        # # Check for key collection
        # if self.player.game.key and self.player.collideWithTile(self.player.game.key):
        #     self.has_key = True
        #     self.game.hasKey = True
        #     reward += self.reward_config["key_collection_reward"]
        
        # # Check for keyhole unlocking
        # if self.has_key and self.game.keyHole:
        #     if self.player.nearTile(self.game.keyHole) != 0:
        #         self.has_key = False
        #         self.game.hasKey = False
        #         reward += self.reward_config["keyhole_unlock_reward"]
        
        # # Check for treasure
        # if self.player.game.treasureTile and self.player.collideWithTile(self.player.game.treasureTile):
        #     reward += self.reward_config["treasure_collection_reward"]
        
        # # Check for death (stuck)
        # if self.player.checkDeath():
        #     reward += self.reward_config["death_penalty"]
        
        # Update sprites (including player animation)
        # self.all_sprites.update()
        # self.score_sprites.update()  # Update player sprite
        # self.updating_block_group.update()
        
        # Get new observation
        # obs = self._get_obs()
        # info = self._get_info()
        
        self.path_cost -= reward
        
        return self

    def __eq__(self, other) -> bool:
        """
        Notice that we here only compare the agent position, but ignore all other fields.
        That means that two states with identical positions but e.g. different parent will be seen as equal.
        """
        if isinstance(other, self.__class__):
            return (self.player.x, self.player.y) == (other.player.x, other.player.y)
        else:
            return False
    
    def __hash__(self):
        """
        Allows the state to be stored in a hash table for efficient lookup.
        Notice that we here only hash the agent positions and box positions, but ignore all other fields.
        That means that two states with identical positions but e.g. different parent will map to the same hash value.
        """
        return hash((self.player.x, self.player.y)) # TODO: Also add haskey

class ThinIceGoal():
    def __init__(self, x: int, y: int) -> None:
        self.goal_coords = (x, y)
    
    def is_goal(self, state: ThinIceState) -> bool:
        player_coords = (state.player.x, state.player.y)
        return self.goal_coords == player_coords
    
class Heuristic():
    def __init__(self, name: str):
        self.name = name
        self.h = self.manhattan_distance if name == "manhattan" else None
    
    def manhattan_distance(self, state: ThinIceState, goal: ThinIceGoal):
        dx, dy = state.player.x - goal.goal_coords[0], state.player.y - goal.goal_coords[1]
        return abs(dx) + abs(dy)