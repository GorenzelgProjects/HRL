from __future__ import annotations

import heapq
import itertools
import numpy as np
from typing import Optional, Dict, Tuple, TYPE_CHECKING
from thin_ice.data.classes.settings import *

if TYPE_CHECKING:
    from thin_ice.data.classes.Player import Player

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
        h = self.heuristic.h(state, goal_description)
        return h + g
    
class ThinIceState():
    def __init__(self, 
                 reward_config: Dict,
                 player: Player, 
                 action: int = None,
                 parent: ThinIceState = None,
                 has_key: bool = False,
                 can_teleport: bool = True,
                 reset_once: bool = False,
                 moved: bool = False,
                 keyhole_unlocked: bool = False,
                 moving_block_pos: Optional[Tuple[int, int]] = None):
        
        self.player = player
        self.action = action
        self.parent = parent
        # Fix: Path cost should increment from parent, not just copy
        if parent is None:
            self.path_cost = 0
        else:
            # Path cost increases by 1 for each step (step cost)
            # Rewards will be subtracted later in result() method
            self.path_cost = parent.path_cost + 1

        self.has_key = has_key
        self.can_teleport = can_teleport
        self.reset_once = reset_once
        self.moved = moved
        self.keyhole_unlocked = keyhole_unlocked
        
        # Track moving block position
        if moving_block_pos is not None:
            self.moving_block_pos = moving_block_pos
        elif parent is not None:
            self.moving_block_pos = parent.moving_block_pos
        elif self.player.game.movingBlock:
            self.moving_block_pos = (self.player.game.movingBlock.x, self.player.game.movingBlock.y)
        else:
            self.moving_block_pos = None
        
        # Fix: Copy visited_tiles from parent if it exists
        if parent is not None:
            self.visited_tiles = parent.visited_tiles.copy()
            self.complete_tiles = parent.complete_tiles
        else:
            self.visited_tiles = set()
            self.complete_tiles = 0
        
        self.reward_config = reward_config
        
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

            # Check if moving block exists and player is near it
            pushing_block = False
            if (self.player.game.currentLevel > MOVINGBLOCKLEVEL and 
                self.moving_block_pos is not None):
                
                block_x, block_y = self.moving_block_pos
                # Check if player is adjacent to moving block
                player_near_block = (
                    (block_x == self.player.x - 1 and block_y == self.player.y) or  # left
                    (block_x == self.player.x + 1 and block_y == self.player.y) or  # right
                    (block_x == self.player.x and block_y == self.player.y - 1) or  # up
                    (block_x == self.player.x and block_y == self.player.y + 1)     # down
                )
                
                if player_near_block:
                    # Check if player is moving TOWARDS the block (pushing it)
                    is_pushing = (
                        (block_x == self.player.x - 1 and dx == -1) or  # block left, moving left
                        (block_x == self.player.x + 1 and dx == 1) or    # block right, moving right
                        (block_y == self.player.y - 1 and dy == -1) or  # block up, moving up
                        (block_y == self.player.y + 1 and dy == 1)      # block down, moving down
                    )
                    
                    if is_pushing:
                        # Player is pushing the block
                        # Check if block can move in that direction
                        new_block_x = block_x + dx
                        new_block_y = block_y + dy
                        
                        # Block can't move if there's a wall in the way
                        if not self._block_collides_with_walls(new_block_x, new_block_y):
                            # Block can move, so player can push it
                            applicable_actions.append((action, (dx, dy)))
                            pushing_block = True
                        # If block is blocked, player can't move in that direction
                        continue
            
            # Normal movement (not pushing block)
            if not pushing_block:
                # Check collisions, but allow moving into block's position if we're not pushing
                # (since if we were pushing, we would have handled it above)
                if not self._collide_with_walls(dx, dy) and not self._collide_with_block(dx, dy):
                    applicable_actions.append((action, (dx, dy)))
        
        return applicable_actions
    
    def _collide_with_block(self, dx, dy):
        """Check if player would collide with moving block"""
        if self.moving_block_pos is None:
            return False
        target_x = self.player.x + dx
        target_y = self.player.y + dy
        block_x, block_y = self.moving_block_pos
        return target_x == block_x and target_y == block_y
    
    def _block_collides_with_walls(self, block_x, block_y):
        """Check if moving block would collide with walls at given position"""
        from data.classes.Immovable import KeyHole
        
        for wall in self.player.game.walls:
            # Skip keyhole if it's unlocked
            if isinstance(wall, KeyHole) and self.keyhole_unlocked:
                continue
            if wall.x == block_x and wall.y == block_y:
                return True
        return False
    
    def _collide_with_walls(self, dx, dy):
        """Check collision with walls, excluding keyhole if it's unlocked"""
        target_x = self.player.x + dx
        target_y = self.player.y + dy
        
        # Import KeyHole for isinstance check
        from data.classes.Immovable import KeyHole
        
        for wall in self.player.game.walls:
            # Skip keyhole if it's unlocked
            if isinstance(wall, KeyHole) and self.keyhole_unlocked:
                continue
            if wall.x == target_x and wall.y == target_y:
                return True
        return False
    
    def result(self, dx, dy):
        # Check if player is pushing moving block
        pushing_block = False
        old_block_pos = self.moving_block_pos
        if (self.moving_block_pos is not None and 
            self.player.game.currentLevel > MOVINGBLOCKLEVEL):
            block_x, block_y = self.moving_block_pos
            # Check if player is adjacent to block and moving towards it
            player_near_block = (
                (block_x == self.player.x - 1 and block_y == self.player.y and dx == -1) or  # left
                (block_x == self.player.x + 1 and block_y == self.player.y and dx == 1) or   # right
                (block_x == self.player.x and block_y == self.player.y - 1 and dy == -1) or  # up
                (block_x == self.player.x and block_y == self.player.y + 1 and dy == 1)      # down
            )
            
            if player_near_block:
                # Player is pushing the block
                new_block_x = block_x + dx
                new_block_y = block_y + dy
                
                # Check if block can move (no wall collision)
                if not self._block_collides_with_walls(new_block_x, new_block_y):
                    # Block can move, update its position
                    self.moving_block_pos = (new_block_x, new_block_y)
                    pushing_block = True
                else:
                    # Block is blocked, player can't move
                    return self
        
        # Check if move is valid (collision check)
        # When pushing block, player moves into block's old position, so skip block collision check
        player_target_x = self.player.x + dx
        player_target_y = self.player.y + dy
        
        # Check wall collision
        wall_collision = self._collide_with_walls(dx, dy)
        
        # Check block collision (but not if we're pushing, since block will have moved)
        block_collision = False
        if not pushing_block:
            # Check collision with block at its current position
            block_collision = self._collide_with_block(dx, dy)
        # If pushing, block collision is not an issue since block moves first
        
        if not wall_collision and not block_collision:
            # Move player position (without modifying game state)
            self.player.x += dx
            self.player.y += dy
            
            reward = 0.0
            
            # Track visited tiles
            new_pos = (self.player.x, self.player.y)
            if new_pos not in self.visited_tiles:
                self.visited_tiles.add(new_pos)
                self.complete_tiles += 1
                reward += self.reward_config.get("new_tile_reward", 0.1)
            
            # Check for key collection
            if self.player.game.key and self.player.collideWithTile(self.player.game.key):
                self.has_key = True
                reward += self.reward_config.get("key_collection_reward", 0.5)
            
            # Check for keyhole unlocking
            if self.has_key and self.player.game.keyHole and not self.keyhole_unlocked:
                # Check if player is near the keyhole (adjacent to it)
                if self.player.nearTile(self.player.game.keyHole) != 0:
                    self.keyhole_unlocked = True
                    self.has_key = False  # Key is consumed when unlocking
                    reward += self.reward_config.get("keyhole_unlock_reward", 0.5)
            
            # Check for teleporting (only after level 16)
            if (self.player.game.currentLevel > TELEPORTLEVEL and 
                self.can_teleport and 
                self.player.game.firstTeleporter and 
                self.player.game.secondTeleporter):
                
                # Check if player stepped on first teleporter
                if self.player.collideWithTile(self.player.game.firstTeleporter):
                    # Remove teleporter tile from visited (same as original game)
                    if new_pos in self.visited_tiles:
                        self.visited_tiles.remove(new_pos)
                        self.complete_tiles -= 1
                    # Teleport to second teleporter
                    self.player.x = self.player.game.secondTeleporter.x
                    self.player.y = self.player.game.secondTeleporter.y
                    self.can_teleport = False
                    reward += self.reward_config.get("teleport_reward", 0.0)
                    # Track the new position after teleporting
                    new_pos = (self.player.x, self.player.y)
                    if new_pos not in self.visited_tiles:
                        self.visited_tiles.add(new_pos)
                        self.complete_tiles += 1
                
                # Check if player stepped on second teleporter
                elif self.player.collideWithTile(self.player.game.secondTeleporter):
                    # Remove teleporter tile from visited (same as original game)
                    if new_pos in self.visited_tiles:
                        self.visited_tiles.remove(new_pos)
                        self.complete_tiles -= 1
                    # Teleport to first teleporter
                    self.player.x = self.player.game.firstTeleporter.x
                    self.player.y = self.player.game.firstTeleporter.y
                    self.can_teleport = False
                    reward += self.reward_config.get("teleport_reward", 0.0)
                    # Track the new position after teleporting
                    new_pos = (self.player.x, self.player.y)
                    if new_pos not in self.visited_tiles:
                        self.visited_tiles.add(new_pos)
                        self.complete_tiles += 1
            
            # Subtract reward from path cost (rewards reduce cost)
            self.path_cost -= reward
        else:
            # Move was blocked, but we still increment path cost for the attempt
            # This ensures we don't get stuck in infinite loops
            pass
        
        return self

    def __eq__(self, other) -> bool:
        """
        Compare states based on player position, key status, keyhole status, moving block position, and teleport status.
        States with same position but different key/keyhole/block/teleport status are different.
        """
        if isinstance(other, self.__class__):
            return (self.player.x, self.player.y, self.has_key, self.keyhole_unlocked, self.moving_block_pos, self.can_teleport) == \
                   (other.player.x, other.player.y, other.has_key, other.keyhole_unlocked, other.moving_block_pos, other.can_teleport)
        else:
            return False
    
    def __hash__(self):
        """
        Hash state based on player position, key status, keyhole status, moving block position, and teleport status.
        This ensures states with different key/keyhole/block/teleport status are treated as different.
        """
        return hash((self.player.x, self.player.y, self.has_key, self.keyhole_unlocked, self.moving_block_pos, self.can_teleport))

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