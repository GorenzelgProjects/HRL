"""
OptionShowcaseEnv – FINAL (Penalty-Aware, OC-Compatible)
------------------------------------------------------

This version implements **reward/penalty asymmetries** that are
*necessary* for option-critic to avoid collapse, exactly as discussed.

Key design principles:
- No uniform step penalty
- Penalties are tied to *environmental failures*
- Large terminal reward dominates, so penalties are worth enduring
- No zero-penalty loops that can dominate the value function

State: (row, col, mode)
Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

Tiles:
    # WALL        : impassable
    . EMPTY       : reward 0
    S START       : reward 0
    G GOAL        : +10 terminal
    ! STICKY      : action failure → -0.2
    ~ SLIP        : overshoot; wall-hit → -0.3
    ? NOISY       : random action → -0.1
    A ALIAS       : bad action → teleport + -0.5
    M MOMENTUM    : reversing direction → -0.2
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import pygame

# ---------------- Tiles ----------------
EMPTY = 0
WALL = 1
GOAL = 2
SLIP = 3
STICKY = 4
NOISY = 5
ALIAS = 6
MOMENTUM = 7

ASCII_TO_TILE = {
    '#': WALL,
    '.': EMPTY,
    'S': EMPTY,
    'G': GOAL,
    '~': SLIP,
    '!': STICKY,
    '?': NOISY,
    'A': ALIAS,
    'M': MOMENTUM,
}

TILE_COLORS = {
    EMPTY: (235, 235, 235),
    WALL: (30, 30, 30),
    GOAL: (60, 180, 75),
    SLIP: (120, 190, 255),
    STICKY: (220, 120, 120),
    NOISY: (240, 220, 120),
    ALIAS: (180, 120, 200),
    MOMENTUM: (120, 200, 160),
}


class OptionShowcaseEnv(gym.Env):
    """
    Render modes:
    - render(mode="rgb_array") -> returns image (H,W,3)
    - render(mode="human")    -> draws persistent window (pygame)

    play() is a thin wrapper around render(mode="window") + keyboard input
    """
    def __init__(self, level=1, level_str: str = None, max_steps: int = 800, seed: Optional[int] = None, reward_config: Optional[dict] = None, render_mode: str = "human", headless=None, settings=None):
        self.level_str = level_str if level_str else BIG_LEVEL
        self.max_steps = None
        self.step_count = 0
        self.prev_action = None

        self.np_random = np.random.RandomState(seed)
        self.action_space = spaces.Discrete(4)

        self.grid = None
        self.height = None
        self.width = None
        self.agent_pos = None
        self.info = {"level": level}

        self._parse_level(self.level_str)

        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(self.height),
                spaces.Discrete(self.width),
                spaces.Discrete(4),
            )
        )

        self.cell_size = 36
        self.screen = None

        if reward_config is None:
            reward_config = {}
        self.reward_config = {
            "goal": reward_config.get("goal", 1),
        }
        
        self.render_mode = render_mode

    # ---------------- Parsing ----------------
    def _parse_level(self, level_str: str):
        rows = [list(r) for r in level_str.strip().splitlines()]
        w = {len(r) for r in rows}
        assert len(w) == 1, "Level must be rectangular"

        self.height = len(rows)
        self.width = w.pop()
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == 'S':
                    self.agent_pos = (r, c)
                self.grid[r, c] = ASCII_TO_TILE[ch]


    # ---------------- Gym API ----------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        self.step_count = 0
        self.prev_action = None

        self._parse_level(self.level_str)
        non_walls = np.argwhere((self.grid != WALL) | (self.grid == GOAL))
        start_idx = np.random.choice(list(range(len(non_walls))))
        self.agent_pos = non_walls[start_idx]
        return self._get_obs(), self.info

    def step(self, action: int):
        self.step_count += 1
        r, c = self.agent_pos
        tile = self.grid[r, c]

        reward = 0.0
        dr, dc = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[action]

        # ----- Tile dynamics -----
        if tile == STICKY and self.np_random.rand() < 0.6:
            reward -= 0.2
            nr, nc = r, c

        elif tile == NOISY and self.np_random.rand() < 0.4:
            reward -= 0.1
            a = self.np_random.randint(0, 4)
            dr, dc = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[a]
            nr, nc = r + dr, c + dc

        elif tile == SLIP:
            nr, nc = r + 2 * dr, c + 2 * dc

        elif tile == ALIAS:
            if action in (2, 3):  # DOWN or LEFT
                reward -= 0.5
                self.agent_pos = self._start_pos()
                return self._get_obs(), reward, False, False, {}
            nr, nc = r + dr, c + dc

        elif tile == MOMENTUM:
            if self.prev_action is not None and action != self.prev_action:
                reward -= 0.2
            nr, nc = r + dr, c + dc

        else:
            nr, nc = r + dr, c + dc

        # ----- Collision -----
        if nr < 0 or nr >= self.height or nc < 0 or nc >= self.width:
            nr, nc = r, c
        if self.grid[nr, nc] == WALL:
            if tile == SLIP:
                reward -= 0.3
            nr, nc = r, c

        self.agent_pos = (nr, nc)
        self.prev_action = action

        terminated = False
        if self.grid[nr, nc] == GOAL:
            reward += self.reward_config["goal"]
            terminated = True

        # truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, False, self.info

    # ---------------- State ----------------
    def _get_obs(self):
        r, c = self.agent_pos
        tile = self.grid[r, c]
        mode = {EMPTY: 0, SLIP: 1, STICKY: 2, NOISY: 2, ALIAS: 3, MOMENTUM: 3}.get(tile, 0)
        return np.array([r, c, mode])

    def _start_pos(self):
        # don't use anymore
        for r in range(self.height):
            for c in range(self.width):
                if self.level_str.splitlines()[r][c] == 'S':
                    return (r, c)

    # ---------------- Render ----------------
    def render(self):
        """
        mode="rgb_array": return numpy image for training visualization
        mode="human"   : draw to persistent pygame window
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            ) if self.render_mode == "human" else pygame.Surface(
                (self.width * self.cell_size, self.height * self.cell_size)
            )

        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, TILE_COLORS[self.grid[r, c]], rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        ar, ac = self.agent_pos
        center = (ac * self.cell_size + self.cell_size // 2, ar * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (40, 40, 200), center, self.cell_size // 3)

        if self.render_mode == "human":
            pygame.display.flip()
            return None

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))
        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size))

        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, TILE_COLORS[self.grid[r, c]], rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        ar, ac = self.agent_pos
        center = (ac * self.cell_size + self.cell_size // 2, ar * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (40, 40, 200), center, self.cell_size // 3)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))


    # ---------------- Interactive Play ----------------
    def play(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        pygame.display.set_caption("Option Showcase Env")
        clock = pygame.time.Clock()
        self.reset()

        running = True
        while running:
            clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_UP:
                        print("Step reward: ", self.step(0)[1])
                    elif event.key == pygame.K_RIGHT:
                        print("Step reward: ", self.step(1)[1])
                    elif event.key == pygame.K_DOWN:
                        print("Step reward: ", self.step(2)[1])
                    elif event.key == pygame.K_LEFT:
                        print("Step reward: ", self.step(3)[1])

            self.render(mode="window")

        pygame.quit()

    def get_wall_mask(self) -> np.ndarray:
        return (self.grid==WALL).astype(float)

    def get_player_loc_from_state(self, state: Any) -> tuple[np.ndarray, Optional[str]]:
        return np.array([state[0], state[1]]), None

# ---------------- Large Showcase Level ----------------
# BIG_LEVEL = """
# ############################
# #S....!.....~~~~~.....A....#
# #.####!#####~~~~~#####A....#
# #....!.....~~~~~.....A.....#
# #....!.....................#
# ######.###############.....#
# #?????.MMMMMMMMMMMMMMM.....#
# #?????.MMMMMMMMMMMMMMM.....#
# #?????.MMMMMMMMMMMMMMM.....#
# ######.................G...#
# ############################
# """
BIG_LEVEL = """
##############
#.......!!!!!#
#..#####!!!!!#
#...........!#
#M######!!..!#
#M#MMM.....!!#
#M#M##!!!..!!#
#M#MM#~~~~~~~#
#M##M#~~~~~#~#
#MMMM#~~~G~~~#
##############
"""

if __name__ == "__main__":
    env = OptionShowcaseEnv(BIG_LEVEL, render_mode="human")
    # env.play()
    # obs, _ = env.reset()
    for _ in range(5):
        obs, _ = env.reset()
        env.render()
        import time
        time.sleep(0.6)
