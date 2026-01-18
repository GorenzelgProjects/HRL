# From https://github.com/lweitkamp/option-critic-pytorch/blob/master/fourrooms.py

from typing import Optional, Dict

import logging
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Fourrooms(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, level=1, render_mode=None, headless=True, reward_config: Optional[Dict] = None, settings: Optional[Dict] = None):

        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.render_mode = render_mode
        self.headless = headless
        self.reward_config = reward_config
        self.settings = settings
        
        self.info = {"level": level}

        # Load reward configuration
        if reward_config is None:
            reward_config = {}
        self.reward_config = {
            "goal": reward_config.get("goal", 1),
        }
        
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0., high=1., shape=(np.sum(self.occupancy == 0),))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62 # East doorway
        self.init_states = list(range(self.observation_space.shape[0]))
        self.init_states.remove(self.goal)
        self.ep_steps = 0

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]
        self.ep_steps = 0
        
        self.info["steps"] = self.ep_steps
        return self.get_state(state), self.info

    def switch_goal(self):
        prev_goal = self.goal
        self.goal = self.rng.choice(self.init_states)
        self.init_states.append(prev_goal)
        self.init_states.remove(self.goal)
        assert prev_goal in self.init_states
        assert self.goal not in self.init_states

    def get_state(self, state):
        obs = np.zeros(self.observation_space.shape[0], dtype = np.float32)
        obs[state] = 1
        return obs

    def render(self, show_goal=True):
        current_grid = np.array(self.occupancy)
        self.currentcell
        current_grid[self.currentcell[0], self.currentcell[1]] = -1
        if show_goal:
            goal_cell = self.tocell[self.goal]
            current_grid[goal_cell[0], goal_cell[1]] = -2

        if self.render_mode == "human":
            if not hasattr(self, "_fig"):
                plt.ion()
                self._fig, self._ax = plt.subplots()
                self._img = self._ax.imshow(
                    current_grid,
                    cmap="viridis",
                    vmin=-2,
                    vmax=np.max(self.occupancy),
                )
                self._ax.set_title("Grid Environment")
                self._ax.set_xticks([])
                self._ax.set_yticks([])
            else:
                self._img.set_data(current_grid)

            plt.pause(0.1)
            return None
        else:
            return current_grid

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        self.ep_steps += 1

        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
                self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = self.reward_config["goal"] if done else 0
        truncated = False

        if not done and self.ep_steps >= 1000:
            done = True ; reward = 0.0

        return self.get_state(state), reward, done, truncated, self.info
    
    def get_wall_mask(self) -> np.ndarray:
        return self.occupancy

    def get_player_loc_from_state(self, state) -> np.ndarray:
        return np.array(self.tocell[np.argwhere(np.array(state)==1).flatten()[0]])