"""
Thin Ice Gymnasium Environment
A Gymnasium wrapper for the Thin Ice game
"""

from typing import Optional, Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame as pg
import sys
import os


# Helper function to resolve paths correctly (defined early)
def _resolve_path(path):
    """Resolve a path relative to the thin_ice directory"""
    # Get the directory where this file is located (thin_ice/)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # If path starts with "thin_ice/", remove it and use base_dir
    if path.startswith("thin_ice/"):
        path = path[len("thin_ice/") :]

    # Join with base_dir
    resolved = os.path.join(base_dir, path)

    # Normalize the path
    return os.path.normpath(resolved)


# Add the current directory to the path to import game classes
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Initialize pygame early (before importing classes that use it)
pg.init()
pg.mixer.init()

# Monkey-patch pygame.image.load to resolve paths automatically
# This must be done before importing game classes that use pg.image.load
_original_pg_image_load = pg.image.load


def _patched_pg_image_load(filename):
    """Patched version of pg.image.load that resolves thin_ice paths"""
    if isinstance(filename, str) and "thin_ice/" in filename:
        filename = _resolve_path(filename)
    return _original_pg_image_load(filename)


# Apply the patch immediately
pg.image.load = _patched_pg_image_load

from environment.thin_ice.data.classes.sprites import *
from environment.thin_ice.data.classes.settings import *


class ThinIceEnv(gym.Env):
    """
    Gymnasium environment for Thin Ice game.

    The agent controls a player on a grid, trying to reach the exit tile.
    The player can move in 4 directions (up, down, left, right).
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 24}

    def __init__(
        self,
        level: int = 1,
        render_mode: Optional[str] = None,
        headless: bool = True,
        reward_config: Optional[Dict] = None,
        generate_water: bool = True,
        use_coord_state_representation: bool = False,
        use_image_state_representation: bool = False
    ):
        """
        Initialize the Thin Ice environment.

        Args:
            level: Level number to play (1-19)
            render_mode: Rendering mode ("human", "rgb_array", "ansi", or None)
            headless: If True, disable pygame display (for faster training)
            reward_config: Optional dict with reward function parameters
        """
        super().__init__()
        
        self.render_surface = pg.Surface((WIDTH, HEIGHT))  # for RL

        self.level = level
        self.render_mode = render_mode
        self.headless = headless

        # Load reward configuration
        if reward_config is None:
            reward_config = {}
        self.reward_config = {
            "new_tile_reward": reward_config.get("new_tile_reward", 0.1),
            "step_reward": reward_config.get("step_reward", -0.1),
            "level_completion_reward": reward_config.get(
                "level_completion_reward", 5.0
            ),
            "perfect_completion_bonus": reward_config.get(
                "perfect_completion_bonus", 10.0
            ),
            "key_collection_reward": reward_config.get("key_collection_reward", 1.0),
            "keyhole_unlock_reward": reward_config.get("keyhole_unlock_reward", 1.0),
            "treasure_collection_reward": reward_config.get(
                "treasure_collection_reward", 2.0
            ),
            "invalid_move_penalty": reward_config.get("invalid_move_penalty", -0.01),
            "death_penalty": reward_config.get("death_penalty", -5.0),
            "use_distance_reward": reward_config.get("use_distance_reward", False),
            "distance_reward_scale": reward_config.get("distance_reward_scale", -0.01),
        }

        # Set environment variable for headless mode BEFORE initializing pygame
        # This helps on Linux/Unix systems, but we'll also handle Windows
        if headless and render_mode != "human":
            if os.name != "nt":  # Not Windows
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        # Initialize pygame display (required for sprite loading)
        # Pygame needs a video mode even in headless mode for image loading
        # Note: pg.init() was already called at module level
        if not pg.display.get_init():
            pg.display.init()

        # Initialize display module explicitly
        if not pg.display.get_init():
            pg.display.init()

        # Initialize a display mode (required for image loading)
        # Even in headless mode, pygame needs a display surface to convert images
        # We MUST set a display BEFORE loading sprites
        try:
            current_surface = pg.display.get_surface()
            if current_surface is None:
                if render_mode == "human" and not headless:
                    # Set proper display for rendering
                    self.screen = pg.display.set_mode((WIDTH, HEIGHT))
                    pg.display.set_caption("Thin Ice - Gymnasium")
                else:
                    # Small display for headless mode (required for image loading)
                    # Using a small but valid size (8x8) that works on all platforms
                    pg.display.set_mode((8, 8))
                    self.screen = pg.display.get_surface()
            else:
                # Display already exists, but we might need to resize it
                if render_mode == "human" and not headless:
                    # Resize to proper size for rendering
                    self.screen = pg.display.set_mode((WIDTH, HEIGHT))
                    pg.display.set_caption("Thin Ice - Gymnasium")
                else:
                    self.screen = current_surface
        except (pg.error, AttributeError) as e:
            # If display initialization fails, this is a problem
            # We need a display for image loading
            raise RuntimeError(
                f"Failed to initialize pygame display (required for image loading): {e}"
            )

        self.settings = {
            "generate_water": generate_water,
            "use_coord_state_representation": use_coord_state_representation,
            "use_image_state_representation": use_image_state_representation
        }

        # Grid dimensions
        self.grid_width = int(GRIDWIDTH)
        self.grid_height = int(GRIDHEIGHT)

        # Initialize game state
        self.current_level = level
        self.game = None
        self.player = None
        self.end_tile = None
        self.walls = None
        self.ice_sprites = None
        self.has_key = False
        self.can_teleport = True
        self.reset_once = False
        self.moved = False

        # Track visited tiles for reward
        self.visited_tiles = set()
        self.total_tiles = 0
        self.complete_tiles = 0

        # Action space: 4 discrete actions (0=right, 1=up, 2=left, 3=down)
        self.action_space = spaces.Discrete(4)

        # Observation space: Grid representation
        # Each cell can be: empty(0), wall(1), ice(2), water(3), player(4), exit(5), key(6), keyhole(7)
        # We'll use a flattened grid representation
        self.observation_space = spaces.Box(
            low=0, high=7, shape=(self.grid_height * self.grid_width,), dtype=np.int32
        )

        # Alternative: Dictionary observation with player position and grid
        # Uncomment if you prefer this format:
        # self.observation_space = spaces.Dict({
        #     "grid": spaces.Box(0, 7, (self.grid_height, self.grid_width), dtype=np.int32),
        #     "player": spaces.Box(0, max(self.grid_width, self.grid_height), (2,), dtype=np.int32),
        #     "exit": spaces.Box(0, max(self.grid_width, self.grid_height), (2,), dtype=np.int32),
        #     "has_key": spaces.Discrete(2),
        # })

        # Load spritesheets (required for game initialization)
        # Display must be set before this point
        self._load_data()

        # Initialize the game
        self._init_game()

    def _load_data(self):
        """Load game data (spritesheets, sounds, etc.)"""
        # Resolve paths to work from any directory
        player_sprite_path = _resolve_path(PLAYERSPRITE)
        player_xml_path = _resolve_path(PLAYERXML)
        water_sprite_path = _resolve_path(WATERSPRITE)
        water_xml_path = _resolve_path(WATERXML)
        key_sprite_path = _resolve_path(KEYSPRITE)
        key_xml_path = _resolve_path(KEYXML)
        teleporter_sprite_path = _resolve_path(TELEPORTERSPRITE)
        teleporter_xml_path = _resolve_path(TELEPORTERXML)

        self.player_sprite_sheet = Spritesheet(player_sprite_path, player_xml_path)
        self.water_sprite_sheet = Spritesheet(water_sprite_path, water_xml_path)
        self.key_sprite_sheet = Spritesheet(key_sprite_path, key_xml_path)
        self.teleporter_sprite_sheet = Spritesheet(
            teleporter_sprite_path, teleporter_xml_path
        )

        # Load sounds (required by Player and other game objects)
        # In headless mode, we can create dummy sounds that do nothing
        try:
            move_sound_path = _resolve_path("thin_ice/data/sound/move.ogg")
            self.move_sound = pg.mixer.Sound(move_sound_path)
            self.move_sound.set_volume(
                0.0 if self.headless else 0.1
            )  # Silent in headless
        except:
            # Create a dummy sound object if file loading fails
            self.move_sound = type("DummySound", (), {"play": lambda: None})()

        try:
            ice_break_sound_path = _resolve_path("thin_ice/data/sound/breakIce.ogg")
            self.ice_break_sound = pg.mixer.Sound(ice_break_sound_path)
            self.ice_break_sound.set_volume(0.0 if self.headless else 0.2)
        except:
            self.ice_break_sound = type("DummySound", (), {"play": lambda: None})()

        try:
            moving_block_sound_path = _resolve_path(
                "thin_ice/data/sound/movingBlockSound.ogg"
            )
            self.moving_block_sound = pg.mixer.Sound(moving_block_sound_path)
            self.moving_block_sound.set_volume(0.0 if self.headless else 0.2)
        except:
            self.moving_block_sound = type("DummySound", (), {"play": lambda: None})()

        try:
            teleport_sound_path = _resolve_path("thin_ice/data/sound/teleportSound.ogg")
            self.teleport_sound = pg.mixer.Sound(teleport_sound_path)
            self.teleport_sound.set_volume(0.0 if self.headless else 0.2)
        except:
            self.teleport_sound = type("DummySound", (), {"play": lambda: None})()

        # Other sounds (may not be used but good to have)
        try:
            key_get_sound_path = _resolve_path("thin_ice/data/sound/keyGet.ogg")
            self.key_get_sound = pg.mixer.Sound(key_get_sound_path)
            self.key_get_sound.set_volume(0.0 if self.headless else 0.2)
        except:
            self.key_get_sound = type("DummySound", (), {"play": lambda: None})()

        try:
            treasure_sound_path = _resolve_path("thin_ice/data/sound/treasure.ogg")
            self.treasure_sound = pg.mixer.Sound(treasure_sound_path)
            self.treasure_sound.set_volume(0.0 if self.headless else 0.2)
        except:
            self.treasure_sound = type("DummySound", (), {"play": lambda: None})()

        try:
            dead_sound_path = _resolve_path("thin_ice/data/sound/dead.ogg")
            self.dead_sound = pg.mixer.Sound(dead_sound_path)
            self.dead_sound.set_volume(0.0 if self.headless else 0.2)
        except:
            self.dead_sound = type("DummySound", (), {"play": lambda: None})()

        try:
            reset_sound_path = _resolve_path("thin_ice/data/sound/reset.ogg")
            self.reset_sound = pg.mixer.Sound(reset_sound_path)
            self.reset_sound.set_volume(0.0 if self.headless else 0.2)
        except:
            self.reset_sound = type("DummySound", (), {"play": lambda: None})()

        try:
            all_tile_complete_sound_path = _resolve_path(
                "thin_ice/data/sound/allTileComplete.ogg"
            )
            self.all_tile_complete_sound = pg.mixer.Sound(all_tile_complete_sound_path)
            self.all_tile_complete_sound.set_volume(0.0 if self.headless else 0.2)
        except:
            self.all_tile_complete_sound = type(
                "DummySound", (), {"play": lambda: None}
            )()

    def _init_game(self):
        """Initialize game objects"""
        # Create sprite groups
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        self.movable = pg.sprite.Group()
        self.items = pg.sprite.Group()
        self.ice_sprites = pg.sprite.Group()
        self.score_sprites = pg.sprite.Group()
        self.updating_block_group = pg.sprite.Group()
        self.no_water_group = pg.sprite.Group()

        # Create a minimal game object for compatibility
        class MinimalGame:
            def __init__(self, env):
                self.allSprites = env.all_sprites
                self.walls = env.walls
                self.movable = env.movable
                self.items = env.items
                self.iceSprites = env.ice_sprites
                self.scoreSprites = env.score_sprites
                self.updatingBlockGroup = env.updating_block_group
                self.noWaterGroup = env.no_water_group
                self.playerSpriteSheet = env.player_sprite_sheet
                self.waterSpriteSheet = env.water_sprite_sheet
                self.keySpriteSheet = env.key_sprite_sheet
                self.teleporterSpriteSheet = env.teleporter_sprite_sheet
                self.currentLevel = env.current_level
                self.hasKey = False
                self.canTeleport = True
                self.resetOnce = False
                self.moved = False
                self.lastLevelSolved = True
                self.blockIsMoving = False
                self.settings = env.settings
                self.endTile = None
                self.key = None
                self.keyHole = None
                self.movingBlock = None
                self.movingBlockTile = None
                self.firstTeleporter = None
                self.secondTeleporter = None
                self.treasureTile = None
                # Add all sound attributes
                self.moveSound = env.move_sound
                self.iceBreakSound = env.ice_break_sound
                self.movingBlockSound = env.moving_block_sound
                self.teleportSound = env.teleport_sound
                self.keyGet = env.key_get_sound
                self.treasureSound = env.treasure_sound
                self.deadSound = env.dead_sound
                self.resetSound = env.reset_sound
                self.allTileComplete = env.all_tile_complete_sound

        self.game = MinimalGame(self)
        self.game.hasKey = self.has_key
        self.game.canTeleport = self.can_teleport
        self.game.resetOnce = self.reset_once
        self.game.moved = self.moved

    def _load_map(self):
        """Load the current level map"""
        # Clear existing sprites
        for sprite in self.all_sprites:
            sprite.kill()

        # Reset state
        self.visited_tiles = set()
        self.complete_tiles = 0
        self.has_key = False
        self.can_teleport = True
        self.game.hasKey = False
        self.game.canTeleport = True

        # Read map file
        map_data = []
        total_free = 0
        filename = f"thin_ice/data/maps/level{self.current_level}.txt"
        filename = _resolve_path(filename)

        try:
            with open(filename, "r") as f:
                map_data = [line.strip() for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Level file not found: {filename}")

        # Parse map
        for row, tiles in enumerate(map_data):
            for col, tile in enumerate(tiles):
                if tile == "W":
                    Wall(self.game, col, row)
                elif tile == "0":
                    Unused(self.game, col, row)
                elif tile == "F":
                    Free(self.game, col, row)
                    total_free += 1
                elif tile == "E":
                    self.end_tile = End(self.game, col, row)
                elif tile == "I":
                    Ice(self.game, col, row)
                    total_free += 2
                elif tile == "K":
                    Free(self.game, col, row)
                    self.game.key = GoldenKey(self.game, col, row)
                    total_free += 1
                elif tile == "B":
                    self.game.movingBlockTile = MovingBlockTile(self.game, col, row)
                elif tile == "T":
                    Free(self.game, col, row)
                    self.game.movingBlock = MovingBlock(self.game, col, row)
                    total_free += 1
                elif tile == "%":
                    Ice(self.game, col, row)
                    self.game.movingBlock = MovingBlock(self.game, col, row)
                    total_free += 2
                elif tile == "&":
                    self.game.movingBlockTile = MovingBlockTile(self.game, col, row)
                    self.game.key = GoldenKey(self.game, col, row)
                elif tile == "!":
                    Ice(self.game, col, row)
                    self.game.key = GoldenKey(self.game, col, row)
                    total_free += 2
                elif tile == "1":
                    self.game.firstTeleporter = Teleporter(self.game, col, row)
                elif tile == "2":
                    self.game.secondTeleporter = Teleporter(self.game, col, row)
                elif tile == "H":
                    self.game.keyHole = KeyHole(self.game, col, row)
                    total_free += 1
                elif tile == "M":
                    Free(self.game, col, row)
                    if self.game.lastLevelSolved:
                        self.game.treasureTile = Treasure(self.game, col, row)
                    total_free += 1
                elif tile == "P":
                    Free(self.game, col, row)
                    if self.player is None:
                        self.player = Player(self.game, col, row, self.settings)
                    else:
                        self.player.movetoCoordinate(col, row)
                    total_free += 1

                    # Add start position to visited tiles
                    self.visited_tiles.add((self.player.x, self.player.y))

        self.total_tiles = total_free - (2 * 19)  # Subtract top/bottom menu rows
        self.game.endTile = self.end_tile

    def _get_coord_obs(self) -> np.ndarray:
        """
        This is only useful for non-transfer learning and state->value mapping
        Returns:
            np.ndarray: [player.x, player.y, has_key(bool), keyhole_unlocked(bool), canTeleport(bool)]
        keyhole_unlocked is 1 in levels without keyhole anyways
        """
        if self.settings["generate_water"]:
            raise NotImplementedError("Not implemented yet")
        # didn't bother to check grid boundaries :3
        return np.array(
            [
                self.player.x,
                self.player.y,
                self.game.hasKey,
                (
                    0 if self.game.keyHole else 1
                ),  # in step() self.game.keyHole is set as None when unlocked
                self.game.canTeleport,
            ]
        )
    
    def _get_obs_as_img(self) -> np.ndarray:
        surface = self.render_surface
        
        surface.fill(BGCOLOR)

        if self.all_sprites:
            self.all_sprites.draw(surface)

        if self.updating_block_group:
            self.updating_block_group.draw(surface)

        if self.score_sprites:
            self.score_sprites.draw(surface)

        # Optional debug border
        pg.draw.rect(surface, (255, 255, 255), (0, 0, WIDTH, HEIGHT), 2)
        
        # Downsample to (84, 84) # NOTE HARDCODED
        small_surface = pg.transform.scale(self.render_surface, (84, 84))
        
        # pygame gives (W, H, C)
        obs = pg.surfarray.array3d(small_surface)

        # Convert to (C, H, W) – what ML frameworks expect
        obs = obs.transpose(2, 1, 0)
        
        # Gray-scale
        obs = obs.mean(axis=0, keepdims=True)  # grayscale
        obs = obs.astype(np.float32) / 255.0
        
        return obs

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation.

        Returns:
            Flattened grid representation where each cell is encoded:
            0 = empty/free, 1 = wall, 2 = ice, 3 = water, 4 = player, 5 = exit, 6 = key, 7 = keyhole
            TODO: CRITICAL, someone forgot portals, boxes and portals - ouf - it should be included
        """
        # Initialize grid with zeros (empty)
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)

        # Mark walls
        for wall in self.walls:
            if 0 <= wall.x < self.grid_width and 0 <= wall.y < self.grid_height:
                grid[wall.y, wall.x] = 1

        # Mark ice
        for ice in self.ice_sprites:
            if 0 <= ice.x < self.grid_width and 0 <= ice.y < self.grid_height:
                grid[ice.y, ice.x] = 2

        # Mark water (check noWaterGroup for water tiles)
        for sprite in self.all_sprites:
            if isinstance(sprite, Water):
                if 0 <= sprite.x < self.grid_width and 0 <= sprite.y < self.grid_height:
                    grid[sprite.y, sprite.x] = 3

        # Mark exit
        if self.end_tile:
            if (
                0 <= self.end_tile.x < self.grid_width
                and 0 <= self.end_tile.y < self.grid_height
            ):
                grid[self.end_tile.y, self.end_tile.x] = 5

        # Mark key
        if self.game.key and hasattr(self.game.key, "x"):
            if (
                0 <= self.game.key.x < self.grid_width
                and 0 <= self.game.key.y < self.grid_height
            ):
                grid[self.game.key.y, self.game.key.x] = 6

        # Mark keyhole
        if self.game.keyHole and hasattr(self.game.keyHole, "x"):
            if (
                0 <= self.game.keyHole.x < self.grid_width
                and 0 <= self.game.keyHole.y < self.grid_height
            ):
                grid[self.game.keyHole.y, self.game.keyHole.x] = 7

        # Mark player (overwrites other tiles)
        if self.player:
            if (
                0 <= self.player.x < self.grid_width
                and 0 <= self.player.y < self.grid_height
            ):
                grid[self.player.y, self.player.x] = 4

        # Flatten and return
        return grid.flatten()

    def _get_info(self) -> Dict:
        """Get auxiliary information"""
        distance = 0
        if self.player and self.end_tile:
            distance = abs(self.player.x - self.end_tile.x) + abs(
                self.player.y - self.end_tile.y
            )

        return {
            "distance": distance,
            "visited_tiles": len(self.visited_tiles),
            "total_tiles": self.total_tiles,
            "complete_tiles": self.complete_tiles,
            "has_key": self.has_key,
            "level": self.current_level,
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Optional dict with 'level' key to set level

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        # Set level if provided
        if options and "level" in options:
            self.current_level = options["level"]
            self.game.currentLevel = self.current_level

        # Reinitialize game
        self._init_game()

        # Create player (will be positioned by map loading)
        self.player = None

        # Load map (this will create and position the player)
        self._load_map()

        # Ensure player exists
        if self.player is None:
            # Fallback: create player at default position if not found in map
            self.player = Player(self.game, 0, 0, self.settings)

        # Update game references
        self.game.hasKey = self.has_key
        self.game.canTeleport = self.can_teleport

        # Update sprites (including player animation)
        self.all_sprites.update()
        self.score_sprites.update()  # Update player sprite
        self.updating_block_group.update()

        # Get initial observation
        if self.settings["use_coord_state_representation"]:
            obs = self._get_coord_obs()
        elif self.settings["use_image_state_representation"]:
            obs = self._get_obs_as_img()
        else:
            obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0=right, 1=up, 2=left, 3=down)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Map action to direction
        action_to_direction = {
            0: (1, 0),  # right
            1: (0, -1),  # up
            2: (-1, 0),  # left
            3: (0, 1),  # down
        }

        dx, dy = action_to_direction[action]

        # Update game state references
        self.game.hasKey = self.has_key
        self.game.canTeleport = self.can_teleport
        self.game.moved = False

        # Ensure player exists
        if self.player is None:
            if self.settings["use_coord_state_representation"]:
                obs = self._get_coord_obs()
            else:
                obs = self._get_obs()
            return obs, -1.0, True, False, self._get_info()

        # Try to move player
        old_pos = (self.player.x, self.player.y) if self.player else None
        self.player.checkAndMove(dx=dx, dy=dy)

        # Check if move was successful
        moved = self.game.moved
        reward = 0.0
        terminated = False
        truncated = False

        if moved:
            # Penalty for stepping
            reward += self.reward_config["step_reward"]

            # Track visited tiles
            new_pos = (self.player.x, self.player.y)
            if new_pos not in self.visited_tiles:
                self.visited_tiles.add(new_pos)
                self.complete_tiles += 1
                reward += self.reward_config["new_tile_reward"]

            # Distance-based reward (if enabled)
            if self.reward_config["use_distance_reward"] and self.end_tile:
                distance = abs(self.player.x - self.end_tile.x) + abs(
                    self.player.y - self.end_tile.y
                )
                reward += self.reward_config["distance_reward_scale"] * distance

            # Check if reached exit
            if self.player.collideWithTile(self.end_tile):
                # Bonus reward for completing level
                if self.complete_tiles == self.total_tiles:
                    reward += self.reward_config["perfect_completion_bonus"]
                else:
                    reward += self.reward_config["level_completion_reward"]
                terminated = True

            # Check for key collection
            if self.game.key and self.player.collideWithTile(self.game.key):
                # Remove key sprite (same behavior as original game)
                self.game.key.kill()
                self.game.key = None  # Clear the reference

                # Give player the key
                self.has_key = True
                self.game.hasKey = True
                reward += self.reward_config["key_collection_reward"]

            # Check for keyhole unlocking
            if self.has_key and self.game.keyHole:
                if self.player.nearTile(self.game.keyHole) != 0:
                    # Store keyhole position before removing it
                    keyhole_x = self.game.keyHole.x
                    keyhole_y = self.game.keyHole.y

                    # Remove keyhole from walls group and replace with Free tile
                    # This is the same behavior as the original game
                    self.game.keyHole.kill()  # Removes from all sprite groups including walls
                    Free(self.game, keyhole_x, keyhole_y)  # Replace with free tile
                    self.game.keyHole = None  # Clear the reference

                    # Consume the key
                    self.has_key = False
                    self.game.hasKey = False
                    reward += self.reward_config["keyhole_unlock_reward"]

            # Check for treasure
            if self.game.treasureTile and self.player.collideWithTile(
                self.game.treasureTile
            ):
                reward += self.reward_config["treasure_collection_reward"]

            # Check for teleporting (only after level 16)
            if (
                self.game.currentLevel > TELEPORTLEVEL
                and self.can_teleport
                and self.game.firstTeleporter
                and self.game.secondTeleporter
            ):

                # Check if player stepped on first teleporter
                if self.player.collideWithTile(self.game.firstTeleporter):
                    # Don't count teleporter tile as visited (same as original game)
                    if new_pos in self.visited_tiles:
                        self.visited_tiles.remove(new_pos)
                        self.complete_tiles -= 1
                    # Teleport to second teleporter
                    self.player.movetoCoordinate(
                        self.game.secondTeleporter.x, self.game.secondTeleporter.y
                    )
                    self.can_teleport = False
                    self.game.canTeleport = False
                    reward += self.reward_config.get("teleport_reward", 0.0)
                    # Track the new position after teleporting
                    teleported_pos = (self.player.x, self.player.y)
                    if teleported_pos not in self.visited_tiles:
                        self.visited_tiles.add(teleported_pos)
                        self.complete_tiles += 1
                        reward += self.reward_config["new_tile_reward"]

                # Check if player stepped on second teleporter
                elif self.player.collideWithTile(self.game.secondTeleporter):
                    # Don't count teleporter tile as visited (same as original game)
                    if new_pos in self.visited_tiles:
                        self.visited_tiles.remove(new_pos)
                        self.complete_tiles -= 1
                    # Teleport to first teleporter
                    self.player.movetoCoordinate(
                        self.game.firstTeleporter.x, self.game.firstTeleporter.y
                    )
                    self.can_teleport = False
                    self.game.canTeleport = False
                    reward += self.reward_config.get("teleport_reward", 0.0)
                    # Track the new position after teleporting
                    teleported_pos = (self.player.x, self.player.y)
                    if teleported_pos not in self.visited_tiles:
                        self.visited_tiles.add(teleported_pos)
                        self.complete_tiles += 1
                        reward += self.reward_config["new_tile_reward"]

            # Check for death (stuck)
            if self.player.checkDeath():
                reward += self.reward_config["death_penalty"]
                terminated = True
                # Reset player position (simulate death)
                self._load_map()

        else:
            # Penalty for invalid move
            reward += self.reward_config["invalid_move_penalty"]

        # Update sprites (including player animation)
        self.all_sprites.update()
        self.score_sprites.update()  # Update player sprite
        self.updating_block_group.update()

        # Get new observation
        if self.settings["use_coord_state_representation"]:
            obs = self._get_coord_obs()
        elif self.settings["use_image_state_representation"]:
            obs = self._get_obs_as_img()
        else:
            obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Can't render in headless mode
            if self.headless:
                return

            # Ensure screen is initialized and visible
            # Always recreate the display to ensure it's visible
            try:
                self.screen = pg.display.set_mode((WIDTH, HEIGHT))
                pg.display.set_caption("Thin Ice - Gymnasium")
            except Exception as e:
                print(f"Error initializing display: {e}")
                import traceback

                traceback.print_exc()
                return

            # Clear screen with background color
            self.screen.fill(BGCOLOR)

            # Update all sprites before drawing (ensures positions are correct)
            # This is critical - sprites need to update their rect positions
            self.all_sprites.update()
            self.updating_block_group.update()
            self.score_sprites.update()

            # Debug: Print sprite counts (uncomment to debug)
            # print(f"Debug: all_sprites={len(self.all_sprites)}, score_sprites={len(self.score_sprites)}, player={self.player is not None}")

            # Draw all game sprites (order matters - draw background first)
            # Draw tiles, walls, items, etc. (all game sprites)
            if self.all_sprites and len(self.all_sprites) > 0:
                self.all_sprites.draw(self.screen)

            # Draw moving blocks
            if self.updating_block_group and len(self.updating_block_group) > 0:
                self.updating_block_group.draw(self.screen)

            # Draw player and score sprites on top
            # Player is in score_sprites, so draw it last to ensure visibility
            if self.score_sprites and len(self.score_sprites) > 0:
                self.score_sprites.draw(self.screen)

            # Debug: Draw a border to verify the full screen is visible
            pg.draw.rect(self.screen, (255, 255, 255), (0, 0, WIDTH, HEIGHT), 2)

            # If no sprites are being drawn, draw a test rectangle to verify rendering works
            if len(self.all_sprites) == 0 and len(self.score_sprites) == 0:
                pg.draw.rect(self.screen, (255, 0, 0), (10, 10, 100, 100))
                print(
                    "WARNING: No sprites to draw! This indicates a problem with sprite initialization."
                )

            # Update display
            pg.display.flip()

            # Handle pygame events to keep window responsive
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pass  # Don't quit, just ignore

        elif self.render_mode == "ansi":
            # ASCII rendering
            grid = self._get_obs().reshape(self.grid_height, self.grid_width)
            symbols = {0: ".", 1: "#", 2: "I", 3: "~", 4: "P", 5: "E", 6: "K", 7: "H"}
            print("\n" + "=" * self.grid_width)
            for row in grid:
                print("".join(symbols.get(int(cell), "?") for cell in row))
            print("=" * self.grid_width + "\n")

    def close(self):
        """Clean up resources"""
        if self.screen:
            pg.display.quit()
        # Don't quit pygame completely as it might be used elsewhere
        
    def get_wall_mask(self) -> np.ndarray:
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        for wall in self.walls:
            if 0 <= wall.x < self.grid_width and 0 <= wall.y < self.grid_height:
                grid[wall.y, wall.x] = 1
        return grid
    
    def get_player_loc_from_state(self, state) -> tuple[np.ndarray, Optional[str]]:
        """
        Only implemented for has key or not and keyhole unlocked or not as extra 
        information as those are the most relevant.
        """
        if self.settings["use_coord_state_representation"]:
            has_key_str = "Has Key" if state[2] else "No Key"
            keyhole_unlocked_str = "Keyhole Unlocked" if state[3] else "Keyhole Locked"
            info_str = has_key_str + "," + keyhole_unlocked_str
            return np.array([state[1], state[0]]), info_str
        else:
            assert 4 in state # Check that the player is in the state
            has_key_str = "Has Key" if 6 in state else "No Key"
            keyhole_unlocked_str = "Keyhole Unlocked" if 7 in state else "Keyhole Locked"
            info_str = has_key_str + "," + keyhole_unlocked_str
            grid_shape = (self.grid_height, self.grid_width)
            player_loc = np.unravel_index(np.argwhere(state == 4)[0,0], grid_shape)
            np.array(player_loc), info_str


# Register the environment
# Note: This will be registered when the module is imported
# You can also register it manually in your test script
