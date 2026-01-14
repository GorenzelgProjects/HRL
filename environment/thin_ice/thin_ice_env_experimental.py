"""
Thin Ice Gymnasium Environment - Experimental Implementation
A standalone Gymnasium environment that implements game logic directly
without relying on pygame sprites.
"""

from typing import Optional, Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import yaml


class ThinIceEnvExperimental(gym.Env):
    """
    Experimental Gymnasium environment for Thin Ice game.
    
    This implementation uses numpy arrays and simple data structures
    instead of pygame sprites, making it faster and easier to maintain.
    """
    
    metadata = {"render_modes": ["human", "rgb_array", "ansi", "semi"], "render_fps": 24}
    
    # Tile types
    EMPTY = 0
    WALL = 1
    ICE = 2
    WATER = 3
    PLAYER = 4
    EXIT = 5
    KEY = 6
    KEYHOLE = 7
    TREASURE = 8
    MOVING_BLOCK = 9
    MOVING_BLOCK_TILE = 10
    TELEPORTER_1 = 11
    TELEPORTER_2 = 12
    
    # Action mapping
    ACTION_RIGHT = 0
    ACTION_UP = 1
    ACTION_LEFT = 2
    ACTION_DOWN = 3
    
    def __init__(self, level: int = 1, render_mode: Optional[str] = None, 
                 headless: bool = True, reward_config: Optional[Dict] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the Thin Ice environment.
        
        Args:
            level: Level number to play (1-19)
            render_mode: Rendering mode ("human", "rgb_array", "ansi", "semi", or None)
            headless: If True, disable pygame display (for faster training)
            reward_config: Optional dict with reward function parameters
            config_path: Optional path to config.yaml file
        """
        super().__init__()
        
        # Load configuration from YAML if provided
        if config_path is None:
            config_path = self._resolve_path("thin_ice/config.yaml")
        
        self.config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        # Get settings from config with defaults
        env_config = self.config.get("environment", {})
        rewards_config = self.config.get("rewards", {})
        
        # Override with explicit parameters if provided (explicit params take precedence)
        if level is not None:
            self.level = level
        else:
            self.level = env_config.get("level", 1)
        
        if render_mode is not None:
            self.render_mode = render_mode
        else:
            self.render_mode = env_config.get("render_mode")
        
        # headless defaults based on render_mode if not explicitly set
        if headless is not None:
            self.headless = headless
        else:
            # If render_mode is "human", default headless to False
            if self.render_mode == "human":
                self.headless = env_config.get("headless", False)
            else:
                self.headless = env_config.get("headless", True)
        
        # Use reward_config from parameter or from config
        if reward_config is None:
            reward_config = rewards_config
        
        # Grid dimensions (from settings)
        self.tile_size = 25
        self.grid_width = 19
        self.grid_height = 17
        self.width = self.grid_width * self.tile_size
        self.height = self.grid_height * self.tile_size
        
        # Load reward configuration
        if reward_config is None:
            reward_config = {}
        self.reward_config = {
            "new_tile_reward": reward_config.get("new_tile_reward", 0.1),
            "level_completion_reward": reward_config.get("level_completion_reward", 5.0),
            "perfect_completion_bonus": reward_config.get("perfect_completion_bonus", 10.0),
            "key_collection_reward": reward_config.get("key_collection_reward", 1.0),
            "keyhole_unlock_reward": reward_config.get("keyhole_unlock_reward", 1.0),
            "treasure_collection_reward": reward_config.get("treasure_collection_reward", 2.0),
            "invalid_move_penalty": reward_config.get("invalid_move_penalty", -0.01),
            "death_penalty": reward_config.get("death_penalty", -5.0),
            "use_distance_reward": reward_config.get("use_distance_reward", False),
            "distance_reward_scale": reward_config.get("distance_reward_scale", -0.01),
        }
        
        # Action space: 4 discrete actions (0=right, 1=up, 2=left, 3=down)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Flattened grid representation
        self.observation_space = spaces.Box(
            low=0, high=12, shape=(self.grid_height * self.grid_width,), dtype=np.int32
        )
        
        # Game state
        self.grid = None
        self.player_pos = None
        self.exit_pos = None
        self.key_pos = None
        self.keyhole_pos = None
        self.treasure_pos = None
        self.moving_block_pos = None
        self.moving_block_tile_pos = None
        self.teleporter_1_pos = None
        self.teleporter_2_pos = None
        self.has_key = False
        self.can_teleport = True
        self.visited_tiles = set()
        self.total_tiles = 0
        self.complete_tiles = 0
        self.ice_tiles = set()
        self.water_tiles = set()
        
        # Level constants
        self.TREASURE_LEVEL = 3
        self.KEY_LEVEL = 9
        self.MOVING_BLOCK_LEVEL = 12
        self.TELEPORT_LEVEL = 16
        
        # Initialize pygame for rendering (only if needed)
        self.screen = None
        self.sprites_loaded = False
        self.sprite_images = {}
        self.player_frame = 16
        self.water_frame = 1
        self.key_frame = 1
        self.teleporter_frame = 1
        
        # Initialize pygame rendering if needed
        if self.render_mode == "human" and not self.headless:
            self._init_pygame_rendering()
    
    def _init_pygame_rendering(self):
        """Initialize pygame and load sprites for full rendering"""
        try:
            import pygame as pg
            import xml.etree.ElementTree as ET
            
            # Initialize pygame
            if not pg.get_init():
                pg.init()
                pg.mixer.init()
            
            # Initialize display
            if not pg.display.get_init():
                pg.display.init()
            
            # Set display mode
            self.screen = pg.display.set_mode((self.width, self.height))
            pg.display.set_caption("Thin Ice - Gymnasium (Experimental)")
            self.pg = pg
            
            # Create sprite group for rendering
            self.render_sprites = pg.sprite.Group()
            self.render_sprites_dict = {}
            
            # Load sprite images
            self._load_sprite_images()
            self.sprites_loaded = True
            
        except ImportError:
            print("Warning: pygame not available, rendering disabled")
            self.render_mode = None
        except Exception as e:
            print(f"Warning: Could not initialize pygame rendering: {e}")
            self.render_mode = None
    
    def _load_sprite_images(self):
        """Load sprite images for rendering"""
        if not hasattr(self, 'pg'):
            return
        
        pg = self.pg
        import xml.etree.ElementTree as ET
        
        # Helper to load image
        def load_image(path):
            full_path = self._resolve_path(path)
            return pg.image.load(full_path).convert_alpha()
        
        # Helper to load spritesheet
        def load_spritesheet(image_path, xml_path):
            full_image_path = self._resolve_path(image_path)
            full_xml_path = self._resolve_path(xml_path)
            
            # Use convert() like the original, not convert_alpha()
            spritesheet = pg.image.load(full_image_path).convert()
            xml_tree = ET.parse(full_xml_path)
            root = xml_tree.getroot()
            
            return spritesheet, root
        
        try:
            # Load simple images
            self.sprite_images['free'] = load_image("thin_ice/data/images/free.png")
            self.sprite_images['wall'] = load_image("thin_ice/data/images/wall.png")
            self.sprite_images['ice'] = load_image("thin_ice/data/images/ice.png")
            self.sprite_images['finish'] = load_image("thin_ice/data/images/finish.png")
            self.sprite_images['socket'] = load_image("thin_ice/data/images/socket.png")
            self.sprite_images['treasure'] = load_image("thin_ice/data/images/treasure.png")
            self.sprite_images['unused'] = load_image("thin_ice/data/images/unused.png")
            self.sprite_images['moving_block'] = load_image("thin_ice/data/images/movingBlock.png")
            self.sprite_images['moving_block_tile'] = load_image("thin_ice/data/images/movingBlockTile.png")
            
            print(f"Loaded {len([k for k in self.sprite_images.keys() if not k.endswith('_xml') and not k.endswith('_sheet')])} sprite images")
            
            # Load spritesheets
            player_sheet, player_xml = load_spritesheet(
                "thin_ice/data/images/player.png",
                "thin_ice/data/images/player.xml"
            )
            self.sprite_images['player_sheet'] = player_sheet
            self.sprite_images['player_sheet_xml'] = player_xml  # Fixed: use consistent naming
            self.player_frame = 16  # Starting frame
            
            water_sheet, water_xml = load_spritesheet(
                "thin_ice/data/images/water.png",
                "thin_ice/data/images/water.xml"
            )
            self.sprite_images['water_sheet'] = water_sheet
            self.sprite_images['water_sheet_xml'] = water_xml  # Fixed: use consistent naming
            self.water_frame = 1
            
            key_sheet, key_xml = load_spritesheet(
                "thin_ice/data/images/key.png",
                "thin_ice/data/images/key.xml"
            )
            self.sprite_images['key_sheet'] = key_sheet
            self.sprite_images['key_sheet_xml'] = key_xml  # Fixed: use consistent naming
            self.key_frame = 1
            
            teleporter_sheet, teleporter_xml = load_spritesheet(
                "thin_ice/data/images/teleporter.png",
                "thin_ice/data/images/teleporter.xml"
            )
            self.sprite_images['teleporter_sheet'] = teleporter_sheet
            self.sprite_images['teleporter_sheet_xml'] = teleporter_xml  # Fixed: use consistent naming
            self.teleporter_frame = 1
            
        except Exception as e:
            print(f"Warning: Could not load some sprites: {e}")
            import traceback
            traceback.print_exc()
            # Continue with basic rendering
    
    def _get_sprite_from_sheet(self, sheet_name, frame_number):
        """Extract a sprite from a spritesheet"""
        if not hasattr(self, 'pg') or sheet_name not in self.sprite_images:
            return None
        
        pg = self.pg
        import xml.etree.ElementTree as ET
        
        try:
            spritesheet = self.sprite_images[sheet_name]
            xml_root = self.sprite_images[sheet_name + '_xml']
            
            # Match the original format: ".//*[@name='%s.png']" % frameNumber
            frame = xml_root.find(f".//*[@name='{frame_number}.png']")
            if frame is None:
                return None
            
            w, h = int(frame.attrib['w']), int(frame.attrib['h'])
            x, y = int(frame.attrib['x']), int(frame.attrib['y'])
            
            # Use regular Surface like the original, not SRCALPHA
            image = pg.Surface((w, h))
            image.blit(spritesheet, (0, 0), (x, y, w, h))
            return image
        except Exception as e:
            print(f"Error extracting sprite from {sheet_name} frame {frame_number}: {e}")
            return None
    
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the thin_ice directory"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if path.startswith("thin_ice/"):
            path = path[len("thin_ice/"):]
        return os.path.normpath(os.path.join(base_dir, path))
    
    def _load_map(self):
        """Load the current level map"""
        # Clear previous state
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        self.player_pos = None
        self.exit_pos = None
        self.key_pos = None
        self.keyhole_pos = None
        self.treasure_pos = None
        self.moving_block_pos = None
        self.moving_block_tile_pos = None
        self.teleporter_1_pos = None
        self.teleporter_2_pos = None
        self.has_key = False
        self.can_teleport = True
        self.visited_tiles = set()
        self.complete_tiles = 0
        self.ice_tiles = set()
        self.water_tiles = set()
        
        # Read map file
        map_path = self._resolve_path(f"thin_ice/data/maps/level{self.level}.txt")
        total_free = 0
        
        try:
            with open(map_path, 'r') as f:
                map_data = [line.strip() for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Level file not found: {map_path}")
        
        # Parse map
        for row, line in enumerate(map_data):
            for col, tile_char in enumerate(line):
                if col >= self.grid_width:
                    break
                
                if tile_char == 'W':
                    self.grid[row, col] = self.WALL
                elif tile_char == '0':
                    self.grid[row, col] = self.EMPTY
                elif tile_char == 'F':
                    self.grid[row, col] = self.EMPTY
                    total_free += 1
                elif tile_char == 'E':
                    self.grid[row, col] = self.EXIT
                    self.exit_pos = (col, row)
                    total_free += 1
                elif tile_char == 'I':
                    self.grid[row, col] = self.ICE
                    self.ice_tiles.add((col, row))
                    total_free += 2
                elif tile_char == 'K':
                    self.grid[row, col] = self.KEY
                    self.key_pos = (col, row)
                    total_free += 1
                elif tile_char == 'B':
                    self.grid[row, col] = self.MOVING_BLOCK_TILE
                    self.moving_block_tile_pos = (col, row)
                elif tile_char == 'T':
                    self.grid[row, col] = self.MOVING_BLOCK
                    self.moving_block_pos = (col, row)
                    total_free += 1
                elif tile_char == '%':
                    self.grid[row, col] = self.ICE
                    self.ice_tiles.add((col, row))
                    self.moving_block_pos = (col, row)
                    total_free += 2
                elif tile_char == '&':
                    self.grid[row, col] = self.MOVING_BLOCK_TILE
                    self.moving_block_tile_pos = (col, row)
                    self.key_pos = (col, row)
                elif tile_char == '!':
                    self.grid[row, col] = self.ICE
                    self.ice_tiles.add((col, row))
                    self.key_pos = (col, row)
                    total_free += 2
                elif tile_char == '1':
                    self.grid[row, col] = self.TELEPORTER_1
                    self.teleporter_1_pos = (col, row)
                elif tile_char == '2':
                    self.grid[row, col] = self.TELEPORTER_2
                    self.teleporter_2_pos = (col, row)
                elif tile_char == 'H':
                    self.grid[row, col] = self.KEYHOLE
                    self.keyhole_pos = (col, row)
                    total_free += 1
                elif tile_char == 'M':
                    self.grid[row, col] = self.EMPTY
                    if self.level > self.TREASURE_LEVEL:
                        self.treasure_pos = (col, row)
                    total_free += 1
                elif tile_char == 'P':
                    self.grid[row, col] = self.EMPTY
                    self.player_pos = (col, row)
                    total_free += 1
        
        self.total_tiles = total_free - (2 * 19)  # Subtract top/bottom menu rows
        
        if self.player_pos is None:
            raise ValueError(f"Player starting position not found in level {self.level}")
    
    def _get_obs(self) -> np.ndarray:
        """Get the current observation"""
        # Create a copy of the grid
        obs_grid = self.grid.copy()
        
        # Mark player position
        if self.player_pos:
            obs_grid[self.player_pos[1], self.player_pos[0]] = self.PLAYER
        
        # Flatten and return
        return obs_grid.flatten()
    
    def _get_info(self) -> Dict:
        """Get auxiliary information"""
        distance = 0
        if self.player_pos and self.exit_pos:
            distance = abs(self.player_pos[0] - self.exit_pos[0]) + \
                      abs(self.player_pos[1] - self.exit_pos[1])
        
        return {
            "distance": distance,
            "visited_tiles": len(self.visited_tiles),
            "total_tiles": self.total_tiles,
            "complete_tiles": self.complete_tiles,
            "has_key": self.has_key,
            "level": self.level,
        }
    
    def _can_move(self, dx: int, dy: int) -> bool:
        """Check if player can move in the given direction"""
        if not self.player_pos:
            return False
        
        new_x = self.player_pos[0] + dx
        new_y = self.player_pos[1] + dy
        
        # Check bounds
        if new_x < 0 or new_x >= self.grid_width or new_y < 0 or new_y >= self.grid_height:
            return False
        
        # Check if target is a wall
        tile_type = self.grid[new_y, new_x]
        if tile_type == self.WALL:
            return False
        
        return True
    
    def _is_stuck(self) -> bool:
        """Check if player is stuck (surrounded by walls)"""
        if not self.player_pos:
            return True
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        can_move_anywhere = False
        
        for dx, dy in directions:
            if self._can_move(dx, dy):
                can_move_anywhere = True
                break
        
        return not can_move_anywhere
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Set level if provided
        if options and "level" in options:
            self.level = options["level"]
        
        # Load map
        self._load_map()
        
        # Create render sprites if rendering
        if self.render_mode == "human" and not self.headless and self.sprites_loaded:
            self._create_render_sprites()
        
        # Get initial observation
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _create_render_sprite(self, image, x, y):
        """Helper to create a render sprite"""
        if not hasattr(self, 'pg'):
            import pygame as pg
            self.pg = pg
        else:
            pg = self.pg
        
        # Ensure image is valid
        if image is None:
            # Create a fallback surface
            image = pg.Surface((self.tile_size, self.tile_size))
            image.fill((128, 128, 128))
        
        sprite = pg.sprite.Sprite()
        sprite.image = image
        sprite.rect = sprite.image.get_rect()
        sprite.x = x
        sprite.y = y
        sprite.tile_size = self.tile_size
        sprite.rect.x = x * self.tile_size
        sprite.rect.y = y * self.tile_size
        
        def update_sprite():
            sprite.rect.x = sprite.x * sprite.tile_size
            sprite.rect.y = sprite.y * sprite.tile_size
        
        sprite.update = update_sprite
        return sprite
    
    def _create_render_sprites(self):
        """Create pygame sprite objects for rendering"""
        if not hasattr(self, 'pg'):
            # Try to initialize pygame if not already done
            try:
                import pygame as pg
                if not pg.get_init():
                    pg.init()
                self.pg = pg
            except:
                print("Warning: pygame not available for rendering")
                return
        
        pg = self.pg
        
        # Ensure sprite group exists
        if self.render_sprites is None:
            self.render_sprites = pg.sprite.Group()
        
        # Clear existing render sprites
        if self.render_sprites:
            for sprite in list(self.render_sprites):
                sprite.kill()
        self.render_sprites_dict = {}
        
        # Debug: print sprite creation info
        print(f"Creating render sprites: grid={self.grid_width}x{self.grid_height}, tile_size={self.tile_size}, sprites_loaded={self.sprites_loaded}")
        print(f"Available sprite images: {list(self.sprite_images.keys())}")
        
        # Create sprites for each tile
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                tile_type = self.grid[y, x]
                key = (x, y)
                
                # Get appropriate image
                img = None
                if tile_type == self.EMPTY:
                    if 'free' in self.sprite_images:
                        img = self.sprite_images['free'].copy()
                elif tile_type == self.WALL:
                    if 'wall' in self.sprite_images:
                        img = self.sprite_images['wall'].copy()
                elif tile_type == self.ICE:
                    if 'ice' in self.sprite_images:
                        img = self.sprite_images['ice'].copy()
                elif tile_type == self.EXIT:
                    if 'finish' in self.sprite_images:
                        img = self.sprite_images['finish'].copy()
                elif tile_type == self.KEYHOLE:
                    if 'socket' in self.sprite_images:
                        img = self.sprite_images['socket'].copy()
                elif tile_type == self.MOVING_BLOCK:
                    if 'moving_block' in self.sprite_images:
                        img = self.sprite_images['moving_block'].copy()
                elif tile_type == self.MOVING_BLOCK_TILE:
                    if 'moving_block_tile' in self.sprite_images:
                        img = self.sprite_images['moving_block_tile'].copy()
                elif tile_type == self.TELEPORTER_1 or tile_type == self.TELEPORTER_2:
                    if 'teleporter_sheet' in self.sprite_images:
                        img = self._get_sprite_from_sheet('teleporter_sheet', self.teleporter_frame)
                
                if img is not None:
                    # Scale to tile size
                    try:
                        img = pg.transform.scale(img, (self.tile_size, self.tile_size))
                        # Set colorkey for transparency
                        if tile_type in [self.EMPTY, self.EXIT]:
                            img.set_colorkey((255, 255, 255))
                        elif tile_type == self.MOVING_BLOCK:
                            img.set_colorkey((0, 0, 0))
                        
                        sprite = self._create_render_sprite(img, x, y)
                        sprite.add(self.render_sprites)
                        self.render_sprites_dict[key] = sprite
                    except Exception as e:
                        print(f"Error creating sprite for tile at ({x}, {y}): {e}")
                        # Fallback: create a colored rectangle
                        fallback_img = pg.Surface((self.tile_size, self.tile_size))
                        if tile_type == self.WALL:
                            fallback_img.fill((100, 100, 100))
                        elif tile_type == self.ICE:
                            fallback_img.fill((173, 216, 230))
                        elif tile_type == self.EXIT:
                            fallback_img.fill((0, 255, 0))
                        else:
                            fallback_img.fill((200, 200, 200))
                        sprite = self._create_render_sprite(fallback_img, x, y)
                        sprite.add(self.render_sprites)
                        self.render_sprites_dict[key] = sprite
                else:
                    # Create a fallback sprite even if image is None
                    fallback_img = pg.Surface((self.tile_size, self.tile_size))
                    if tile_type == self.WALL:
                        fallback_img.fill((100, 100, 100))
                    elif tile_type == self.ICE:
                        fallback_img.fill((173, 216, 230))
                    elif tile_type == self.EXIT:
                        fallback_img.fill((0, 255, 0))
                    else:
                        fallback_img.fill((200, 200, 200))
                    sprite = self._create_render_sprite(fallback_img, x, y)
                    sprite.add(self.render_sprites)
                    self.render_sprites_dict[key] = sprite
        
        # Create player sprite
        if self.player_pos:
            px, py = self.player_pos
            if 'player_sheet' in self.sprite_images:
                player_img = self._get_sprite_from_sheet('player_sheet', self.player_frame)
                if player_img:
                    player_img = pg.transform.scale(player_img, (self.tile_size, self.tile_size))
                    player_img.set_colorkey((30, 45, 255))  # BLUE
                    player_sprite = self._create_render_sprite(player_img, px, py)
                    player_sprite.add(self.render_sprites)
                    self.render_sprites_dict['player'] = player_sprite
                    print(f"Created player sprite at ({px}, {py})")
                else:
                    print(f"Warning: Could not get player sprite frame {self.player_frame}")
                    # Fallback: yellow rectangle for player
                    fallback_img = pg.Surface((self.tile_size, self.tile_size))
                    fallback_img.fill((255, 255, 0))  # Yellow
                    player_sprite = self._create_render_sprite(fallback_img, px, py)
                    player_sprite.add(self.render_sprites)
                    self.render_sprites_dict['player'] = player_sprite
            else:
                print("Warning: player_sheet not loaded, using fallback")
                # Fallback: yellow rectangle for player
                fallback_img = pg.Surface((self.tile_size, self.tile_size))
                fallback_img.fill((255, 255, 0))  # Yellow
                player_sprite = self._create_render_sprite(fallback_img, px, py)
                player_sprite.add(self.render_sprites)
                self.render_sprites_dict['player'] = player_sprite
        
        # Create key sprite
        if self.key_pos and 'key_sheet' in self.sprite_images:
            kx, ky = self.key_pos
            key_img = self._get_sprite_from_sheet('key_sheet', self.key_frame)
            if key_img:
                key_img = pg.transform.scale(key_img, (self.tile_size, self.tile_size))
                key_img.set_colorkey((30, 45, 255))  # BLUE
                key_sprite = self._create_render_sprite(key_img, kx, ky)
                key_sprite.add(self.render_sprites)
                self.render_sprites_dict['key'] = key_sprite
        
        # Create treasure sprite
        if self.treasure_pos and 'treasure' in self.sprite_images:
            tx, ty = self.treasure_pos
            treasure_img = self.sprite_images['treasure'].copy()
            treasure_img = pg.transform.scale(treasure_img, (self.tile_size, self.tile_size))
            treasure_img.set_colorkey((255, 255, 255))
            treasure_sprite = self._create_render_sprite(treasure_img, tx, ty)
            treasure_sprite.add(self.render_sprites)
            self.render_sprites_dict['treasure'] = treasure_sprite
        
        # Create water sprites
        for wx, wy in self.water_tiles:
            if 'water_sheet' in self.sprite_images:
                water_img = self._get_sprite_from_sheet('water_sheet', self.water_frame)
                if water_img:
                    water_img = pg.transform.scale(water_img, (self.tile_size, self.tile_size))
                    water_img.set_colorkey((255, 255, 255))
                    water_sprite = self._create_render_sprite(water_img, wx, wy)
                    water_sprite.add(self.render_sprites)
                    self.render_sprites_dict[('water', wx, wy)] = water_sprite
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        if not self.player_pos:
            return self._get_obs(), -1.0, True, False, self._get_info()
        
        # Map action to direction
        action_to_direction = {
            self.ACTION_RIGHT: (1, 0),
            self.ACTION_UP: (0, -1),
            self.ACTION_LEFT: (-1, 0),
            self.ACTION_DOWN: (0, 1),
        }
        
        dx, dy = action_to_direction[action]
        reward = 0.0
        terminated = False
        truncated = False
        
        # Try to move
        if self._can_move(dx, dy):
            # Update position
            old_pos = self.player_pos
            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy
            self.player_pos = (new_x, new_y)
            
            # Track visited tiles
            if self.player_pos not in self.visited_tiles:
                self.visited_tiles.add(self.player_pos)
                self.complete_tiles += 1
                reward += self.reward_config["new_tile_reward"]
            
            # Check if on ice
            if self.player_pos in self.ice_tiles:
                self.ice_tiles.remove(self.player_pos)
                # Ice breaks, becomes empty (water generation can be added if needed)
                self.grid[new_y, new_x] = self.EMPTY
            
            # Distance-based reward (if enabled)
            if self.reward_config["use_distance_reward"] and self.exit_pos:
                distance = abs(self.player_pos[0] - self.exit_pos[0]) + \
                          abs(self.player_pos[1] - self.exit_pos[1])
                reward += self.reward_config["distance_reward_scale"] * distance
            
            # Check if reached exit
            if self.player_pos == self.exit_pos:
                if self.complete_tiles == self.total_tiles:
                    reward += self.reward_config["perfect_completion_bonus"]
                else:
                    reward += self.reward_config["level_completion_reward"]
                terminated = True
            
            # Check for key collection
            elif self.player_pos == self.key_pos and self.key_pos:
                self.has_key = True
                self.key_pos = None
                reward += self.reward_config["key_collection_reward"]
            
            # Check for keyhole unlocking
            elif self.has_key and self.player_pos == self.keyhole_pos and self.keyhole_pos:
                self.has_key = False
                self.keyhole_pos = None
                self.grid[new_y, new_x] = self.EMPTY
                reward += self.reward_config["keyhole_unlock_reward"]
            
            # Check for treasure
            elif self.player_pos == self.treasure_pos and self.treasure_pos:
                self.treasure_pos = None
                reward += self.reward_config["treasure_collection_reward"]
            
            # Check for teleporters
            elif self.level > self.TELEPORT_LEVEL:
                if self.player_pos == self.teleporter_1_pos and self.can_teleport:
                    self.player_pos = self.teleporter_2_pos
                    self.can_teleport = False
                elif self.player_pos == self.teleporter_2_pos and self.can_teleport:
                    self.player_pos = self.teleporter_1_pos
                    self.can_teleport = False
            
            # Check if stuck (death)
            if self._is_stuck():
                reward += self.reward_config["death_penalty"]
                terminated = True
        else:
            # Invalid move
            reward += self.reward_config["invalid_move_penalty"]
        
        # Update render sprites if rendering
        if self.render_mode == "human" and self.sprites_loaded:
            self._update_render_sprites()
        
        # Get new observation
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _update_render_sprites(self):
        """Update render sprites positions and animations"""
        if not self.render_sprites or not hasattr(self, 'pg'):
            return
        
        pg = self.pg
        
        # Update animation frames
        self.player_frame = (self.player_frame + 1) % 100
        if self.player_frame >= 86:
            self.player_frame = 28
        
        if 'water_sheet' in self.sprite_images:
            self.water_frame = (self.water_frame + 1) % 50
            if self.water_frame >= 39:
                self.water_frame = 7
        
        if 'key_sheet' in self.sprite_images:
            self.key_frame = (self.key_frame + 1) % 33
            if self.key_frame >= 32:
                self.key_frame = 1
        
        if 'teleporter_sheet' in self.sprite_images:
            if self.can_teleport:
                self.teleporter_frame = (self.teleporter_frame + 1) % 22
                if self.teleporter_frame >= 21:
                    self.teleporter_frame = 1
            else:
                self.teleporter_frame = 22
        
        # Update player sprite
        if 'player' in self.render_sprites_dict and self.player_pos:
            px, py = self.player_pos
            player_sprite = self.render_sprites_dict['player']
            player_sprite.x = px
            player_sprite.y = py
            # Update player image
            player_img = self._get_sprite_from_sheet('player_sheet', self.player_frame)
            if player_img:
                player_img = pg.transform.scale(player_img, (self.tile_size, self.tile_size))
                player_img.set_colorkey((30, 45, 255))
                player_sprite.image = player_img
        
        # Update key sprite
        if 'key' in self.render_sprites_dict and self.key_pos:
            kx, ky = self.key_pos
            key_sprite = self.render_sprites_dict['key']
            key_sprite.x = kx
            key_sprite.y = ky
            # Update key image
            key_img = self._get_sprite_from_sheet('key_sheet', self.key_frame)
            if key_img:
                key_img = pg.transform.scale(key_img, (self.tile_size, self.tile_size))
                key_img.set_colorkey((30, 45, 255))
                key_sprite.image = key_img
        elif 'key' in self.render_sprites_dict and not self.key_pos:
            # Key collected, remove sprite
            self.render_sprites_dict['key'].kill()
            del self.render_sprites_dict['key']
        
        # Update treasure sprite
        if 'treasure' in self.render_sprites_dict and not self.treasure_pos:
            # Treasure collected, remove sprite
            self.render_sprites_dict['treasure'].kill()
            del self.render_sprites_dict['treasure']
        
        # Update tile sprites (e.g., when ice breaks)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                key = (x, y)
                tile_type = self.grid[y, x]
                
                # Update existing tile sprite if type changed
                if key in self.render_sprites_dict:
                    sprite = self.render_sprites_dict[key]
                    # Check if we need to update the image
                    if tile_type == self.EMPTY and 'free' in self.sprite_images:
                        img = self.sprite_images['free'].copy()
                        img = pg.transform.scale(img, (self.tile_size, self.tile_size))
                        img.set_colorkey((255, 255, 255))
                        sprite.image = img
                    elif tile_type == self.ICE and 'ice' in self.sprite_images:
                        img = self.sprite_images['ice'].copy()
                        img = pg.transform.scale(img, (self.tile_size, self.tile_size))
                        sprite.image = img
                    elif tile_type in [self.TELEPORTER_1, self.TELEPORTER_2] and 'teleporter_sheet' in self.sprite_images:
                        img = self._get_sprite_from_sheet('teleporter_sheet', self.teleporter_frame)
                        if img:
                            img = pg.transform.scale(img, (self.tile_size, self.tile_size))
                            sprite.image = img
        
        # Update water sprites
        current_water_keys = set(('water', wx, wy) for wx, wy in self.water_tiles)
        # Remove water sprites that no longer exist
        for key in list(self.render_sprites_dict.keys()):
            if isinstance(key, tuple) and len(key) == 3 and key[0] == 'water':
                if key not in current_water_keys:
                    self.render_sprites_dict[key].kill()
                    del self.render_sprites_dict[key]
        # Add new water sprites
        for wx, wy in self.water_tiles:
            water_key = ('water', wx, wy)
            if water_key not in self.render_sprites_dict and 'water_sheet' in self.sprite_images:
                water_img = self._get_sprite_from_sheet('water_sheet', self.water_frame)
                if water_img:
                    water_img = pg.transform.scale(water_img, (self.tile_size, self.tile_size))
                    water_img.set_colorkey((255, 255, 255))
                    water_sprite = self._create_render_sprite(water_img, wx, wy)
                    water_sprite.add(self.render_sprites)
                    self.render_sprites_dict[water_key] = water_sprite
            elif water_key in self.render_sprites_dict:
                # Update existing water sprite animation
                water_sprite = self.render_sprites_dict[water_key]
                water_img = self._get_sprite_from_sheet('water_sheet', self.water_frame)
                if water_img:
                    water_img = pg.transform.scale(water_img, (self.tile_size, self.tile_size))
                    water_img.set_colorkey((255, 255, 255))
                    water_sprite.image = water_img
        
        # Update all sprite positions
        self.render_sprites.update()
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "ansi" or self.render_mode == "semi":
            # ASCII rendering (semi-render mode)
            # Create a display grid
            display_grid = self.grid.copy()
            
            # Mark player position
            if self.player_pos:
                px, py = self.player_pos
                display_grid[py, px] = self.PLAYER
            
            # Symbol mapping
            symbols = {
                self.EMPTY: ".",
                self.WALL: "#",
                self.ICE: "I",
                self.WATER: "~",
                self.PLAYER: "P",
                self.EXIT: "E",
                self.KEY: "K",
                self.KEYHOLE: "H",
                self.TREASURE: "T",
                self.MOVING_BLOCK: "B",
                self.MOVING_BLOCK_TILE: "b",
                self.TELEPORTER_1: "1",
                self.TELEPORTER_2: "2",
            }
            
            # Print grid based on render mode
            if self.render_mode == "semi":
                # Semi-render mode: Experimental ASCII output matching the image format
                # Clear screen for experimental display
                import os
                try:
                    os.system('cls' if os.name == 'nt' else 'clear')
                except:
                    pass  # If clearing fails, continue anyway
                
                # Print grid (experimental format like in the image)
                for row in display_grid:
                    row_str = "".join(symbols.get(int(cell), "?") for cell in row)
                    print(row_str)
                
                # Print bottom border
                print("=" * self.grid_width)
                
            else:  # ansi mode
                # ANSI mode: Print with borders and info
                print("\n" + "=" * self.grid_width)
                for row in display_grid:
                    print("".join(symbols.get(int(cell), "?") for cell in row))
                print("=" * self.grid_width + "\n")
        
        elif self.render_mode == "human":
            # Full pygame sprite rendering (using sprite groups like original)
            if self.headless:
                return
            
            # Initialize pygame rendering if not already done
            if not self.sprites_loaded:
                self._init_pygame_rendering()
            
            if not self.screen or not hasattr(self, 'pg'):
                return
            
            pg = self.pg
            
            # Ensure render sprites are created
            if not self.render_sprites or len(self.render_sprites) == 0:
                if self.grid is not None:
                    self._create_render_sprites()
            
            # Clear screen
            self.screen.fill((40, 40, 40))  # Dark grey background (BGCOLOR)
            
            # Update render sprites (positions and animations)
            if self.render_sprites:
                self.render_sprites.update()
            
            # Draw all sprites using pygame's sprite group drawing
            # This is the same approach as the original implementation
            if self.render_sprites and len(self.render_sprites) > 0:
                self.render_sprites.draw(self.screen)
            else:
                # Debug: Draw test rectangles if no sprites
                pg.draw.rect(self.screen, (255, 0, 0), (10, 10, 100, 100))
                print(f"WARNING: No sprites to draw! render_sprites={self.render_sprites}, len={len(self.render_sprites) if self.render_sprites else 0}")
            
            # Draw player on top (if it exists in render sprites)
            # Player should already be in render_sprites, but ensure it's drawn last
            if 'player' in self.render_sprites_dict:
                player_sprite = self.render_sprites_dict['player']
                self.screen.blit(player_sprite.image, player_sprite.rect)
            
            # Debug border
            pg.draw.rect(self.screen, (255, 255, 255), (0, 0, self.width, self.height), 2)
            
            pg.display.flip()
            
            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pass
    
    def close(self):
        """Clean up resources"""
        if self.screen and hasattr(self, 'pg'):
            self.pg.display.quit()


# Register the environment
gym.register(
    id="ThinIceExperimental-v0",
    entry_point=ThinIceEnvExperimental,
    max_episode_steps=1000,
)
