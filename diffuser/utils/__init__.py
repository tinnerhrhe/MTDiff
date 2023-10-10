from .serialization import *
from .training import *
from .progress import *
from .setup import *
from .config import *
from .rendering import *
from .arrays import *
from .colab import *
from .logger import *
from gym.envs.registration import register
register(
    id='maze2d-1',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_1,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-2',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_2,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-3',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_3,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-4',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_4,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-5',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_5,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-6',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_6,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-7',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_7,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-8',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_8,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
