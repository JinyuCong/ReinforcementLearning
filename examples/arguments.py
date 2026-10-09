__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]
'''
Specify parameters of the env
'''
from typing import Union
import numpy as np
import argparse

parser = argparse.ArgumentParser("Grid World Environment")

## ==================== User settings ===================='''
# 原 5x5 地图：
#   env-size=(5, 5), start-state=(2, 2), target-state=(4, 4)
#   forbidden-states=[(2, 1), (3, 3), (1, 3)]

# 10x10 迷宫（坐标为 (x, y)，x 是列，y 是行）：
#      x: 0 1 2 3 4 5 6 7 8 9
#   y=0   S . . . . # . . . .
#   y=1   . # # # . # . # # .
#   y=2   . # . . . # . # . .
#   y=3   . # . # # # . # . #
#   y=4   . . . # . . . . . .
#   y=5   # # . # . # # # # .
#   y=6   . . . . . # . . . .
#   y=7   . # # # . # . # # .
#   y=8   . . . # . . . # T .
#   y=9   # # . . . # . . . .
#   S=起点 T=终点 #=禁区，最短路径 18 步，有多条路线和若干死胡同

# specify the number of columns and rows of the grid world
parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], default=(10, 10) )

# specify the start state
parser.add_argument("--start-state", type=Union[list, tuple, np.ndarray], default=(0, 0))

# specify the target state
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=(8, 8))

# sepcify the forbidden states
parser.add_argument("--forbidden-states", type=list, default=[
    (5, 0),
    (1, 1), (2, 1), (3, 1), (5, 1), (7, 1), (8, 1),
    (1, 2), (5, 2), (7, 2),
    (1, 3), (3, 3), (4, 3), (5, 3), (7, 3), (9, 3),
    (3, 4),
    (0, 5), (1, 5), (3, 5), (5, 5), (6, 5), (7, 5), (8, 5),
    (5, 6),
    (1, 7), (2, 7), (3, 7), (5, 7), (7, 7), (8, 7),
    (3, 8), (7, 8),
    (0, 9), (1, 9), (5, 9),
])

# sepcify the reward when reaching target
parser.add_argument("--reward-target", type=float, default = 10)

# sepcify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default = -5)

# sepcify the reward for each step
parser.add_argument("--reward-step", type=float, default = -1)
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
parser.add_argument("--action-space", type=list, default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] )  # down, right, up, left, stay           
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--animation-interval", type=float, default = 0.2)
## ==================== End of Advanced settings ====================


args = parser.parse_args()     
def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    if not (isinstance(env_size, tuple) or isinstance(env_size, list) or isinstance(env_size, np.ndarray)) and len(env_size) != 2:
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    
    for i in range(2):
        assert start_state[i] < env_size[i]
        assert target_state[i] < env_size[i]
        for j in range(len(forbidden_states)):
            assert forbidden_states[j][i] < env_size[i]
try:
    validate_environment_parameters(args.env_size, args.start_state, args.target_state, args.forbidden_states)
except ValueError as e:
    print("Error:", e)