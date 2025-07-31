from __future__ import annotations
from enum import Enum
import numpy as np

class Pos(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class ColorNode:
    next: ColorNode
    prev: ColorNode
    next_relative_pos: Pos
    prev_relative_pos: Pos
    color: np.ndarray