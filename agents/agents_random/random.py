from argparse import Action
from typing import Optional, Tuple
import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    return Action, saved_state

