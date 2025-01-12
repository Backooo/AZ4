from argparse import Action
from typing import Optional, Tuple
import numpy as np
from game_utils import BoardPiece, MoveStatus, PlayerAction, SavedState, check_move_status

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
        valid_moves = [
            col for col in range(board.shape[1])
            if check_move_status(board, PlayerAction(col)) == MoveStatus.IS_VALID
        ]
        return PlayerAction(np.random.choice(valid_moves)), saved_state

