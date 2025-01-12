import numpy as np
from typing import Optional, Tuple
from game_utils import PLAYER1, PLAYER2, BoardPiece, PlayerAction, SavedState
from .node import Node

def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[PlayerAction, Optional[SavedState]]:
    """Generate moves using MCTS

    Args:
        board (np.ndarray): current game board
        player (BoardPiece): current player making the move
        saved_state (Optional[SavedState]): saved state for MCTS

    Returns:
        Tuple[PlayerAction, Optional[SavedState]]: selected move and updated saved state
    """
    Node.set_root(board, player)
    num_simulations = 7000
    
    for _ in range(num_simulations):
        leaf = Node.current_root
        while leaf.children:
            leaf = leaf.select_child()

        if leaf.total_simulations == 0: # Ich glaube hier könnte Ineffizient sein
            result = leaf.simulate()
            leaf.backpropagate(result)
        else:
            leaf.expand()

    return Node.best_move(), saved_state