import numpy as np
from typing import Optional, Tuple
from game_utils import PLAYER1, PLAYER2, BoardPiece, PlayerAction, SavedState
from .node import Node
import torch

def generate_move_mcts(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], model: torch.nn.Module
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generate moves using MCTS with neural network guidance.

    Args:
        board (np.ndarray): Current game board.
        player (BoardPiece): Current player making the move.
        saved_state (Optional[SavedState]): Saved state for MCTS.
        model (torch.nn.Module): Trained AlphaZero model.

    Returns:
        Tuple[PlayerAction, Optional[SavedState]]: Selected move and updated saved state.
    """
    Node.set_root(board, player)
    model.eval()  # Ensure the model is in evaluation mode
    
    num_simulations = 7000
    
    for _ in range(num_simulations):
        leaf = Node.current_root
        # Selection: Traverse the tree using UCB until a leaf node
        while leaf.children:
            leaf = leaf.select_child()

        if leaf.total_simulations == 0:
            # Evaluation: Use the neural network to guide the expansion
            board_tensor = torch.tensor(leaf.board, dtype=torch.float32).unsqueeze(0)
            policy, value = model(board_tensor)
            policy = policy.squeeze().detach().numpy()
            value = value.item()
            # Backpropagate the value from the network
            leaf.backpropagate(value > 0)  # Convert value to a binary win/loss signal
        else:
            # Expansion: Add child nodes guided by the policy
            leaf.expand(policy)
    
    return Node.best_move(), saved_state
