import numpy as np
from typing import Optional, Tuple
import torch.nn as nn

from game_utils import (
    PLAYER1,
    PLAYER2,
    BoardPiece,
    PlayerAction,
    SavedState,
)
from .node import Node


def generate_move_mcts(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[object] = None,
    num_simulations: int = 7000,
    model: Optional[nn.Module] = None,
):
    """
    Generates the next move for a player using the Monte Carlo Tree Search (MCTS) algorithm.

    Args:
        board (np.ndarray): The current state of the game board.
        player (BoardPiece): The current player making the move (PLAYER1 or PLAYER2).
        saved_state (Optional[SavedState]): An optional saved state for the game (not utilized here).
        model (Optional[nn.Module]): Optional neural network model.

    Returns:
        Tuple[PlayerAction, Optional[SavedState]]: The column index where the player will place their piece
        and the (unchanged) saved state.
    """
    root = Node(
        board=board,
        parent=None,
        player_to_move=player,
        root_player=player,
        net=model,
    )

    for _ in range(num_simulations):
        node = root
        while not node.is_terminal and not node.untried_moves and node.children:
            node = node.select_child()

        if not node.is_terminal:
            child = node.expand()
            if child is not None:
                node = child

        result = node.rollout()

        node.backpropagate(result)

    chosen_move = root.best_move()
    return chosen_move, saved_state


def generate_move_mcts_with_policy(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[object] = None,
    model: Optional[nn.Module] = None,
    num_simulations: int = 700,
) -> Tuple[PlayerAction, np.ndarray, Optional[object]]:
    """
    Generates the next move for a player using the Monte Carlo Tree Search (MCTS) algorithm with policy distribution.

    Args:
        board (np.ndarray): The current state of the game board.
        player (BoardPiece): The current player making the move (PLAYER1 or PLAYER2).
        saved_state (Optional[SavedState]): An optional saved state for the game (not utilized here).
        model (Optional[nn.Module]): Optional neural network model.

    Returns:
        Tuple[PlayerAction, Optional[SavedState]]: The column index where the player will place their piece
        and the (unchanged) saved state.
    """
    root = Node(
        board=board,
        parent=None,
        player_to_move=player,
        root_player=player,
        net=model,
    )

    for _ in range(num_simulations):
        node = root
        while not node.is_terminal and not node.untried_moves and node.children:
            node = node.select_child()

        if not node.is_terminal:
            child = node.expand()
            if child is not None:
                node = child

        result = node.rollout()

        node.backpropagate(result)

    visit_counts = np.zeros(
        board.shape[1], dtype=np.float32
    ) 
    for child in root.children:
        col = child.column_played
        visit_counts[col] = child.total_simulations

    if visit_counts.sum() > 1e-6:
        visit_distribution = visit_counts / visit_counts.sum()
    else:
        visit_distribution = visit_counts

    chosen_move = chosen_move = PlayerAction(
        np.random.choice(np.arange(len(visit_distribution)), p=visit_distribution)
    )  

    return chosen_move, visit_distribution, saved_state
