from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from game_utils import (
    PLAYER1,
    PLAYER2,
    BoardPiece,
    GameState,
    PlayerAction,
    MoveStatus,
    apply_player_action,
    check_move_status,
    check_end_state,
)

def other_player(player: BoardPiece) -> BoardPiece:
    return PLAYER1 if player == PLAYER2 else PLAYER2

class Node:
    """Class representing a node in the Monte Carlo Tree Search for Connect Four with optional neural network insertion"""

    def __init__(
        self,
        board: np.ndarray,
        parent: Optional["Node"],
        player_to_move: BoardPiece,
        root_player: BoardPiece,
        last_move_player: Optional[BoardPiece] = None,
        column_played: Optional[int] = None,
        net: Optional[nn.Module] = None,
    ):
        self.board = board.copy()
        self.parent = parent
        self.player_to_move = player_to_move
        self.last_move_player = last_move_player
        self.column_played = column_played
        self.root_player = root_player
        self.net = net

        self.children = []
        self.untried_moves = [
            col
            for col in range(self.board.shape[1])
            if check_move_status(self.board, PlayerAction(col)) == MoveStatus.IS_VALID
        ]

        self.total_simulations = 0
        self.win_simulations = 0

        if self.last_move_player is not None:
            self.is_terminal = check_end_state(self.board, self.last_move_player) != GameState.STILL_PLAYING
        else:
            self.is_terminal = any(
                check_end_state(self.board, player) != GameState.STILL_PLAYING
                for player in (PLAYER1, PLAYER2)
            )

        self.policy = None 
        self.value_est = None

    def expand(self) -> Optional["Node"]:
        """
        Expand node by creating child node for untried valid move.
        If neural network is available, use it to get policy probabilities.
        """
        if self.is_terminal or not self.untried_moves:
            return None

        if self.net is not None and self.policy is None:
            with torch.no_grad():
                board_tensor = torch.tensor(self.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('cuda')
                policy_tensor, value_tensor = self.net(board_tensor)
            policy_arr = policy_tensor.squeeze(0).cpu().numpy()
            value_est = value_tensor.item()

            self.policy = policy_arr
            self.value_est = value_est

            self.untried_moves.sort(key=lambda col: self.policy[col], reverse=True)

        col = self.untried_moves.pop(0)
        new_board = self.board.copy()
        apply_player_action(new_board, col, self.player_to_move)

        child_node = Node(
            board=new_board,
            parent=self,
            player_to_move=other_player(self.player_to_move),
            root_player=self.root_player,
            last_move_player=self.player_to_move,
            column_played=col,
            net=self.net,
        )
        self.children.append(child_node)
        return child_node

    def rollout(self) -> bool:
        """
        Random rollout from this node or if network is available, use it to estimate value.
        For simplicity sakes draw is considered a loss.
        
        Returns:
            bool: result of the rollout
        """
        if self.is_terminal and self.last_move_player is not None:
            return (
                check_end_state(self.board, self.last_move_player) == GameState.IS_WIN
                and self.last_move_player == self.root_player
            )

        if self.value_est is not None:
            if self.player_to_move == self.root_player:
                return (self.value_est > 0.0)
            else:
                return (self.value_est < 0.0)

        board_copy = self.board.copy()
        current_player = self.player_to_move

        while True:
            valid_moves = [
                col
                for col in range(board_copy.shape[1])
                if check_move_status(board_copy, PlayerAction(col)) == MoveStatus.IS_VALID
            ]
            if not valid_moves:
                return False

            chosen_col = np.random.choice(valid_moves)
            apply_player_action(board_copy, chosen_col, current_player)

            if check_end_state(board_copy, current_player) == GameState.IS_WIN:
                return (current_player == self.root_player)
            current_player = other_player(current_player)

    def backpropagate(self, result: bool):
        """
        Backpropagation of simulated boards
        
        Args:
            result (bool): board result from simulation
        """
        self.total_simulations += 1
        if result:
            self.win_simulations += 1

        if self.parent is not None:
            self.parent.backpropagate(result)

    def select_child(self) -> "Node":
        """
        "Select child with best Upper Confidence Bound (UCB) value (Otherwise MCTS filled board up from left to right with no strategy)
           Reference: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation

        Returns:
            Node: selected child node
        """
        def ucb(child: Node) -> float:
            if child.total_simulations == 0:
                return float("inf")
            exploitation = child.win_simulations / child.total_simulations
            exploration = np.sqrt(np.log(self.total_simulations) / child.total_simulations)
            return exploitation + np.sqrt(2) * exploration

        return max(self.children, key=ucb)

    def best_move(self) -> PlayerAction:
        """
        Return column of best move from root

        Returns:
            PlayerAction: best move from simulated games
        """
        if not self.children:
            if not self.untried_moves:
                return PlayerAction(0)
            return PlayerAction(np.random.choice(self.untried_moves))

        best_child = max(self.children, key=lambda c: c.total_simulations)
        return PlayerAction(best_child.column_played)
