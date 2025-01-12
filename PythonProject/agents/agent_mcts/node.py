from __future__ import annotations
from typing import Optional
from game_utils import check_end_state, BoardPiece, GameState, BOARD_COLS, apply_player_action, PLAYER1, PLAYER2, check_move_status, PlayerAction, MoveStatus
import numpy as np
import math
import random

class Node:
    """A class representing a node in the Monte Carlo Tree Search for Connect Four.

    Attributes:
        current_root (Node): The current root of the search tree.
        player (BoardPiece): The player whose move is being evaluated.
        opponent (BoardPiece): The opponent of the player.
        parent (Node): The parent node of this node.
        chosen_column (int): The column chosen to reach this node.
        board (np.ndarray): The board state at this node.
        children (list[Node]): List of child nodes.
        win_simulations (int): Number of winning simulations from this node.
        total_simulations (int): Total number of simulations from this node.
        value (float): The value of the node based on simulations.
        weight (int): The weight of the node used for choosing moves.
        move_made_by (BoardPiece): The player who made the move to reach this node.
    """

    current_root: Node = None# von klassen auf instanz ebene besser
    player: BoardPiece = None
    opponent: BoardPiece = None

    def __init__(self, parent: Node, chosen_column: int, board: np.ndarray):
        """Initializes a new node.

        Args:
            parent (Node): The parent node.
            chosen_column (int): The column chosen to reach this node.
            board (np.ndarray): The board state at this node.
        """
        self.parent = parent
        self.chosen_column = chosen_column
        self.board = board.copy()
        self.children: list[Node] = []
        self.win_simulations = 0
        self.total_simulations = 0
        self.value = 0
        self.weight = 0

        if self.parent is not None:
            self.move_made_by = PLAYER1 if self.parent.move_made_by == PLAYER2 else PLAYER2
            self.parent.children.append(self)
            apply_player_action(self.board, self.chosen_column, self.move_made_by)
        else:
            self.move_made_by = PLAYER1 if self.player == PLAYER2 else PLAYER2

    def set_value(self):# properties statt set/get anschauen zum manipulieren von attrbuten
        """Calculates and sets the value for the node using the Upper Confidence Bound (UCB) formula."""
        if self.total_simulations == 0 or check_end_state(self.board,self.player) != GameState.STILL_PLAYING or check_end_state(self.board,Node.opponent) != GameState.STILL_PLAYING:
            self.value = 0
            return
        self.value = (self.win_simulations / self.total_simulations) + \
                     (math.sqrt(2) * math.sqrt(math.log(self.parent.total_simulations) / self.total_simulations))

    @classmethod
    def update_root_by_board(cls, board: np.ndarray):
        """Updates the current root based on the board state.

        Args:
            board (np.ndarray): The current board state.
        """
        if cls.current_root is not None:
            for child in cls.current_root.children:
                if np.array_equal(child.board, board):
                    cls.current_root = child
                    return
        
        cls.current_root = Node(None, 0, board)

    @classmethod
    def get_leaf_by_best_value(cls) -> Node:
        """Finds the leaf node with the best value.

        Returns:
            Node: The leaf node with the highest value.
        """
        return cls.current_root.choose_next_node_by_value()

    def choose_next_node_by_value(self) -> Node:
        """Selects the next node based on the highest value.

        Returns:
            Node: The node with the highest value among the children.
        """
        if self == Node.current_root and len(self.children) != 7: #was passiert wenn ein parent höheres value ha als seine kinder aber schon 7 kinder hat? Schau und mach besser
            return self

        chosen_child = False
        max_value = self.value
        max_child = self
        for child in self.children:
            if child.value >= max_value:
                max_child = child
                max_value = child.value
                chosen_child = True

        if chosen_child:
            return max_child.choose_next_node_by_value() if max_child.children else max_child
        return self

    def update_ancestors(self, win: Optional[bool] = False):
        """Updates the ancestors of this node based on the simulation result.

        Args:
            win (Optional[bool]): Whether the simulation resulted in a win.
        """
        if self == Node.current_root:
            return
        else:
            if win:        
                self.parent.total_simulations += 1
                self.parent.win_simulations += 1
                self.set_weight(True)
                self.set_value()
                self.parent.update_ancestors(True)
                return
            self.parent.total_simulations += 1
            self.set_weight()
            self.set_value()
            self.parent.update_ancestors()

    @classmethod
    def set_player(cls, player: BoardPiece):
        """Sets the current player for the game and his opponent.

        Args:
            player (BoardPiece): The player piece (PLAYER1 or PLAYER2).
        """
        cls.player = player
        cls.opponent = PLAYER1 if player == PLAYER2 else PLAYER2

    def create_children(self):
        """Creates child nodes by simulating moves up to a end node.
        """
    
        if check_end_state(self.board,self.move_made_by) == GameState.STILL_PLAYING:
            if len(self.children) == 7:
                random.choice(self.children).create_children()
                return
            columns = list(range(0,7))
            is_not_valid = True
            while(is_not_valid):
                if not columns:
                    random.choice(self.children).create_children()
                    return
                column = random.choice(columns)
                columns.remove(column)
                if self.child_not_generated(column) and check_move_status(self.board, PlayerAction(column)) == MoveStatus.IS_VALID:
                    is_not_valid = False
            Node(self,column,self.board).create_children()     
            return
        
        if check_end_state(self.board,Node.player) == GameState.IS_WIN:
            self.update_ancestors(True)
        else:
            self.update_ancestors()
 

    def choose_child_as_move_by_weight(self) -> Node:
        """Selects the best move based on node weights and makes it the new root.

        Returns:
            Node: The child node with the highest weight.
        """
        current_child = random.choice(self.children)
        for child in self.children:
            if child.weight > current_child.weight:
                current_child = child
        Node.update_root_by_board(current_child.board)
        return current_child

    def set_weight(self, win: bool = False):
        """Calculates and sets the weight for the current node.

        Args:
            win (bool): Whether the simulation resulted in a win.
        """
        win_player = check_end_state(self.board, self.player) == GameState.IS_WIN
        win_opponent = check_end_state(self.board, Node.opponent) == GameState.IS_WIN

        self.weight -= 100

        if win:
            self.weight += 10000
        if win_player:
            self.weight += 3000000
        if win_opponent:
            self.weight -= 3000000

        if self.move_made_by == self.player:
            if self.weight <= -1000000:
                count_no_losses = sum(1 for child in self.parent.children if child.weight > -1000000)
                if count_no_losses < 1:
                    self.parent.weight = min(self.weight, self.parent.weight)
                    return
            self.parent.weight = max(self.weight, self.parent.weight)

        if self.move_made_by == Node.opponent:
            self.parent.weight = min(self.weight, self.parent.weight)

    def child_not_generated(self, column: int)->bool:
        for child in self.children:
            if child.chosen_column == column:
                return False
        return True