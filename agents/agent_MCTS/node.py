import numpy as np
from game_utils import PLAYER1, PLAYER2, BoardPiece, GameState, PlayerAction, MoveStatus, apply_player_action, check_move_status, check_end_state, connected_four


# Entferne Checks aus Simulation und pass weiter an
# Simulate an Backpropagation anpassen sodass nur Win zurückgegeben wird und nicht permanente Perspektivenwechsel
# Wurzel 2
# Backpropagation anpassen
# 
class Node:
    """Class representing a node in the Monte Carlo Tree Search for Connect Four"""

    current_root = None

    def __init__(self, board: np.ndarray, parent=None, player: BoardPiece = None):
        self.board = board.copy()
        self.parent = parent
        self.children = []
        self.total_simulations = 0
        self.win_simulations = 0
        self.player = player
        self.is_terminal = check_end_state(self.board, self.player) == GameState.IS_WIN
        # Attribut hinzufügen der checkt ob eine Node gewinnt, somit müssen Siblings nicht mehr überprüft werden und Selection kann da gestoppt werden
        # Nur simuliert wenn noch kein Endstate gibt, wenn Endstate einfach backpropagaten

    @classmethod
    def set_root(cls, board: np.ndarray, player: BoardPiece):
        """Sets root node

        Args:
            board (np.ndarray): game board
            player (BoardPiece): current player
        """
        cls.current_root = Node(board, player=player)

    def expand(self):
        """Expand node by creating child nodes for valid moves"""
        if self.is_terminal:
            return
        
        valid_moves = [
            col for col in range(self.board.shape[1])
            if check_move_status(self.board, PlayerAction(col)) == MoveStatus.IS_VALID
        ]

        for col in valid_moves:
            new_board = self.board.copy()
            apply_player_action(new_board, col, self.player)
            child = Node(new_board, parent=self, player=PLAYER1 if self.player == PLAYER2 else PLAYER2)
            self.children.append(child)

    def simulate(self, max_depth=20):
        """Simulate random game from selected node

        Args:
            max_depth (int): Maximum simulation depth

        Returns:
            bool: True if simulated game results in win, False orwise
        """
        board_copy = self.board.copy()
        current_player = self.player
        depth = 0
        
        if self.is_terminal:
            return check_end_state(board_copy, self.player) == GameState.IS_WIN

        while depth < max_depth: # Eventuell kann Tiefe entfernt werden
            valid_moves = [
                col for col in range(board_copy.shape[1])
                if check_move_status(board_copy, PlayerAction(col)) == MoveStatus.IS_VALID
            ]
            if not valid_moves:
                return False

            for col in valid_moves:
                temp_board = board_copy.copy()
                apply_player_action(temp_board, col, current_player)
                """ if check_end_state(temp_board, current_player) == GameState.IS_WIN: ## Warum zweimal apply_player_action ? 
                    apply_player_action(board_copy, col, current_player)
                    return True """

            apply_player_action(board_copy, np.random.choice(valid_moves), current_player)
            current_player = PLAYER1 if current_player == PLAYER2 else PLAYER2
            depth += 1

        return check_end_state(board_copy, self.player) == GameState.IS_WIN

    def backpropagate(self, result: bool):
        """Backpropagation of simulated boards

        Args:
            result (bool): board result from simulation
        """
        _result = not result
        self.total_simulations += 1
        if _result:
            self.win_simulations += 1
        if self.parent:
            self.parent.backpropagate(result)

    def select_child(self):
        """Select child with best Upper Confidence Bound (UCB) value (Otherwise MCTS filled board up from left to right with no strategy)
           Reference: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation

        Returns:
            Node: selected child node
        """
        def ucb(child):
            if child.total_simulations == 0:
                return float('inf')
            exploitation = child.win_simulations / child.total_simulations
            exploration = np.sqrt(np.log(self.total_simulations) / child.total_simulations)
            return exploitation + np.sqrt(2) * exploration

        return max(self.children, key=ucb)

    @classmethod
    def best_move(cls):
        """Return column of best move from  root

        Returns://
            PlayerAction: best move from simulated games
        """
        best_child = max(cls.current_root.children, key=lambda c: c.total_simulations)
        for col in range(cls.current_root.board.shape[1]):
            if not np.array_equal(cls.current_root.board[:, col], best_child.board[:, col]):
                return PlayerAction(col)
        return PlayerAction(np.random.choice(range(cls.current_root.board.shape[1])))