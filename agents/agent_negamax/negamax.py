from typing import Optional, Tuple
import numpy as np
from game_utils import *

"""
   Negamax agent that utilisies alpha-beta pruning to employ a viable Connect4 strategy.
   
   This module contains multiple functions that are used together to implement the negamax algorithm with alpha-beta pruning.
   
   Functions:
        - heuristic_value: apply a heuristic value to each simulated board state.
        - get_opponent: return the opponent of the current player.
        - evaluate_position: apply a score to a single position on the board.
        - count_in_direction: count the number of consecutive pieces in a specific direction to determine winning and losing positions.
        - calculate_line_score: apply a score to a move on the basis of connected pieces.
        - check_terminal_state: evaluate if next move will win, lose or draw the game.
        - negamax: employ all the above functions to implement the negamax algorithm with alpha-beta pruning.
        - dynamic_depth: adjust search depth automatically based on pieces on board to reduce computation time.
        - negamax_agent: agent to play the game using the negamax algorithm.
    
    
    Usage:
        - call negamax_agent in main function to simulate the games with negamax algorithm.
"""

def heuristic_value(board: np.ndarray, player: BoardPiece) -> int:
    """
    Apply a heuiristic value to each board state that is simulated to assist in determining the best move.
    
    Args:
        board (np.ndarray): the current board state.
        player (BoardPiece): the current player.
        
    returns:
        int: calculated heuristic value for simulated board state.
    """
    # Remark: I don't think the comments in this function are necessary
    opponent = get_opponent(player)
    score = 0

    for row in range(BOARD_ROWS): # Loop through bord and apply score to each position
        for col in range(BOARD_COLS):
            if board[row, col] == player:
                score += evaluate_position(board, row, col, player)
            elif board[row, col] == opponent:
                score -= evaluate_position(board, row, col, opponent)

    center_column = 3
    center_control = sum(
        3 if board[row, center_column] == player else -3 if board[row, center_column] == opponent else 0
        for row in range(BOARD_ROWS)
    ) # Apply score to center column
    score += center_control

    return score

def get_opponent(player):
    """
    Return the opponent of the current player.
    
    Args:
        player (BoardPiece): current player.
    
    returns:
        BoardPiece: opponent of the current player.
    """
    return BoardPiece(3 - player) # 3 - 1 for player 2 and 3 - 2 for player 1

def evaluate_position(board: np.ndarray, row: int, col: int, player: BoardPiece) -> int:
    """
    Apply a score to a single position on the board.
    
    Args:
        board (np.ndarray): the current board state.
        row (int): row of the position.
        col (int): column of the position.
        player (BoardPiece): the current player.
        
    returns:
        int: calculated score for the position.
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Right, Down, Down-Right, Down-Left
    score = 0

    for d_row, d_col in directions:
        count = count_in_direction(board, row, col, d_row, d_col, player)
        score += calculate_line_score(count)
    
    return score

def count_in_direction(
    board: np.ndarray, row: int, col: int, d_row: int, d_col: int, player: BoardPiece
) -> int:
    """
    Count consecutive pieces in a specific direction to determine winning and losing positions.
    
    Args:
        board (np.ndarray): the current board state.
        row (int): row of the position.
        col (int): column of the position.
        d_row (int): row direction.
        d_col (int): column direction.
        player (BoardPiece): the current player.
        
    returns:
        int: number of consecutive pieces in the direction to check for connect 4 situation.
    """
    count = 0
    for step in range(4):  # Check up to 4 in a line # Remark: You don't really need to check for 4, since negamax checks for wins explicitly
        new_row, new_col = row + step * d_row, col + step * d_col
        if 0 <= new_row < BOARD_ROWS and 0 <= new_col < BOARD_COLS and board[new_row, new_col] == player:
            count += 1
        else:
            break
    return count

def calculate_line_score(count: int) -> int:
    """
    Appply a score to a connect line based on pieces connected to determine best move.
    
    args:
        count (int): number of connected pieces.
        
    returns:
        int: calculated score for the connected pieces.
    """
    # Remark: I'd prefer to have the score values as constants at the top of the file,
    # since that would allow for easy tweaking of the values and reuse in other functions,
    # but in your solution they are at least all in one place
    if count == 2:
        return 10
    elif count == 3:
        return 50
    elif count == 4:
        return 1000
    return 0

def check_terminal_state(board: np.ndarray, player: BoardPiece, depth: int) -> Optional[Tuple[int, Optional[PlayerAction]]]:
    """
    Checks if next move will win, lose or draw and apply a heuristic value to the board state.
    
    Args:
        board (np.ndarray): Current board state.
        player (BoardPiece): Current player.
        depth (int): Tree search depth.

    Returns:
        Optional[Tuple[int, Optional[PlayerAction]]]: Tuple that contains heuristic value and best actions for simulated board state and player.
    """
    # Remark: I like this function
    game_state = check_end_state(board, player)
    if game_state == GameState.IS_WIN:
        return 1000, None  
    elif game_state == GameState.IS_DRAW:
        return 0, None
    elif depth == 0:
        return heuristic_value(board, player), None
    return None


def negamax(
    board: np.ndarray, depth: int, alpha: int, beta: int, player: BoardPiece
) -> Tuple[int, Optional[PlayerAction]]:
    """
    NegaMax algorithm with alpha-beta pruning that employs all above functions.
    
    Args:
        board (np.ndarray): Current board state.
        depth (int): Tree search depth.
        alpha (int): Alpha value for alpha-beta pruning (-inf for Player A).
        beta (int): Beta value for alpha-beta pruning (inf for Player B).
        player (BoardPiece): Current player.
        
    returns:
        Tuple[int, Optional[PlayerAction]]: Tuple that contains heuristic value and best actions for simulated board state and player.
    """
    # Remark: Nicely done!
    terminal_state = check_terminal_state(board, player, depth)
    if terminal_state is not None:
        return terminal_state
    
    max_value = -float('inf')
    best_action = ""

    for col in range(BOARD_COLS): # Loop through all columns to simulate multile game states
        if check_move_status(board, col) != MoveStatus.IS_VALID:
            continue

        simulation = board.copy()
        apply_player_action(simulation, col, player)

        value, _ = negamax(simulation, depth - 1, -beta, -alpha, get_opponent(player))
        value = -value

        if value > max_value:
            max_value = value
            best_action = col

        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return max_value, best_action

def dynamic_depth(board: np.ndarray) -> int:
    """
    Adjust search depth automatically based on already placed number of pieces on the board.
    # Remark: You should add that the depth is decreasing over the course of a game.
    
    Args:
        board (np.ndarray): Current board state.
    
    returns:
        int: Tree search depth.
    """
    # Remark: This is a cool idea! However, I think it would make more sense to
    # increase the depth with more pieces on the board, since the tree will become
    # more shallow and you'll get deeper in the same amount of time
    num_pieces = np.count_nonzero(board)
    if num_pieces < 10:
        return 6
    elif num_pieces <= 20:
        return 5
    else:
        return 4

def negamax_agent(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth: int = None
    # Remark: if depth is supposed to be an int, don't use None as a default value
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Agent to play the game using the negamax algorithm.
    
    Args:
        board (np.ndarray): Current board state.
        player (BoardPiece): Current player.
        saved_state (Optional[SavedState]): Saved state of the game to simulate multiple tree searches.
        depth (int): Tree search depth.
        
    returns:
        Tuple[PlayerAction, Optional[SavedState]]: Tuple that contains best action and saved state of the game.
    """
    depth = depth or dynamic_depth(board) 
    # Remark: None is not a boolean, so you should do this instead: depth = depth if depth is not None else dynamic_depth(board)
    # I know it works here, but using types that are not booleans as booleans is error prone.
    
    # Set negative and positive infinity values for alpha-beta pruning
    negative_inf = -float('inf')
    positive_inf = float('inf')
    
    _, action = negamax(board, depth, negative_inf, positive_inf, player)
    return PlayerAction(action), saved_state