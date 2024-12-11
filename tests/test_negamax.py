from agents.agent_negamax.negamax import *
import numpy as np

# Remark: Great tests here!
def test_get_opponent_player1():
    """
    Test get_opponent function for player 1.
    """
    assert get_opponent(PLAYER1) == PLAYER2

def test_get_opponent_player2():
    """
    Test get_opponent function for player
    """
    assert get_opponent(PLAYER2) == PLAYER1

# Remark: Thanks for including the tests for the calculate_line_score function.
# I think it would be okay to leave them out since the function is so simple, or
# to break the rule of having only one assert per test function and include all
# cases in one function using a loop over the inputs and outputs for brevity.
def test_calculate_line_score_2():
    """
    Test calculate_line_score function for 2 pieces in a line.
    """
    assert calculate_line_score(2) == 10

def test_calculate_line_score_3():
    """
    Test calculate_line_score function for 3 pieces in a line.
    """
    assert calculate_line_score(3) == 50

def test_calculate_line_score_4():
    """
    Test calculate_line_score function for 4 pieces in a line.
    """
    assert calculate_line_score(4) == 1000

def test_calculate_line_score_1():
    """
    Test calculate_line_score function for less than 2 pieces in a line.
    """
    assert calculate_line_score(1) == 0

def test_count_in_direction_horizontal():
    """
    Test count_in_direction function for horizontal pieces
    """
    board = np.zeros((6, 7), dtype=np.int8)
    board[0, 0] = PLAYER1
    board[0, 1] = PLAYER1
    assert count_in_direction(board, 0, 0, 0, 1, PLAYER1) == 2

def test_count_in_direction_vertical():
    """
    Test count_in_direction function for vertical pieces
    """
    board = np.zeros((6, 7), dtype=np.int8)
    board[0, 0] = PLAYER1
    board[1, 0] = PLAYER1
    assert count_in_direction(board, 0, 0, 1, 0, PLAYER1) == 2

def test_count_in_direction_diagonal():
    """
    Test count_in_direction function for diagonal pieces
    """
    board = np.zeros((6, 7), dtype=np.int8)
    board[0, 0] = PLAYER1
    board[1, 1] = PLAYER1
    assert count_in_direction(board, 0, 0, 1, 1, PLAYER1) == 2

def test_count_in_direction_no_pieces():
    """
    Test count_in_direction function for no pieces
    """
    board = np.zeros((6, 7), dtype=np.int8)
    assert count_in_direction(board, 0, 0, 0, 1, PLAYER1) == 0

def test_evaluate_position_center():
    """
    Test evaluate_position function for center position
    """
    board = np.zeros((6, 7), dtype=np.int8)
    board[2, 3] = PLAYER1
    assert evaluate_position(board, 2, 3, PLAYER1) == 0

def test_evaluate_position_no_piece():
    """
    Test evaluate_position function for empty position
    """
    board = np.zeros((6, 7), dtype=np.int8)
    assert evaluate_position(board, 0, 0, PLAYER1) == 0

def test_heuristic_value_empty_board():
    """
    Test heuristic_value function for empty board
    """
    board = np.zeros((6, 7), dtype=np.int8)
    assert heuristic_value(board, PLAYER1) == 0

def test_heuristic_value_center_control():
    """
    Test heuristic_value function if center column is controlled by player
    """
    board = np.zeros((6, 7), dtype=np.int8)
    board[0, 3] = PLAYER1
    assert heuristic_value(board, PLAYER1) > 0

def test_heuristic_value_opponent_control():
    """
    Test heuristic_value function if center column is controlled by opponent
    """
    board = np.zeros((6, 7), dtype=np.int8)
    board[0, 3] = PLAYER2
    assert heuristic_value(board, PLAYER1) < 0


def test_check_terminal_state_win():
    """
    Test check_terminal_state function for win state
    """
    board = np.zeros((6, 7), dtype=np.int8)
    for col in range(4):
        apply_player_action(board, col, PLAYER1)
    assert check_terminal_state(board, PLAYER1, depth=3) == (1000, None)

def test_check_terminal_state_draw():
    """
    Test check_terminal_state function for draw state
    """
    board = np.ones((6, 7), dtype=np.int8) * PLAYER1
    board[0, :] = PLAYER2
    assert check_terminal_state(board, PLAYER1, depth=3) == (0, None)

def test_check_terminal_state_ongoing():
    """
    Test check_terminal_state function for ongoing state
    """
    board = np.zeros((6, 7), dtype=np.int8)
    apply_player_action(board, 0, PLAYER1)
    assert check_terminal_state(board, PLAYER1, depth=3) is None

def test_negamax_win():
    """
    Test negamax function for win state
    """
    board = np.zeros((6, 7), dtype=np.int8)
    for col in range(3):
        apply_player_action(board, col, PLAYER1)
    _, action = negamax(board, depth=4, alpha=-float('inf'), beta=float('inf'), player=PLAYER1)
    assert action == 3

def test_negamax_block():
    """
    Test negamax function for block state
    """
    board = np.zeros((6, 7), dtype=np.int8)
    for col in range(3):
        apply_player_action(board, col, PLAYER2)
    _, action = negamax(board, depth=4, alpha=-float('inf'), beta=float('inf'), player=PLAYER1)
    assert action == 3

def test_dynamic_depth_early_game():
    """
    Test deep dynamic_depth function for board with less than 10 placed pieces
    """
    test_board = np.array(
        [[0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,1,1,1,1,0,0]
        ],
        dtype=BoardPiece
    )
    assert dynamic_depth(test_board) == 6

def test_dynamic_depth_mid_game():
    """
    Test medium dynamic_depth function for board with less than 21 placed pieces
    """
    test_board = np.array(
        [[0,0,1,2,0,0,0],
        [0,0,2,1,0,0,0],
        [0,0,2,2,0,0,0],
        [0,0,1,2,0,0,0],
        [0,0,1,2,0,0,0],
        [0,2,1,1,1,0,0]
        ],
        dtype=BoardPiece
    )
    assert dynamic_depth(test_board) == 5

def test_dynamic_depth_late_game():
    """
    Test shallow dynamic_depth function otherwise
    """
    test_board = np.array(
        [[2,0,1,2,0,0,2],
        [1,0,2,1,0,0,1],
        [2,0,2,2,0,0,2],
        [1,0,1,2,0,0,1],
        [2,0,1,2,0,0,2],
        [1,2,1,1,1,0,1]
        ],
        dtype=BoardPiece
    )
    assert dynamic_depth(test_board) == 4

def test_negamax_agent_valid_move():
    """
    Test negamax_agent function for valid move
    """
    board = np.zeros((6, 7), dtype=np.int8)
    apply_player_action(board, 3, PLAYER1)
    action, _ = negamax_agent(board, PLAYER2, saved_state=None)
    assert action in range(7)

def test_negamax_agent_block_opponent():
    """
    Test negamax_agent function for blocking opponent
    """
    board = np.zeros((6, 7), dtype=np.int8)
    for col in range(3):
        apply_player_action(board, col, PLAYER2)
    action, _ = negamax_agent(board, PLAYER1, saved_state=None)
    assert action == 3