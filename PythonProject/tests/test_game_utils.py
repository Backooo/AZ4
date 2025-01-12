from game_utils import *
import numpy as np

def test_initialize_game_state(board):
    assert board.shape == (BOARD_ROWS, BOARD_COLS)
    assert np.all(board == NO_PLAYER)

def test_pretty_print_board(board):
    apply_player_action(board,PlayerAction(0),PLAYER1)
    apply_player_action(board, PlayerAction(0), PLAYER2)
    assert pretty_print_board(board) == "|=============|\n|             |\n|             |\n|             |\n|             |\n|O            |\n|X            |\n|=============|\n|0 1 2 3 4 5 6|"

def test_string_to_board(board):
    apply_player_action(board,PlayerAction(0),PLAYER1)
    apply_player_action(board, PlayerAction(0), PLAYER2)
    assert np.array_equal(board,string_to_board("|=============|\n|             |\n|             |\n|             |\n|             |\n|O            |\n|X            |\n|=============|\n|0 1 2 3 4 5 6|"))

def test_apply_player_action(board):
    apply_player_action(board, PlayerAction(0), PLAYER1)
    assert board[0, 0] == PLAYER1
    apply_player_action(board, PlayerAction(0), PLAYER2)
    assert board[1, 0] == PLAYER2

def test_connected_four_row(board):
    board[5, 0] = PLAYER2
    board[5, 1] = PLAYER1
    board[5, 2] = PLAYER1
    board[5, 3] = PLAYER1
    board[5, 4] = PLAYER1
    assert connected_four(board, PLAYER1)

def test_connected_four_column(board):
    board[5, 0] = PLAYER1
    board[4, 0] = PLAYER1
    board[3, 0] = PLAYER1
    board[2, 0] = PLAYER1
    assert connected_four(board, PLAYER1)

def test_connected_four_diagonal(board):
    board[5, 0] = PLAYER1
    board[4, 1] = PLAYER1
    board[3, 2] = PLAYER1
    board[2, 3] = PLAYER1
    assert connected_four(board, PLAYER1)

def test_no_connected_four(board):
    board[5, 0] = PLAYER1
    board[4, 1] = PLAYER1
    board[3, 2] = PLAYER1
    assert not connected_four(board, PLAYER1)

def test_check_end_state_win(board):
    board[5, 0] = PLAYER1
    board[5, 1] = PLAYER1
    board[5, 2] = PLAYER1
    board[5, 3] = PLAYER1
    assert check_end_state(board, PLAYER1) == GameState.IS_WIN

def test_check_end_state_still_playing(board):
    board[5, 0] = PLAYER1
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

def test_check_end_state_draw(board):
    board[5, 0] = PLAYER1
    board[5, 1] = PLAYER2
    board[5, 2] = PLAYER1
    board[5, 3] = PLAYER1
    board[5, 4] = PLAYER1
    board[5, 5] = PLAYER2
    board[5, 6] = PLAYER1
    assert check_end_state(board, PLAYER1) == GameState.IS_DRAW

def test_check_move_status_valid_move(board):
    move_status = check_move_status(board, PlayerAction(0))
    assert move_status == MoveStatus.IS_VALID

def test_check_move_status_wrong_type(board):
    move_status = check_move_status(board, "not an int")
    assert move_status == MoveStatus.WRONG_TYPE

def test_check_move_status_out_of_bounds(board):
    move_status = check_move_status(board, PlayerAction(7))
    assert move_status == MoveStatus.OUT_OF_BOUNDS

def test_check_move_status_full_column(board):
    for row in range(BOARD_ROWS):
        board[row, 0] = PLAYER1
    move_status = check_move_status(board, PlayerAction(0))
    assert move_status == MoveStatus.FULL_COLUMN
