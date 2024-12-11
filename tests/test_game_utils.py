
import numpy as np
from game_utils import *

def test_initialize_game_state_as_wrong_shape():
    test_shape = (9, 10)
    assert not initialize_game_state().shape == test_shape
    
def test_init_game_state_as_correct_shape():
    test_shape = (6, 7)
    assert initialize_game_state().shape == test_shape

def test_initialize_game_state_as_wrong_dtype():
    ret = initialize_game_state()
    assert ret.dtype == BoardPiece

def test_initialize_game_state_as_correct_dtype():
    test_dtype = np.int16
    ret =initialize_game_state()
    assert ret.dtype != test_dtype
    
def test_initialize_game_stat_with_no_players():
    ret = initialize_game_state()
    assert np.all(ret == NO_PLAYER)

def test_initialize_game_state_with_players():
    ret = initialize_game_state()
    assert not np.all(ret == BoardPiece(1))
    
def test_wrong_pretty_print_board():
    test_board = ''
    assert not pretty_print_board(np.full((6,7), NO_PLAYER, dtype=BoardPiece)) == test_board
    
def test_empty_pretty_print_board():
    test_board = (
        "|==============|\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )
    assert pretty_print_board(np.full((6,7), NO_PLAYER, dtype=BoardPiece)) == test_board
    
def test_player1_pretty_print_board():
    test_board = (
        "|==============|\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )
    assert pretty_print_board(np.full((6,7), PLAYER1, dtype=BoardPiece)) == test_board
    
def test_player2_pretty_print_board():
    test_board = (
        "|==============|\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )
    assert pretty_print_board(np.full((6,7), PLAYER2, dtype=BoardPiece)) == test_board
    
def test_empty_board_string_to_board():
    test_board = (
        "|==============|\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )
    assert np.all(string_to_board(test_board) == np.full((6,7), NO_PLAYER, dtype=BoardPiece))
    
def test_player1_board_string_to_board():
    test_board = (
        "|==============|\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|X X X X X X X |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )
    assert np.array_equal(string_to_board(test_board), np.full((6, 7), PLAYER1, dtype=BoardPiece))
    
def test_player2_board_string_to_board():
    test_board = (
        "|==============|\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|O O O O O O O |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )
    assert np.array_equal(string_to_board(test_board), np.full((6, 7), PLAYER2, dtype=BoardPiece))
    
def test_row1_player_action():
    test_board = np.full((6,7), NO_PLAYER, dtype=BoardPiece)
    apply_player_action(test_board, 1, PLAYER1)
    assert test_board[0, 1] == PLAYER1
    
def test_row6_player_action():
    test_board = np.array(
        [[2,2,2,2,2,2,2],
         [2,2,2,2,2,2,2],
         [2,2,2,2,2,2,2],
         [2,2,2,2,2,2,2],
         [2,2,2,2,2,2,2],
         [0,0,0,0,0,0,0]
         ],
    dtype=BoardPiece
    )
    apply_player_action(test_board, 1, PLAYER1)
    assert test_board[5, 1] == PLAYER1
    
def test_not_connected_four():
    test_board = np.full((6,7), NO_PLAYER, dtype=BoardPiece)
    assert connected_four(test_board, PLAYER1) == False
    
def test_connected_four_diagonal():
    test_board = np.array(
        [[0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0]
        ],
        dtype=BoardPiece
        )
    assert connected_four(test_board, PLAYER1) == True

def test_connected_four_horizontal():
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
    assert connected_four(test_board, PLAYER1) == True
    
def test_connected_four_vertical():
    test_board = np.array(
        [[0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0]
        ],
        dtype=BoardPiece
    )
    assert connected_four(test_board, PLAYER1) == True
    
def test_check_end_state_win():
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
    assert check_end_state(test_board, PLAYER1) == GameState.IS_WIN
    
    
def test_check_end_state_draw():
    test_board = np.array(
        [
        [1,1,1,2,2,2,1],
        [2,2,2,1,1,1,2],
        [1,1,1,2,2,2,1],
        [2,2,2,1,1,1,2],
        [1,1,1,2,2,2,1],
        [2,2,2,1,1,1,2]
        ],
        dtype=BoardPiece
    )
    assert check_end_state(test_board, PLAYER1) == GameState.IS_DRAW 
    
def test_check_end_state_still_playing():
    test_board = np.array(
        [
        [1,1,1,2,2,2,1],
        [2,2,2,1,1,1,2],
        [1,0,1,2,2,2,0],
        [2,0,2,0,1,1,0],
        [1,0,1,0,0,2,0],
        [0,0,0,0,0,0,0]
        ],
        dtype=BoardPiece
    )
    assert check_end_state(test_board, PLAYER1) == GameState.STILL_PLAYING 
    
def test_check_move_status_wrong_type():
    test_board = np.array(
        [
        [1,1,1,2,2,2,1],
        [2,2,2,1,1,1,2],
        [1,0,1,2,2,2,0],
        [2,0,2,0,1,1,0],
        [1,0,1,0,0,2,0],
        [0,0,0,0,0,0,0]
        ],
        dtype=BoardPiece
    )
    assert check_move_status(test_board, "a") == MoveStatus.WRONG_TYPE
    
def test_check_move_status_out_of_bounds():
    test_board = np.array(
        [
        [1,1,1,2,2,2,1],
        [2,2,2,1,1,1,2],
        [1,0,1,2,2,2,0],
        [2,0,2,0,1,1,0],
        [1,0,1,0,0,2,0],
        [0,0,0,0,0,0,0]
        ],
        dtype=BoardPiece
    )
    assert check_move_status(test_board, 999) == MoveStatus.OUT_OF_BOUNDS
    
def test_check_move_status_full():
    test_board = np.array(
        [
        [1,1,1,2,2,2,1],
        [2,2,2,1,1,1,2],
        [1,0,1,2,2,2,0],
        [2,0,2,0,1,1,0],
        [1,0,1,0,0,2,0],
        [0,0,0,0,0,0,0]
        ],
        dtype=BoardPiece
    )
    assert check_move_status(test_board, 1) == MoveStatus.IS_VALID
    
def test_check_move_status_valid():
    test_board = np.array(
        [
        [1,1,1,2,2,2,1],
        [2,2,2,1,1,1,2],
        [1,0,1,2,2,2,0],
        [2,0,2,0,1,1,0],
        [1,0,1,0,0,2,0],
        [0,0,0,0,0,0,0]
        ],
        dtype=BoardPiece
    )
    assert check_move_status(test_board, 2) == MoveStatus.IS_VALID