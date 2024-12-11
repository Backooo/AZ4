
import numpy as np
from game_utils import *

# Remark: Overall your tests are okay. You have a few tests that are not really necessary (see below),
# and some of the test cases are somewhat special (like the full boards for pretty_print_board).
# It's enough here, but keep in mind that test cases should be relatively generic.
# One useful mental test is to ask the question how easy it would be to pass the test with a wrong implementation.

# Remark: As a tip for the future, look into fixtures. They allow you to reuse setup code for multiple tests.

# Remark: If you have strong names for test functions, there's no need for the docstring.

def test_initialize_game_state_as_wrong_shape():
    """
    Test if the shape of the initialized game state is wrong.
    """
    # Remark: But you're only testing one false shape. If you're worried the shape
    # comes out wrong, you'd have to test all wrong shapes? That of course doesn't work,
    # which is why in tests you always check whether the outcome is the expected 
    # correct one (as you're doing below).
    test_shape = (9, 10)
    assert not initialize_game_state().shape == test_shape
    
def test_init_game_state_as_correct_shape():
    """
    Test if the shape of the initialized game state is correct.
    """
    test_shape = (6, 7)
    assert initialize_game_state().shape == test_shape

def test_initialize_game_state_as_wrong_dtype():
    """
    Test if the initialized game state is of the wrong dtype.
    """
    ret = initialize_game_state()
    assert ret.dtype == BoardPiece

def test_initialize_game_state_as_correct_dtype():
    """
    Test if the initialized game state is of the correct dtype.
    """
    test_dtype = np.int16
    ret =initialize_game_state()
    assert ret.dtype != test_dtype
    
def test_initialize_game_stat_with_no_players():
    """
    Test if the initialized game state has no players.
    """
    ret = initialize_game_state()
    assert np.all(ret == NO_PLAYER)

def test_initialize_game_state_with_players():
    """
    Test if the initialized game state has players.
    """
    ret = initialize_game_state()
    assert not np.all(ret == BoardPiece(1))
    
def test_wrong_pretty_print_board():
    """
    Test if the pretty print board is wrong.
    """
    test_board = ''
    assert not pretty_print_board(np.full((6,7), NO_PLAYER, dtype=BoardPiece)) == test_board
    
def test_empty_pretty_print_board():
    """
    Test if the pretty print board is empty and correct.
    """
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
    """
    Test if the pretty print board is correct for board filled only with player 1.
    """
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
    """
    Test if the pretty print board is correct for board filled only with player 2.
    """
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
    """
    Test if string to board is correct for an empty board.
    """
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
    """
    Test if string to board is correct for a board filled only with player 1.
    """
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
    """
    Test if string to board is correct for a board filled only with player 2.
    """
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
    """
    Test if the player action is applied correctly in row 1.
    """
    test_board = np.full((6,7), NO_PLAYER, dtype=BoardPiece)
    apply_player_action(test_board, 1, PLAYER1)
    assert test_board[0, 1] == PLAYER1
    # Remark: You should also test that the rest of the board is unchanged.
    
def test_row6_player_action():
    """
    Test if the player action is applied correctly in row 6.
    """
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
    """
    Test if the function returns False for a board with no connected four.
    """
    test_board = np.full((6,7), NO_PLAYER, dtype=BoardPiece)
    assert connected_four(test_board, PLAYER1) == False
    
def test_connected_four_diagonal():
    """
    Test if the function returns True for a board with a diagonal connect4.
    """
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
    """
    Test if the function returns True for a board with a horizontal connect4.
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
    assert connected_four(test_board, PLAYER1) == True
    
def test_connected_four_vertical():
    """
    Test if the function returns True for a board with a vertical connect4.
    """
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
    """
    Test if the function returns IS_WIN for a won state for given player.
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
    assert check_end_state(test_board, PLAYER1) == GameState.IS_WIN
    
    
def test_check_end_state_draw():
    """
    Test if the function returns IS_DRAW for a won state for given player.
    """
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
    """
    Test if the function returns STILL_PLAYING for a won state for given player.
    """
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
    """
    Test if the function returns WRONG_TYPE for a wrong input.
    """
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
    """
    Test if the function returns OUT_OF_BOUNDS for a move out of bounds.
    """
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
    """
    Test if the function returns FULL for a full column.
    """
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
    """
    Test if the function returns IS_VALID for a valid move
    """
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