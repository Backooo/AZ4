from agents.agent_random.random import *
from game_utils import *

def test_generate_move_random(board):
    board[5,0] = PLAYER2
    board[5,1] = PLAYER2
    board[5,2] = PLAYER2
    board[5,3] = PLAYER2
    board[5,4] = PLAYER2
    board[5,5] = PLAYER2
    assert generate_move_random(board,PLAYER1,SavedState) == [PlayerAction(6) ,SavedState]