import sys
sys.path.append(r"C:\Users\amirm\Desktop\PythonProject")
sys.path.append(r"C:\Users\Amir\Desktop\PythonProject")
from game_utils import initialize_game_state
from agents.agent_mcts.node import *
import pytest

@pytest.fixture
def board():
    return initialize_game_state()
@pytest.fixture(autouse=True)
def node(board):
    Node.set_player(PLAYER1)
    Node.update_root_by_board(board)