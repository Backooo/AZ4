from agents.agent_mcts.mcts import *
from game_utils import *
from agents.agent_mcts.node import *
import numpy as np

def test_generate_move_mcts(board):
    child_found = False
    [column,saved_state] = generate_move_mcts(board,PLAYER1,SavedState)
    for child in Node.current_root.children:
        if [child.chosen_column, saved_state] == [column, saved_state]:
            child_found = True
    assert child_found

def test_node_update_root_by_board(board):
    Node.current_root.create_children(1)
    new_root = Node.current_root.children[2]
    apply_player_action(board, PlayerAction(2),PLAYER1)
    Node.update_root_by_board(board)
    assert Node.current_root == new_root

def test_choose_child_as_move_by_weight(board):
    win_board = np.full(BOARD_SHAPE, PLAYER1, BoardPiece)
    win_board[5,2] = NO_PLAYER
    Node(Node.current_root,2,win_board)
    Node(Node.current_root,2,board)
    best_child = Node.current_root.children[0]
    chosen_child_as_move = Node.current_root.choose_child_as_move_by_weight()
    assert chosen_child_as_move == best_child

def test_set_weight_problem(board):#überarbeiten
    board[0,1] = PLAYER2
    board[0,2] = PLAYER2
    board[0,3] = PLAYER2
    Node(Node.current_root,5,board.copy())
    Node(Node.current_root,0,board.copy())
    Node(Node.current_root.children[1],0,Node.current_root.children[1].board.copy())
    Node(Node.current_root.children[1],0,Node.current_root.children[1].board.copy())
    loser_board = Node.current_root.children[0].board.copy()
    Node(Node.current_root.children[0],4,loser_board)
    Node(Node.current_root.children[0],3,Node.current_root.children[0].board.copy())#loser board entfernt schau coverage
    best_child: Node = random.choice(Node.current_root.children)
    for child in Node.current_root.children[0].children:
        child.move_made_by = Node.player
        child.set_weight()
    for child in Node.current_root.children:
        if best_child.weight < child.weight:
            best_child = child
    chosen_child_as_move = Node.current_root.choose_child_as_move_by_weight()
    assert chosen_child_as_move == best_child
    for child in Node.current_root.children:
        child.weight = -1000000
        child.set_weight()
    assert Node.current_root.weight == -1000001
    

def test_choose_next_node_by_value(board):#überabreiten der boards mit copy
    board[0,0] = PLAYER1
    board[0,1] = PLAYER1
    board[0,2] = PLAYER1
    Node(Node.current_root,4,board.copy())
    Node(Node.current_root,5,board.copy())
    for child in Node.current_root.children:
        Node(child,3,child.board.copy())
    best_node = Node.current_root.children[1].children[0]
    chosen_node = Node.current_root.choose_next_node_by_value()
    assert chosen_node == best_node
