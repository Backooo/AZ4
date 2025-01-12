import numpy as np

from game_utils import BoardPiece, PlayerAction, SavedState, MoveStatus, check_move_status
from typing import Optional, Callable, Any


def query_user(prompt_function: Callable) -> Any:
    usr_input = prompt_function("Column? ")
    return usr_input


def user_move(board: np.ndarray,
              _player: BoardPiece,
              saved_state: SavedState | None) -> tuple[PlayerAction, SavedState | None]:
    move_status = None
    while move_status != MoveStatus.IS_VALID:
        input_move_string = query_user(input)
        input_move = convert_str_to_action(input_move_string)
        if input_move is None:
            continue
        move_status = check_move_status(board, input_move)
        if move_status != MoveStatus.IS_VALID:
            print(f'Move is invalid: {move_status.value}')
            print('Try again.')
    return input_move, saved_state

def convert_str_to_action(input_move_string: str) -> PlayerAction | None: #hier irgendwelche probleme er returned auch wenn er in except drin war (move status =valid???)
    input_move = None # Meine Anpassug, da sonst bei Buchstaben (Not Integer) Errors entstehen
    try:
        input_move = PlayerAction(input_move_string)
    except ValueError:
        print('Invalid move: Input must be an integer.')
        print('Try again.')
    return input_move