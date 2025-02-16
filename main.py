from typing import Callable, Any
import time
import torch
import numpy as np
import concurrent.futures
from agents.agent_MCTS.mcts import generate_move_mcts, generate_move_mcts_with_policy
from agents.agent_MCTS.neuralNetwork import AlphaZeroNet
from agents.agent_MCTS.train import Connect4Dataset, self_play_games, train_model
from game_utils import (
    PLAYER1,
    PLAYER1_PRINT,
    PLAYER2,
    PLAYER2_PRINT,
    BoardPiece,
    GameState,
    GenMove,
    MoveStatus,
    PlayerAction,
    SavedState,
    apply_player_action,
    check_end_state,
    check_move_status,
    initialize_game_state,
    pretty_print_board,
)


def query_user(prompt_function: Callable) -> Any:
    usr_input = prompt_function("Column? ")
    return usr_input


def user_move(
    board: np.ndarray, _player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    move_status = None
    while move_status != MoveStatus.IS_VALID:
        input_move_string = query_user(input)
        input_move = convert_str_to_action(input_move_string)
        if input_move is None:
            continue
        move_status = check_move_status(board, input_move)
        if move_status != MoveStatus.IS_VALID:
            print(f"Move is invalid: {move_status.value}")
            print("Try again.")
    return input_move, saved_state


def convert_str_to_action(input_move_string: str) -> PlayerAction | None:
    try:
        input_move = PlayerAction(input_move_string)
    except ValueError:
        print("Invalid move: Input must be an integer.")
        print("Try again.")
    return input_move


def human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None):

    players = (PLAYER1, PLAYER2)
    
    player1_wins = 0
    player2_wins = 0
    draws = 0
    
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players,
                player_names,
                gen_moves,
                gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f"{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}"
                )
                action, saved_state[player] = gen_move(
                    board.copy(),  # copy board to be safe, even though agents shouldn't modify it
                    player,
                    saved_state[player],
                    *args,
                )
                print(f"Move time: {time.time() - t0:.3f}s")

                move_status = check_move_status(board, action)
                if move_status != MoveStatus.IS_VALID:
                    print(f"Move {action} is invalid: {move_status.value}")
                    print(f"{player_name} lost by making an illegal move.")
                    if player_name == player_names[0]:
                        player2_wins += 1
                    else:
                        player1_wins += 1
                    playing = False
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                        draws += 1
                    else:
                        print(
                            f"{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}"
                        )
                    playing = False
                    break

def parallel_self_play(
    model: torch.nn.Module, total_games: int, num_simulations: int, num_workers: int = 4
):
    """
    Run self_play_games in parallel using threads.
    Each worker runs a given number of games and returns (states, policies, values).
    """

    def worker(games):
        return self_play_games(model, num_games=games, num_simulations=num_simulations)

    games_per_worker = total_games // num_workers
    remainder = total_games % num_workers
    tasks = [games_per_worker] * num_workers
    for i in range(remainder):
        tasks[i] += 1

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    all_states, all_policies, all_values = [], [], []
    for states, policies, values in results:
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
    return all_states, all_policies, all_values

if __name__ == "__main__":
    
    model = AlphaZeroNet(board_shape=(6, 7), num_actions=7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model_current = AlphaZeroNet(board_shape=(6, 7), num_actions=7).to(device)
    model_current.load_state_dict(torch.load("az4_trained.pt", map_location=device))

    num_cycles = 1

    for cycle_i in range(num_cycles):
        print(f"\n=== Cycle {cycle_i} ===")
        states, policies, values = parallel_self_play(
            model_current, total_games=1, num_simulations=1, num_workers=1
        )

        dataset = Connect4Dataset(states, policies, values)
        train_model(
            model=model_current,
            dataset=dataset,
            epochs=1,
            batch_size=512,
            lr=3e-4,
            device=device,
        )

        #cycle_checkpoint = f"az4_trained_cycle_{cycle_i}.pt"
        #torch.save(model_current.state_dict(), cycle_checkpoint)
        #print(f"[CYCLE {cycle_i}] Saved checkpoint to {cycle_checkpoint}")


    model_new = AlphaZeroNet(board_shape=(6, 7), num_actions=7).to(device)
    model_new.load_state_dict(torch.load(f"az4_trained_cycle_{cycle_i}.pt", map_location=device))
    model_new.eval()
    
    model.load_state_dict(torch.load("az4_trained.pt", map_location=device))
    model.eval()
    
    human_vs_agent(
        generate_move_1=lambda board, player, saved_state: generate_move_mcts(
            board=board,
            player=player,
            saved_state=saved_state,
            model=model_new,
            num_simulations=1600,
        ),
        generate_move_2=lambda board, player, saved_state: generate_move_mcts(
            board=board,
            player=player,
            saved_state=saved_state,
            model=model,
            num_simulations=1600,
        ),
        player_1="New Model",
        player_2="Reference Model",
    )