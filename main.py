from typing import Callable
import time

import torch
from agents.agent_MCTS.mcts import generate_move_mcts
from agents.agent_MCTS.neuralNetwork import AlphaZeroNet
from agents.agent_MCTS.train import Connect4Dataset, self_play_games, train_model
from agents.agents_random.random import generate_move_random
from game_utils import PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT, GameState, MoveStatus, GenMove
from game_utils import initialize_game_state, pretty_print_board, apply_player_action, check_end_state, check_move_status
from agents.agent_human_user import user_move


def human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):

    players = (PLAYER1, PLAYER2)
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
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                )
                action, saved_state[player] = gen_move(
                    board.copy(),  # copy board to be safe, even though agents shouldn't modify it
                    player, saved_state[player], *args
                )
                print(f'Move time: {time.time() - t0:.3f}s')

                move_status = check_move_status(board, action)
                if move_status != MoveStatus.IS_VALID:
                    print(f'Move {action} is invalid: {move_status.value}')
                    print(f'{player_name} lost by making an illegal move.')
                    playing = False
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print('Game ended in draw')
                    else:
                        print(
                            f'{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                        )
                    playing = False
                    break


if __name__ == "__main__":
    # Create or load model
    model = AlphaZeroNet(board_shape=(6, 7), num_actions=7)  
    model.to('cpu')
    
    model.train()  # Set in train mode
    
    t0 = time.time()
    print("Generating self-play data...")
    
    states, policies, values = self_play_games(
        model=model,
        num_games=2, # how many full self-play games to generate
        num_simulations=1000 # how many MCTS simulations per move
    )
    print(f"Time to generate self-play data: {time.time() - t0:.3f}s")
    t1 = time.time()
    print(f"Collected {len(states)} training positions.")

    dataset = Connect4Dataset(states, policies, values)
    train_model(
        model=model,
        dataset=dataset,
        epochs=10,
        batch_size=128,
        lr=1e-4,
        device="cpu"
    )
    print(f"Time to train model: {time.time() - t1:.3f}s")

    torch.save(model.state_dict(), "az4_trained.pt")
    
    
    print("\Letting two MCTS agents play after training\n")
    model.load_state_dict(torch.load("az4_trained.pt", weights_only=True))
    model.eval()
    human_vs_agent(
        generate_move_1=lambda board, player, saved_state, model=model: generate_move_mcts(
            board=board,
            player=player,
            saved_state=saved_state,
            model=model,
            num_simulations=2500
        ),
        generate_move_2=lambda board, player, saved_state: generate_move_mcts(
            board=board,
            player=player,
            saved_state=saved_state,
            model=model,
            num_simulations=2500
        ),
        player_1="MCTS Agent 1",
        player_2="MCTS Agent 2"
    )
