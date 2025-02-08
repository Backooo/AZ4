import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter

from agents.agent_MCTS import neuralNetwork
from agents.agent_MCTS.mcts import generate_move_mcts_with_policy
from agents.agent_MCTS.node import other_player
from game_utils import (
    PLAYER1,
    PLAYER2,
    GameState,
    apply_player_action,
    check_end_state,
    initialize_game_state,
)


class Connect4Dataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        state = state.unsqueeze(0)
        policy = torch.tensor(self.policies[idx], dtype=torch.float32)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return state, policy, value


def train_model(
    model: nn.Module,
    dataset: Connect4Dataset,
    epochs=10,
    batch_size=64,
    lr=1e-3,
    device="cuda",
):
    """Train the AlphaZero model on stored (state, policy, value) data"""
    
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        cooldown=0,
        min_lr=1e-6
    )
    
    scaler = torch.cuda.amp.GradScaler()
    compiled_model = torch.compile(model)
    
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for state, policy, value in dataloader:
            state, policy, value = state.to(device), policy.to(device), value.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predicted_policy, predicted_value = compiled_model(state)
                loss_policy = F.mse_loss(predicted_policy, policy)
                loss_value = F.mse_loss(predicted_value.squeeze(-1), value)
                loss = loss_policy + loss_value
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()

        avg_loss = total_loss / len(dataloader)
        
        current_lr = scheduler.get_last_lr()[0]
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "az4_trained.pt")

def self_play_games(model: nn.Module, num_games: int = 5, num_simulations: int = 700):
    """Generate self-play data using the AlphaZero algorithm
    
    Args:
        model (nn.Module): Neural network model
        num_games (int): Number of games to play
        num_simulations (int): Number of MCTS simulations to run per move
    """
    all_states = []
    all_policies = []
    all_values = []

    for game_index in range(num_games):
        board = initialize_game_state()
        saved_state = {PLAYER1: None, PLAYER2: None}
        current_player = PLAYER1

        game_history = []
        move_count = 0

        while True:
            move_count += 1
            move, policy, saved_state[current_player] = generate_move_mcts_with_policy(
                board=board,
                player=current_player,
                saved_state=saved_state[current_player],
                model=model,
                num_simulations=num_simulations,
            )

            board_copy = board.copy()
            game_history.append((board_copy, policy, current_player))

            apply_player_action(board, move, current_player)
            end_state = check_end_state(board, current_player)

            current_player = other_player(current_player)

            if end_state == GameState.IS_DRAW:
                outcome_p1 = 0.0
                outcome_p2 = 0.0
            else:
                if current_player == PLAYER1:
                    outcome_p1 = 1.0
                    outcome_p2 = -1.0
                else:
                    outcome_p1 = -1.0
                    outcome_p2 = 1.0

            for st, pol, pl in game_history:
                if pl == PLAYER1:
                    all_states.append(st)
                    all_policies.append(pol)
                    all_values.append(outcome_p1)
                else:
                    all_states.append(st)
                    all_policies.append(pol)
                    all_values.append(outcome_p2)

    return all_states, all_policies, all_values