import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class Connect4Dataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx]), torch.tensor(self.policies[idx]), torch.tensor(self.values[idx])

def train_model(model, dataset, epochs=10, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for state, policy, value in dataloader:
            optimizer.zero_grad()
            predicted_policy, predicted_value = model(state)
            loss_policy = F.cross_entropy(predicted_policy, policy)
            loss_value = F.mse_loss(predicted_value.squeeze(), value)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
