import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import LR, GAMMA, TARGET_UPDATE_FREQ

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Linear_Qnet(nn.Module):
    """
    Fully-connected Q-network: INPUT_SIZE → HIDDEN_SIZE (ReLU) → OUTPUT_SIZE.

    Kept deliberately shallow because the state space is only 11 binary features.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_folder_path, file_name))

    def load(self, file_name='model.pth'):
        path = os.path.join('./model', file_name)
        if os.path.isfile(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            self.eval()
            print('Loading existing state dict.')
            return True
        print('No existing state dict found. Starting from scratch.')
        return False


class QTrainer:
    """
    DQN trainer with four key improvements over vanilla Q-learning:

    1. Target network — a frozen copy of the online network used to compute
       Bellman targets. Updated every TARGET_UPDATE_FREQ gradient steps.
       Prevents the moving-target problem that causes training oscillation.

    2. Huber loss (SmoothL1) — behaves like MSE for small errors and like
       MAE for large ones, making it robust to outlier reward values.

    3. Vectorised Q-target computation — the entire mini-batch is processed
       in a single tensor operation instead of a Python loop per sample.

    4. Gradient clipping — caps gradient L2-norm at 1.0 to prevent
       exploding gradients on steep regions of the Q-value surface.
    """

    def __init__(self, model, lr=LR, gamma=GAMMA,
                 target_update_freq=TARGET_UPDATE_FREQ):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        # Frozen copy of the online network used only to compute Q-targets
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss().to(device)   # Huber loss
        self.train_count = 0
        self.target_update_freq = target_update_freq

    def _sync_target(self):
        """Hard-copy online network weights → target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, game_is_over):
        state      = torch.tensor(state,      dtype=torch.float, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        action     = torch.tensor(action,     dtype=torch.float, device=device)
        reward     = torch.tensor(reward,     dtype=torch.float, device=device)

        # Single-sample path: add batch dim so the rest is uniform
        if state.dim() == 1:
            state        = state.unsqueeze(0)
            next_state   = next_state.unsqueeze(0)
            action       = action.unsqueeze(0)
            reward       = reward.unsqueeze(0)
            game_is_over = (game_is_over,)

        # --- Vectorised Bellman target (no Python loop) ----------------------
        # Frozen target network prevents the network from chasing moving targets
        with torch.no_grad():
            max_next_q = self.target_model(next_state).max(dim=1).values

        done = torch.tensor(game_is_over, dtype=torch.float, device=device)
        # Q*(s,a) = r  +  γ · max_a' Q_target(s', a')   (masked to 0 if terminal)
        q_targets = reward + self.gamma * max_next_q * (1.0 - done)

        # Compute predictions and update only the Q-value for the taken action
        pred   = self.model(state)
        target = pred.clone().detach()
        action_idx = torch.argmax(action, dim=1)
        target[torch.arange(target.size(0)), action_idx] = q_targets

        # --- Gradient update -------------------------------------------------
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Periodically hard-copy weights to target network
        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self._sync_target()
