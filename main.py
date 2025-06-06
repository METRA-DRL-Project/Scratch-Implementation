import gymnasium as gym
import ale_py
from ale_py import ALEInterface

import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import copy
import numpy as np
from datetime import datetime
import torchvision.models as models

import cv2
import matplotlib.pyplot as plt
import imageio
import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.manifold import TSNE
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import gc
from tensorboard.plugins.hparams import api as hp

import pytesseract
import re
import os
from gymnasium.wrappers import RecordVideo
from sklearn.decomposition import PCA

# code should work on either, faster on gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.autograd.set_detect_anomaly(True)

now = datetime.now()
timestamp = now.strftime("%d_%H%M%S")
writer = SummaryWriter(log_dir=f'runs/SAC-Discrete-{timestamp}')

ale = ALEInterface()
gym.register_envs(ale_py)

METRICS = [
    "eval/mean_intrinsic_reward",
    "eval/total_state_coverage",
    "eval/mean_episode_length",
    "representation/silhouette_score",
    "policy/entropy",
    "loss/q_loss",
    "loss/ - policy loss"
]

HPARAMS = {
    'alpha': hp.HParam('alpha', hp.RealInterval(0.0, 1.0)),
    'q_lr': hp.HParam('q_lr', hp.RealInterval(1e-5, 1e-3)),
    'policy_lr': hp.HParam('policy_lr', hp.RealInterval(1e-5, 1e-3)),
    'representation_lr': hp.HParam('representation_lr', hp.RealInterval(1e-6, 1e-3)),
    'lambda_lr': hp.HParam('lambda_lr', hp.RealInterval(1e-6, 1e-3)),
    'epsilon': hp.HParam('epsilon', hp.RealInterval(0.0, 10.0)),
    'lambda_param': hp.HParam('lambda_param', hp.RealInterval(0.01, 2.0)),
    'n_skill': hp.HParam('n_skill', hp.RealInterval(2.0, 10.0))
}

class Config():
    def __init__(self, lambda_param, lambda_lr, alpha, q_lr, policy_lr, representation_lr, \
         timestamp, epsilon, n_skill, warmup_epochs, n_update_policy, n_update_repr):
        self.lambda_param = lambda_param
        self.lambda_lr = lambda_lr
        self.alpha = alpha
        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.representation_lr = representation_lr
        self.epsilon = epsilon
        self.n_skill = n_skill
        self.warmup_epochs = warmup_epochs
        self.n_update_policy = n_update_policy
        self.n_update_repr = n_update_repr

        folder = f'./model/{timestamp}'
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, 'config.txt'), 'w') as f:
            f.write(f'alpha: {self.alpha}\n')
            f.write(f'lambda: {self.lambda_param}\n')
            f.write(f'lambda_lr: {self.lambda_lr}\n')
            f.write(f'q_lr: {self.q_lr}\n')
            f.write(f'policy_lr: {self.policy_lr}\n')
            f.write(f'representation_lr: {self.representation_lr}\n')
            f.write(f'epsilon: {self.epsilon}\n')
            f.write(f'n_skill: {self.n_skill}\n')
            f.write(f'warmup_epochs: {self.warmup_epochs}\n')
            f.write(f'n_update_policy" {self.n_update_policy}\n')
            f.write(f'n_update_repr: {self.n_update_repr}\n')


def cleanup_memory():
    gc.collect()  # Clean up CPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()            # Release cached GPU memory (PyTorch only)
        torch.cuda.ipc_collect()            # Clean up interprocess memory (multi-GPU safe)

import warnings


import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch

class PretrainedResNetEncoder(nn.Module):
    def __init__(self, out_dim=256, freeze_base=True):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # Modify first layer to accept 3-channel input (RGB Atari)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Truncate after last convolution
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # [B, 512, 1, 1]

        if freeze_base:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Flatten(),            # [B, 512]
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = F.interpolate(x / 255.0, size=(224, 224))  # match ImageNet size
        x = self.feature_extractor(x)
        return self.fc(x)


# @title Define Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = 64

    def store(self, state, skill, action, reward, next_state, done):
        transitions = list(zip(state, skill, action, reward, next_state, 1 - torch.Tensor(done)))
        self.buffer.extend(transitions)

    def sample(self):
        if len(self.buffer) < self.batch_size:
            batch = random.choices(self.buffer, k=self.batch_size)
            warnings.warn(f"Requested batch size {self.batch_size} is larger than buffer size \
                 {len(self.buffer)}. Sampling with replacement.", category=UserWarning)

        else:
            batch = random.sample(self.buffer, self.batch_size)

        return [torch.stack(e).to(device) for e in zip(*batch)]  # state, skill, action, reward, next_state, not_done

    def sample_by_skill(self, skill_id, num_samples=128): ## THIS METHOD ASSUMES ONE HOT ENCODED SKILLS -- but the buffer holds zero centered encoding... -- that will also work :)
        filtered = [transition for transition in self.buffer
                    if torch.argmax(transition[1]).item() == skill_id]

        if len(filtered) < num_samples:
            batch = filtered
            warnings.warn(f"Not enough samples for skill {skill_id}. Sampling with replacement.", category=UserWarning)
        else:
            batch = random.sample(filtered, num_samples)

        return [torch.stack(e).to(device) for e in zip(*batch)]


    def __len__(self):
        return len(self.buffer)

class DRL:
    def __init__(self, buffer_size = 10000):
        self.n_envs = 8 # n_envs different runs of the same environment -- so this basically handles the number of episodes
        self.n_steps = 512

        self.envs = gym.vector.SyncVectorEnv(
            [lambda: gym.make('ALE/MsPacman-v5') for _ in range(self.n_envs)])

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def rollout(self, agent, i, n_skill, encoder):
        """Collect experience and store it in the replay buffer"""

        obs, _ = self.envs.reset()
        # obs = torch.tensor(obs, dtype=torch.float32, device=device).view(obs.shape[0], -1) ## CURRENTLY JUST FLATTENING, BUT CAN POTENTIALLY USE CNN TO EXTRACT FEATURES
        obs = torch.tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        enc_obs = encoder(obs)

        # Sample a skill per environment (shape: [n_envs, skill_dim]) 
        # skills = torch.rand(self.n_envs, n_skill, device=device)

        ## Sample a zero centered one hot skill 
        one_hot = torch.eye(n_skill, device=device)  # shape (n_skills, n_skills)
        mean = one_hot.mean(dim=0, keepdim=True)
        zero_centered = one_hot - mean  # subtract mean from each row
        skills = random.choices(zero_centered, k=self.n_envs)
        skills = torch.stack(skills, dim=0)

        ## SAMPLE ONLY ONE SKILL PER ENVIRONMENT
        # skills = torch.eye(n_skill)[torch.randint(0, n_skill, (self.n_envs,))].to(device) # One hot encoded skills

        total_rewards = torch.zeros(self.n_envs, device=device)

        for _ in range(self.n_steps):
            with torch.no_grad():
                actions = agent.get_action(enc_obs, skills)

            next_obs, rewards, dones, truncateds, _ = self.envs.step(actions.cpu().numpy())
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            not_done = ~(dones | truncateds)

            self.replay_buffer.store(obs, skills, actions, rewards, next_obs, not_done) ### SO THE REPLAY BUFFER STORE 4 experiences 
            obs = next_obs
            total_rewards += rewards

        writer.add_scalar("stats/Rewards", total_rewards.mean().item() / self.n_steps, i)


    # def rollout(self, agent, i, n_skill, encoder):
    #     obs, _ = self.envs.reset()
    #     obs = torch.tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    #     enc_obs = encoder(obs)

    #     # Initialize skill vector for each environment
    #     # env_skills = torch.eye(n_skill)[torch.randint(0, n_skill, (self.n_envs,))].to(device)  # shape: (n_envs, n_skill)
    #     env_skills = F.normalize(torch.randn(self.n_envs, n_skill, device=device), dim=1)  # shape: (n_envs, n_skill)
    #     total_rewards = torch.zeros(self.n_envs, device=device)

    #     for step_num in range(self.n_steps):
    #         with torch.no_grad():
    #             actions = agent.get_action(enc_obs, env_skills)

    #         next_obs, rewards, dones, truncateds, _ = self.envs.step(actions.cpu().numpy())
    #         next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    #         rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    #         not_done = ~(dones | truncateds)

    #         # Store the transitions
    #         self.replay_buffer.store(obs, env_skills, actions, rewards, next_obs, not_done)
    #         obs = next_obs
    #         enc_obs = encoder(obs)
    #         total_rewards += rewards

    #         # Resample skills only for environments where episode ended
    #         done_mask = dones | truncateds
    #         if done_mask.any(): ## RESAMPLING SKILLS QUITE A LOT :)
    #             new_skills = F.normalize(torch.randn(done_mask.sum(), n_skill, device=device), dim=1)
    #             env_skills[done_mask] = new_skills

    #     writer.add_scalar("stats/Rewards", total_rewards.sum().item() / self.n_steps, i)

def encode_state(state, encoder):
    if state.dim() == 4 and state.shape[1] == 3:
        features = encoder(state)
    elif state.dim() == 3 and state.shape[0] == 3:
        state = state.unsqueeze(0)
        features = encoder(state)
    elif state.dim() == 2:
        features = state
    else:
        raise ValueError(f'found a state with shape: {state.shape} in get_latent_representationss')
    
    return features

class SkillQNet(nn.Module):
    def __init__(self, obs_dim, skill_dim, n_actions, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + skill_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + skill_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, n_actions)
        self.skill_scale = 3.0

    def forward(self, obs, skill):
        skill = torch.tanh(skill * self.skill_scale)
        x = torch.cat([obs, skill], dim=-1)
        x = F.silu(self.fc1(x))
        x = torch.cat([x, skill], dim=-1)  # skip connection
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        return self.out(x)  # [B, n_actions]

class SkillPolicy:
    def __init__(self, n_obs, n_skills, n_actions, representation, config):
        self.alpha_start = config.alpha
        self.alpha_end = config.alpha - (0.001 * 551) # Ensuring that I get 0.0001 step down every epoch -- which isn't too much -- hopefully ideal
        self.n_actions = n_actions
        self.encoder = representation.encoder
        self.representation = representation
        self.n_step_update = config.n_update_policy

        self.policy = SkillPolicyNet(obs_dim=n_obs, skill_dim=n_skills, n_actions=n_actions).to(device)

        self.q1_net = SkillQNet(n_obs, n_skills, n_actions).to(device)
        self.q2_net = SkillQNet(n_obs, n_skills, n_actions).to(device)
        self.q1_target_net = SkillQNet(n_obs, n_skills, n_actions).to(device)
        self.q2_target_net = SkillQNet(n_obs, n_skills, n_actions).to(device)

        self.q_optimizer = Adam(
            list(self.q1_net.parameters()) + list(self.q2_net.parameters()), lr=config.q_lr
        )
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.policy_lr)
        self.gamma = 0.99

    def get_policy_distribution(self, states, skills):
        states = encode_state(states, self.encoder)
        logits = self.policy(states, skills)
        return Categorical(logits=logits)

    def get_action(self, states, skills, eval=False):
        dist = self.get_policy_distribution(states, skills)
        if eval:
            return torch.argmax(dist.probs, dim=-1)
        return dist.sample()

    def get_q_loss(self, states, actions, rewards, next_states, not_dones, skills):
        with torch.no_grad():
            phi = self.representation.get_latent_representation(states)
            phi_next = self.representation.get_latent_representation(next_states)

            delta_phi = phi_next - phi

            intrinsic_reward = torch.einsum('bi,bi->b', delta_phi, skills).unsqueeze(-1)
            not_dones = not_dones.unsqueeze(-1)

            next_states = encode_state(next_states, self.encoder)
            next_q1 = self.q1_target_net(next_states, skills)
            next_q2 = self.q2_target_net(next_states, skills)
            next_q = torch.min(next_q1, next_q2)

            next_pi = self.get_policy_distribution(next_states, skills)
            log_probs = next_pi.logits.log_softmax(dim=-1)
            next_q_val = (next_pi.probs * (next_q - self.alpha * log_probs)).sum(dim=-1, keepdim=True)
            # print(f'intrinsic_reward.shape: {intrinsic_reward.shape}, next_q_val.shape: {next_q_val.shape}, not_dones.shape: {not_dones.shape}')
            q_target = 10 * intrinsic_reward + self.gamma * not_dones * next_q_val

        states = encode_state(states, self.encoder)

        q1 = self.q1_net(states, skills).gather(1, actions.long().unsqueeze(-1))
        q2 = self.q2_net(states, skills).gather(1, actions.long().unsqueeze(-1))

        # print(f'q1.shape: {q1.shape}, q_targte.shpae; {q_target.shape}, q2.shape: {q2.shape}')

        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        return loss

    def get_policy_loss(self, states, skills):
        dist = self.get_policy_distribution(states, skills)
        probs = dist.probs
        log_probs = dist.logits.log_softmax(dim=-1)

        states = encode_state(states, self.encoder)

        q1 = self.q1_net(states, skills)
        q2 = self.q2_net(states, skills)
        q = torch.min(q1, q2)

        policy_loss = -(probs * (q - self.alpha * log_probs)).sum(dim=1).mean()
        return policy_loss

    def get_entropy(self, states, skills):
        dist = self.get_policy_distribution(states, skills)
        return dist.entropy()

    def update(self, replay_buffer, i):
      frac = min(i, 551) / float(551)
      self.alpha = self.alpha_start + frac * (self.alpha_end - self.alpha_start)
      writer.add_scalar("anneal/alpha", self.alpha, i)

      for _ in range(self.n_step_update):
        states, skills, actions, rewards, next_states, not_dones = replay_buffer.sample()
        # Compute Q-loss
        q_loss = self.get_q_loss(states, actions, rewards, next_states, not_dones, skills)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update the policy
        policy_loss = self.get_policy_loss(states, skills)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

      # Soft update the target networks using Polyak averaging
      self.soft_update(self.q1_net, self.q1_target_net)
      self.soft_update(self.q2_net, self.q2_target_net)
      entropy = self.get_entropy(states, skills).mean().item()
      writer.add_scalar("policy/entropy", entropy, i)

      writer.add_scalar("loss/q_loss", q_loss.item(), i)
      writer.add_scalar("loss/ - policy loss", -policy_loss.item(), i)

    def soft_update(self, source, target, tau=0.005):
        """Soft update the target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# def get_ghost_locations(obs):
#     bad_ghosts = [198, 200, 180, 84]
#     ghost_locs = []
#     for ghost in bad_ghosts:
#         rows, cols = np.where(obs[:, :, 0] == 198)
#         if len(rows) > 0 and len(cols) > 0:
#             top = np.min(rows)
#             bottom = np.max(rows)
#             left = np.min(cols)
#             right = np.max(cols)
            
#         if len(rows) > 0:
#             ghost_y = np.mean(rows)
#             ghost_x = np.mean(cols)
#         else:
#             ghost_x, ghost_y = 0.0, 0.0
        
#         ghost_locs.append(ghost_x)
#         ghost_locs.append(ghost_y)
    
#     while len(ghost_locs) < 8:
#         ghost_locs.append(0.0)
    
#     hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, np.array([11, 168, 194]), np.array([11, 168, 194]))  # Binary image
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         if cv2.contourArea(cnt) < 10:  # filter noise
#             continue

#         x, y, w, h = cv2.boundingRect(cnt)
#         cx = x + w // 2
#         cy = y + h // 2
#         ghost_locs.append(cx)
#         ghost_locs.append(cy)

#     while len(ghost_locs) < 16:
#         ghost_locs.append(0.0)

#     return ghost_locs

# def get_pacman_location(obs):
#     lower_yellow = np.array([20, 100, 100])
#     upper_yellow = np.array([30, 255, 255])
#     hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
#     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  # Binary image
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     num_lives = -1
#     ghost_loc = None
#     for cnt in contours:
#         if cv2.contourArea(cnt) < 10:  # filter noise
#             continue

#         x, y, w, h = cv2.boundingRect(cnt)
#         cx = x + w // 2
#         cy = y + h // 2
#         num_lives += 1

#         if cy != 178:
#             ghost_loc = [cx, cy]

#     if ghost_loc is None:
#         warnings.warn(f"Couldn't find pacman location while encoding", category=UserWarning)
#         return ([0, 0], 0)
    
#     return (ghost_loc, num_lives)

# def get_score_from_frame(frame):
#     obs = frame[182: , :, :]
#     gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
#     gray = (gray * 255).astype('uint8')
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#     # # Optional: resize for better OCR accuracy
#     thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

#     # # Run OCR using pytesseract
#     custom_config = r'--oem 3 --psm 6 outputbase digits'
#     text = pytesseract.image_to_string(thresh, config=custom_config)

#     # # Extract only numbers
#     numbers = re.findall(r'\d+', text)
#     if len(numbers) == 0:
#         return 0
#     else:
#         return int(numbers[0])

# class CNNEncoder(nn.Module):
#     def __init__(self, out_dim):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=8, stride=4),  # (3, 210, 160) → (32, 52, 39)
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2), # → (64, 25, 18)
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1), # → (64, 23, 16)
#             nn.ReLU()
#         ).to(device)
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 22 * 16 + 20, 8192),
#             nn.ReLU(),
#             nn.Linear(8192, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 512),
#             nn.ReLU(),
#             nn.Linear(512, out_dim),
#         ).to(device)

#     def forward(self, x):
#         B = x.size(0)
        
#         manual_features = []

#         for i in range(B):
#             with torch.no_grad():
#                 np_x = x[i].cpu().numpy()
#                 np_x = np.transpose(np_x, (1, 2, 0))
#                 np_x = np_x.astype(np.uint8)
#                 ghost_locs = get_ghost_locations(np_x) # Length 16
#                 pacman_loc, lives = get_pacman_location(np_x) # Length 2
#                 current_score = get_score_from_frame(np_x)
            
#             feats = pacman_loc.copy()
#             feats.extend(ghost_locs)
#             feats.append(current_score)
#             feats.append(lives) ## Length of feats is 20

#             manual_features.append(feats)

#         manual_tensor = torch.tensor(manual_features, dtype=torch.float32, device=x.device)

#         cnn_features = self.conv(x / 255.0)
#         cnn_flat = cnn_features.reshape(B, -1) # You cannot use view here because dropout made it non contiguous

#         combined = torch.cat([cnn_flat, manual_tensor], dim=1)  

#         out = self.fc(combined)
#         return out

class CNNEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ).to(device)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        ).to(device)
        self.conv_out = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        cnn_features = self.conv(x / 255.0)
        pooled = self.conv_out(cnn_features)
        return self.fc(pooled)


class RepresentationFunction(nn.Module):
    def __init__(self, n_obs, n_skill, config):
        super().__init__()

        self.representation_func = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_skill)
        ).to(device)

        # self.encoder = CNNEncoder(out_dim=256)
        self.encoder = PretrainedResNetEncoder(out_dim=256).to(device)
        self.optimizer = Adam(list(self.parameters()) + list(self.encoder.parameters()), lr=config.representation_lr)
        self.lambda_param = torch.tensor(config.lambda_param, requires_grad=True)
        self.lambda_optimizer = Adam([self.lambda_param], lr=config.lambda_lr)
        self.epsilon = config.epsilon
        self.update_steps = config.n_update_repr

    def get_latent_representation(self, state, normalize=False):
        features = encode_state(state, self.encoder)
        x = self.representation_func(features)
        if normalize:
            return F.normalize(x, p=2, dim=1)
        return x

    def update(self, replay_buffer, i):
        """
        Updates the representation function φ and the Lagrange multiplier λ
        based on skill-consistency and distance constraints.
        """
        for _ in range(self.update_steps): # Update the representation function

          state, skill, action, reward, next_state, not_done  = replay_buffer.sample()
          current_representations = self.get_latent_representation(state)
          next_representations = self.get_latent_representation(next_state)
          # Consistency loss term
          consistency_term = 10 * torch.einsum('bi,bi->b', (next_representations - current_representations), skill) 
          # (\phi_{s'} - \phi_s)^T * z <-- maximize this difference (so in loss term negate it)

          # Distance penalty term (to enforce norm constraints)
          diff_norm_squared = torch.norm(current_representations - next_representations, dim=1) ** 2
          penalty_term = torch.clamp(diff_norm_squared.clone() - self.epsilon, min=0.0) # diff_norm_squared > \epsilon, so if it is bigger, loss =  \epsilon - diff_norm_squares

          # Representation loss

          representation_loss = (-consistency_term + self.lambda_param.view(1).to(device) * penalty_term).mean()

          # Lambda loss (only on penalty term) <-- if penalty is large, lambda should increase -- penalty is bound to be positive, so to minimize the loss
          lambda_loss = -(self.lambda_param * penalty_term.detach()).mean()

          # Backprop: update \phi
          self.optimizer.zero_grad()
          representation_loss.backward(retain_graph=True)
          self.optimizer.step()

          # Backprop: update \lambda
          self.lambda_optimizer.zero_grad()
          lambda_loss.backward()
          self.lambda_optimizer.step()

        writer.add_scalar("loss/representation_loss", representation_loss.item(), i)
        writer.add_scalar("loss/  lambda_loss", lambda_loss.item(), i)


def compute_heatmap(frames, save_path):
    heatmap = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # naive thresholding to isolate the player
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        if heatmap is None:
            heatmap = np.zeros_like(thresh, dtype=np.float32)
        heatmap += thresh.astype(np.float32)

    # Normalize and save heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_representation(replay_buffer, representation, global_step, n_samples=1000, folder='.', n_skills=10):
    skill_data = []
    for skill_id in range(n_skills):
        samples = replay_buffer.sample_by_skill(skill_id, num_samples=max(int(n_samples/10), 100))
        skill_data.append(samples)

    # Combine all samples
    states = torch.cat([s[0] for s in skill_data], dim=0)
    skills = torch.cat([s[1] for s in skill_data], dim=0)

    with torch.no_grad():
        skill_ids = torch.argmax(skills, dim=1)  # assume Gaussian skills aren't passed here!

        phis = representation.get_latent_representation(states, normalize=True).cpu().numpy()
        skill_ids_np = skill_ids.cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=20)
    tsne_result = tsne.fit_transform(phis)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(phis)

    # Plot
    plt.figure(figsize=(8, 6))
    for skill_id in range(n_skills):
        idx = skill_ids_np == skill_id
        if not idx.any():
            continue
        plt.scatter(pca_result[idx, 0], pca_result[idx, 1], label=f'Skill {skill_id}', alpha=0.6)
    plt.legend()
    plt.title('PCA of Representation φ(s)')
    plt.savefig(f"{folder}/pca_representation_pca.png")
    plt.close()


    plt.figure(figsize=(8, 6))
    for skill_id in range(n_skills):
        idx = skill_ids_np == skill_id
        if np.sum(idx) == 0:
            continue
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], label=f'Skill {skill_id}', alpha=0.6)
    plt.legend()
    plt.title('t-SNE of Representation φ(s)')
    plt.savefig(f"{folder}/tsne_representation_tsne.png")
    plt.close()

    if len(np.unique(skill_ids_np)) > 1:
        sil_score = silhouette_score(phis, skill_ids_np)
        writer.add_scalar("representation/silhouette_score", sil_score, global_step)

        db_index = davies_bouldin_score(phis, skill_ids_np)
        writer.add_scalar("representation/davies_bouldin_index", db_index, global_step)



def evaluate_skills(env_name, policy, representation, global_step, writer=None, n_skills=5, steps_per_skill=512, video_dir='skill_videos'):
    os.makedirs(video_dir, exist_ok=True)

    temp_buffer = ReplayBuffer(capacity=steps_per_skill * n_skills)
    temp_buffer.batch_size = 512

    mean_intrinsic_reward = []
    big_vs = set()
    norms = []
    mean_episode_length = []

    one_hot = torch.eye(n_skills, device=device)  # shape (n_skills, n_skills)
    mean = one_hot.mean(dim=0, keepdim=True)
    zero_centered = one_hot - mean  # subtract mean from each row

    for skill_id in range(n_skills): # For every z vector, I will perform ONLY the action related to that skill and get the reward
        env = gym.make(env_name, render_mode='rgb_array')
        obs, _ = env.reset()
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32, device=device), 0).permute(0, 3, 1, 2)

        # USING ONE HOT ENCODED SKILLS HERE SO THAT WE CAN FIND THE DIFFERENT SKILLS SEPARATELY
        skill = zero_centered[skill_id].unsqueeze(0)

        # frames = []
        vs = set()
        total_intrinsic_reward = 0
        step_count = 0
        print(f'evaluating skill {skill_id}')

        while step_count < steps_per_skill:
            with torch.no_grad():
                action = policy.get_action(obs, skill, eval=True).item()
            next_obs, _, terminated, truncated, _ = env.step(action)
            # frame = env.render()
            vs.add(hash(next_obs.tobytes()))

            # next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).view(1, -1)
            next_obs_tensor = torch.unsqueeze(torch.tensor(next_obs, dtype=torch.float32, device=device), 0).permute(0, 3, 1, 2) ## Encoding doesn't matter because get_latent_representation encodes
            delta_phi = representation.get_latent_representation(next_obs_tensor) - representation.get_latent_representation(obs)
            norms.append(delta_phi.norm(dim=1))  
            intrinsic_reward = torch.einsum('bi,bi->b', delta_phi, skill).item()
            # frames.append(frame)
            temp_buffer.store(obs, skill, torch.tensor([action]), torch.tensor([intrinsic_reward]), next_obs_tensor, torch.tensor([~(terminated or truncated)]))
            total_intrinsic_reward += intrinsic_reward

            if terminated or truncated:
                break

            obs = next_obs_tensor
            step_count += 1

        env.close()

        # video_path = os.path.join(video_dir, f"skill_{skill_id}.mp4")
        # imageio.mimsave(video_path, frames, fps=30)

        # del frames
        torch.cuda.empty_cache() 
        mean_intrinsic_reward.append(total_intrinsic_reward)
        big_vs.update(vs)
        mean_episode_length.append(step_count)
        # print(f"Skill {skill_id}: steps = {step_count}, intrinsic reward = {total_intrinsic_reward:.2f}, saved to {video_path}")
        if writer:
            writer.add_scalar(f"eval/intrinsic_reward_skill_{skill_id}", total_intrinsic_reward, global_step)
            writer.add_scalar(f"eval/episode_length_skill_{skill_id}", step_count, global_step)
            writer.add_scalar(f"eval/state_coverage_skill_{skill_id}", len(vs), global_step)

    writer.add_scalar(f"eval/total_state_coverage", len(big_vs), global_step)
    del big_vs

    mean_norm = torch.cat(norms).mean().item()
    writer.add_scalar("representation/mean_delta_phi_norm", mean_norm, global_step)

    mean_intrinsic_reward = sum(mean_intrinsic_reward) / n_skills
    mean_episode_length = sum(mean_episode_length) / n_skills

    writer.add_scalar(f"eval/mean_intrinsic_reward", mean_intrinsic_reward, global_step)
    writer.add_scalar(f"eval/mean_episode_length", mean_episode_length, global_step)
    
    visualize_representation(temp_buffer, representation, global_step, folder=video_dir, n_skills=n_skills)
    plot_skill_action_distributions(policy, temp_buffer, n_skills, 9, save_path=f"{video_dir}/skill_action_distribution.png")
    # check_skill_conditioned_policy(temp_buffer, n_skills, policy)

    ## Evaluate action distribution -- irrespective of the skill
    # states, skills, _, _, _, _ = temp_buffer.sample()
    # with torch.no_grad():
    #     dist  = policy.get_policy_distribution(states, skills)
    #     probs = dist.probs
    # mean_probs = probs.mean(dim=0)

    # for a in range(9): ## 9 actions in Ms. Pacman
    #     writer.add_scalar(f"action_dist/action_{a}", mean_probs[a].item(), epoch)
    # writer.add_histogram("action_dist/all_actions", mean_probs.cpu().numpy(), epoch)

class SkillPolicyNet(nn.Module):
    def __init__(self, obs_dim, skill_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + skill_dim, 512)
        self.fc2 = nn.Linear(512 + skill_dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, n_actions)
        self.skill_scale = 5

    def forward(self, obs, skill):
        skill = torch.tanh(skill * self.skill_scale)
        x = torch.cat([obs, skill], dim=-1)
        x = F.tanh(self.fc1(x))
        x = torch.cat([x, skill], dim=-1)  # reinforce skill injection
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return self.out(x)


def plot_skill_action_distributions(policy, replay_buffer, n_skills, n_actions, save_path="skill_action_distribution.png"):
    fig, axes = plt.subplots(nrows=n_skills, ncols=1, figsize=(8, 3 * n_skills), sharex=True)

    for skill_id in range(n_skills):
        try:
            states, skills, *_ = replay_buffer.sample_by_skill(skill_id, num_samples=512)
        except Exception as e:
            print(f"Skipping skill {skill_id}: {e}")
            continue

        with torch.no_grad():
            dist = policy.get_policy_distribution(states, skills)
            mean_probs = dist.probs.mean(dim=0).cpu().numpy()

        ax = axes[skill_id] if n_skills > 1 else axes
        ax.bar(range(n_actions), mean_probs)
        ax.set_ylim(0, 1)
        ax.set_title(f"Skill {skill_id}")
        ax.set_ylabel("P(a)")
        ax.set_xlabel("Action")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved action distribution plot to {save_path}")

# def check_skill_conditioned_policy(replay_buffer, n_skills, policy):
#     state = replay_buffer.sample()[0][0:1]  # shape [1, obs_dim]
#     one_hot = torch.eye(n_skills, device=device)  # shape (n_skills, n_skills)
#     mean = one_hot.mean(dim=0, keepdim=True)
#     zero_centered = one_hot - mean  # subtract mean from each row
#     for skill_id in range(n_skills):
#         z = zero_centered[skill_id].unsqueeze(0)
#         dist = policy.get_policy_distribution(state, z)
#         print(f"Skill {skill_id}: {dist.probs.detach().cpu().numpy()}")



if __name__ == "__main__":
    env = gym.make('ALE/MsPacman-v5')
    # n_obs = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    n_obs = 256 # Encoded dimension
    n_actions = env.action_space.n

    config = Config(lambda_param=0.001, lambda_lr=8e-5, alpha=0.45, q_lr=1e-4,\
         policy_lr=1e-4, representation_lr=8e-5, timestamp=timestamp, epsilon=0.8, n_skill=4,
         warmup_epochs=0, n_update_policy=40, n_update_repr=40)
    
    n_skill = config.n_skill # 0 - NOOP, 1 - UP, 2 - LEFT, 3 - RIGHT, 4 - DOWN

    hparam_dict = {h.name: getattr(config, h.name) for h in HPARAMS.values()}
    writer.add_hparams(hparam_dict, {m: 0 for m in METRICS})  # placeholders -- metrics will get updated by scalars that will be added later

    representation = RepresentationFunction(n_obs, n_skill, config)
    policy = SkillPolicy(n_obs, n_skill, n_actions, representation, config)

    start_epoch = 0

    chkpt_path = ''
    if len(chkpt_path) > 0:
        if not os.path.exists(chkpt_path):
            raise ValueError("Load folder path doesn't exist")
        checkpoint = torch.load(chkpt_path)
        policy.policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.q1_net.load_state_dict(checkpoint['q1_state_dict'])
        policy.q2_net.load_state_dict(checkpoint['q2_state_dict'])
        policy.q1_target_net.load_state_dict(checkpoint['q1_target_state_dict'])
        policy.q2_target_net.load_state_dict(checkpoint['q2_target_state_dict'])
        policy.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        policy.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])

        representation.representation_func.load_state_dict(checkpoint['representation_state_dict'])
        representation.optimizer.load_state_dict(checkpoint['representation_optimizer_state_dict'])
        representation.lambda_param = torch.tensor(checkpoint['lambda_param'], requires_grad=True)
        representation.lambda_optimizer.load_state_dict(checkpoint['lambda_optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1


    # Training loop
    num_epochs = 551

    drl = DRL(buffer_size = 10_000)

    if not os.path.exists(f"./model/{timestamp}"):
        os.makedirs(f"./model/{timestamp}")

    folder = f'./model/{timestamp}'

    for epoch in range(start_epoch, num_epochs + start_epoch):
        print(f'EPOCH: {epoch}')
        drl.rollout(policy, epoch, n_skill, representation.encoder)
        representation.update(drl.replay_buffer, epoch)

        if epoch >= config.warmup_epochs:
            policy.update(drl.replay_buffer, epoch)
        if epoch % 10 == 0:
            ## EVALUATING SKILL
            evaluate_skills('ALE/MsPacman-v5', policy, representation,\
                 epoch, writer=writer, n_skills=n_skill, video_dir=f'{folder}/video_eval_{epoch}') ## ONLY doing for the first 3 skills

            gc.collect()
        cleanup_memory()

    torch.save({
        'epoch': num_epochs,
        'policy_state_dict': policy.policy.state_dict(),
        'q1_state_dict': policy.q1_net.state_dict(),
        'q2_state_dict': policy.q2_net.state_dict(),
        'q1_target_state_dict': policy.q1_target_net.state_dict(),
        'q2_target_state_dict': policy.q2_target_net.state_dict(),
        'policy_optimizer_state_dict': policy.policy_optimizer.state_dict(),
        'q_optimizer_state_dict': policy.q_optimizer.state_dict(),

        'representation_state_dict': representation.representation_func.state_dict(),
        'representation_optimizer_state_dict': representation.optimizer.state_dict(),
        'lambda_param': representation.lambda_param.detach().item(),
        'lambda_optimizer_state_dict': representation.lambda_optimizer.state_dict(),
    }, f"./model/{timestamp}/checkpoint.pth")
