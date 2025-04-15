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

import cv2
import matplotlib.pyplot as plt
import imageio
import tqdm

from sklearn.manifold import TSNE
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import gc

import os
from gymnasium.wrappers import RecordVideo

# code should work on either, faster on gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.autograd.set_detect_anomaly(True)

# random seeds for reproducability
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

now = datetime.now()
timestamp = now.strftime("%d_%H%M%S")
writer = SummaryWriter(log_dir=f'runs/SAC-Discrete-{timestamp}')

ale = ALEInterface()
gym.register_envs(ale_py)

def cleanup_memory():
    gc.collect()  # Clean up CPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()            # Release cached GPU memory (PyTorch only)
        torch.cuda.ipc_collect()            # Clean up interprocess memory (multi-GPU safe)


# @title Define Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = 512

    def store(self, state, skill, action, reward, next_state, done):
        transitions = list(zip(state, skill, action, reward, next_state, 1 - torch.Tensor(done)))
        self.buffer.extend(transitions)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return [torch.stack(e).to(device) for e in zip(*batch)]  # state, skill, action, reward, next_state, not_done

    def __len__(self):
        return len(self.buffer)

class DRL:
    def __init__(self, buffer_size = 10000):
        self.n_envs = 4 # n_envs different runs of the same environment
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

        ## SAMPLE ONLY ONE SKILL PER ENVIRONMENT
        skills = torch.eye(n_skill)[torch.randint(0, n_skill, (self.n_envs,))].to(device) # One hot encoded skills

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


class SkillPolicy:
    def __init__(self, n_obs, n_skills, n_actions, encoder):
        self.alpha = 0.1
        self.n_actions = n_actions
        self.encoder = encoder

        self.q1_net = nn.Sequential(
            nn.Linear(n_obs + n_skills, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        ).to(device)

        self.q2_net = copy.deepcopy(self.q1_net).to(device)

        self.policy = nn.Sequential(
            nn.Linear(n_obs + n_skills, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        ).to(device)

        self.q1_target_net = copy.deepcopy(self.q1_net).to(device)
        self.q2_target_net = copy.deepcopy(self.q1_net).to(device)

        self.q_optimizer = Adam(
            list(self.q1_net.parameters()) + list(self.q2_net.parameters()), lr=3e-4
        )
        self.policy_optimizer = Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = 0.99

    def get_policy_distribution(self, states, skills):
        states = encode_state(states, self.encoder)
        inputs = torch.cat([states, skills], dim=-1)
        logits = self.policy(inputs)
        return Categorical(logits=logits)

    def get_action(self, states, skills, eval=False):
        dist = self.get_policy_distribution(states, skills)
        if eval:
            return torch.argmax(dist.probs, dim=-1)
        return dist.sample()

    def get_entropy(self, states, skills):
        dist = self.get_policy_distribution(states, skills)
        return dist.entropy()

    def get_q_loss(self, states, actions, rewards, next_states, not_dones, skills):
        with torch.no_grad():
            intrinsic_reward = torch.einsum('bi,bi->b', \
                (representation.get_latent_representation(next_states) -\
                     representation.get_latent_representation(states)), skills).unsqueeze(-1)
            
            next_states = encode_state(next_states, self.encoder)
            next_input = torch.cat([next_states, skills], dim=-1)
            next_q1 = self.q1_target_net(next_input)
            next_q2 = self.q2_target_net(next_input)
            next_q = torch.min(next_q1, next_q2)
            next_pi = policy.get_policy_distribution(next_states, skills)
            next_entropy = next_pi.entropy().unsqueeze(-1)
            next_q_val = (next_pi.probs * (next_q - self.alpha * next_entropy)).sum(dim=-1, keepdim=True)
            q_target = intrinsic_reward + self.gamma * not_dones * next_q_val

        states = encode_state(states, self.encoder)
        current_inputs = torch.cat([states, skills], dim=-1)
        q1 = self.q1_net(current_inputs).gather(1, actions.long().unsqueeze(-1))
        q2 = self.q2_net(current_inputs).gather(1, actions.long().unsqueeze(-1))

        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        return loss


    def get_policy_loss(self, states, skills):
        dist = self.get_policy_distribution(states, skills)
        probs = dist.probs
        log_probs = dist.logits.log_softmax(dim=-1)

        states = encode_state(states, self.encoder)

        inputs = torch.cat([states, skills], dim=-1)
        q1 = self.q1_net(inputs)
        q2 = self.q2_net(inputs)
        q = torch.min(q1, q2)

        policy_loss = -(probs * (q - self.alpha * log_probs)).sum(dim=1).mean()
        return policy_loss

    def update(self, replay_buffer, i):

      for _ in range(10):
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

class CNNEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # (3, 210, 160) → (32, 52, 39)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # → (64, 25, 18)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # → (64, 23, 16)
            nn.ReLU()
        ).to(device)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 16, out_dim),
            nn.ReLU()
        ).to(device)

    def forward(self, x):
        return self.fc(self.conv(x / 255.0))


class RepresentationFunction(nn.Module):
    def __init__(self, n_obs, n_skill):
        super().__init__()

        self.representation_func = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_skill)
        ).to(device)

        self.encoder = CNNEncoder(out_dim=256)
        self.optimizer = Adam(list(self.parameters()) + list(self.encoder.parameters()), lr=3e-4)
        self.lambda_param = torch.tensor(1.0, requires_grad=True)
        self.lambda_optimizer = Adam([self.lambda_param], lr=3e-4)
        self.epsilon = 0.1

    def get_latent_representation(self, state):
        features = encode_state(state, self.encoder)
        x = self.representation_func(features)
        return F.normalize(x, p=2, dim=1)

    def update(self, replay_buffer, i):
        """
        Updates the representation function φ and the Lagrange multiplier λ
        based on skill-consistency and distance constraints.
        """
        for _ in range(10):

          state, skill, action, reward, next_state, not_done  = replay_buffer.sample()

          current_representations = self.get_latent_representation(state)    # φ(s)
          next_representations = self.get_latent_representation(next_state)  # φ(s')

          # Consistency loss term
          consistency_term = torch.einsum('bi,bi->b', (next_representations - current_representations), skill)

          # Distance penalty term (to enforce norm constraints)
          diff_norm_squared = torch.norm(current_representations - next_representations, dim=1) ** 2
          penalty_term = torch.minimum(
              torch.tensor(self.epsilon, device=diff_norm_squared.device),
              (1.0 - diff_norm_squared).clone()
          )

          # Representation loss
          representation_loss = (consistency_term + self.lambda_param.detach() * penalty_term).mean()

          # Lambda loss (only on penalty term)
          lambda_loss = (self.lambda_param * penalty_term.detach()).mean()

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


def visualize_representation(replay_buffer, representation, n_samples=1000, folder='.'):
    states, skills, *_ = replay_buffer.sample()
    states = states[:n_samples]
    skills = skills[:n_samples]
    with torch.no_grad():
        phis = representation.get_latent_representation(states).cpu().numpy()
        skills_np = skills.argmax(dim=1).cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=10)
    tsne_result = tsne.fit_transform(phis)

    plt.figure(figsize=(8, 6))
    for skill in np.unique(skills_np):
        idx = skills_np == skill
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], label=f'Skill {skill}', alpha=0.6)
    plt.legend()
    plt.title('t-SNE of Representation φ(s)')
    plt.savefig(f"{folder}/representation_tsne.png")
    plt.close()



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


def evaluate_skills(env_name, policy, representation, global_step, writer=None, n_skills=5, steps_per_skill=512, video_dir='skill_videos'):
    os.makedirs(video_dir, exist_ok=True)

    for skill_id in range(min(5, n_skills)): # For every z vector, I will perform ONLY the action related to that skill and get the reward
        env = gym.make(env_name, render_mode='rgb_array')
        obs, _ = env.reset()
        # obs = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1) ## NO need to flatten CNN encoder in representation function will handle it
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32, device=device), 0).permute(0, 3, 1, 2)

        skill = torch.zeros(1, n_skills, device=device)
        skill[0, skill_id] = 1.0

        frames = []
        total_intrinsic_reward = 0
        step_count = 0

        while step_count < steps_per_skill:
            with torch.no_grad():
                action = policy.get_action(obs, skill, eval=True).item()
            next_obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(frame)

            # next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).view(1, -1)
            next_obs_tensor = torch.unsqueeze(torch.tensor(next_obs, dtype=torch.float32, device=device), 0).permute(0, 3, 1, 2) ## Encoding doesn't matter because get_latent_representation encodes
            delta_phi = representation.get_latent_representation(next_obs_tensor) - representation.get_latent_representation(obs)
            intrinsic_reward = torch.einsum('bi,bi->b', delta_phi, skill).item()
            total_intrinsic_reward += intrinsic_reward

            if terminated or truncated:
                break

            obs = next_obs_tensor
            step_count += 1

        env.close()

        video_path = os.path.join(video_dir, f"skill_{skill_id}.mp4")
        imageio.mimsave(video_path, frames, fps=30)

        heatmap_path = os.path.join(video_dir, f"skill_{skill_id}_heatmap.png")
        compute_heatmap(frames, heatmap_path)

        del frames
        torch.cuda.empty_cache()  

        # print(f"Skill {skill_id}: steps = {step_count}, intrinsic reward = {total_intrinsic_reward:.2f}, saved to {video_path}")
        if writer:
            writer.add_scalar(f"eval/intrinsic_reward_skill_{skill_id}", total_intrinsic_reward, global_step)
            writer.add_scalar(f"eval/episode_length_skill_{skill_id}", step_count, global_step)

if __name__ == "__main__":
    env = gym.make('ALE/MsPacman-v5')
    # n_obs = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    n_obs = 256 # Encoded dimension
    n_actions = env.action_space.n
    n_skill = 10 # 0 - NOOP, 1 - UP, 2 - LEFT, 3 - RIGHT, 4 - DOWN

    representation = RepresentationFunction(n_obs, n_skill)
    policy = SkillPolicy(n_obs, n_skill, n_actions, representation.encoder)

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
        policy.update(drl.replay_buffer, epoch)
        if epoch % 10 == 0:
            ## EVALUATING SKILL
            evaluate_skills('ALE/MsPacman-v5', policy, representation,\
                 epoch, writer=writer, n_skills=n_skill, video_dir=f'{folder}/video_eval_{epoch}') ## ONLY doing for the first 3 skills

            ## EVALUATING REPRESENTATION FUNCTION
            visualize_representation(replay_buffer=drl.replay_buffer,\
                 representation=representation, folder=f"{folder}/video_eval_{epoch}")
            # writer.add_image("eval/tSNE", \
            #      torch.tensor(np.array(Image.open(f"{folder}/video_eval_{epoch}/representation_tsne.png"))).permute(2, 0, 1), epoch)

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
