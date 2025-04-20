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
import matplotlib.cm as cm
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

import warnings

# code should work on either, faster on gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.autograd.set_detect_anomaly(True)

now = datetime.now()
timestamp = now.strftime("%d_%H%M%S")
writer = SummaryWriter(log_dir=f'runs/SAC-{timestamp}')

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

visited_bins = set()

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
            [lambda: gym.make('Ant-v5') for _ in range(self.n_envs)])

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def rollout(self, agent, i, n_skill):
        """Collect experience and store it in the replay buffer"""

        obs, _ = self.envs.reset()
        # obs = torch.tensor(obs, dtype=torch.float32, device=device).view(obs.shape[0], -1) ## CURRENTLY JUST FLATTENING, BUT CAN POTENTIALLY USE CNN TO EXTRACT FEATURES
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        enc_obs = obs

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
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            not_done = ~(dones | truncateds)

            self.replay_buffer.store(obs, skills, actions, rewards, next_obs, not_done) ### SO THE REPLAY BUFFER STORE 4 experiences 
            obs = next_obs
            total_rewards += rewards

        writer.add_scalar("stats/Rewards", total_rewards.mean().item() / self.n_steps, i)


class SkillQNet(nn.Module):
    def __init__(self, obs_dim, skill_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + skill_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, skill, action):
        x = torch.cat([obs, skill, action], dim=-1)
        return self.fc(x)

class SkillPolicy:
    def __init__(self, n_obs, n_skills, n_actions, representation, config):
        self.alpha_start = config.alpha
        self.alpha_end = config.alpha - (0.001 * 551) # Ensuring that I get 0.0001 step down every epoch -- which isn't too much -- hopefully ideal
        self.n_actions = n_actions
        self.representation = representation
        self.n_step_update = config.n_update_policy

        self.policy = GaussianSkillPolicyNet(obs_dim=n_obs, skill_dim=n_skills, action_dim=n_actions).to(device)

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
        mean, std = self.policy(states, skills)
        dist = torch.distributions.Normal(mean, std)
        return dist


    def sample_action(self, mean, std):
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def get_action(self, states, skills, eval=False):
        mean, std = self.policy(states, skills)
        if eval:
            return torch.tanh(mean)
        action, _ = self.sample_action(mean, std)
        return action


    def get_q_loss(self, states, actions, rewards, next_states, not_dones, skills):
        with torch.no_grad():
            phi = self.representation.get_latent_representation(states)
            phi_next = self.representation.get_latent_representation(next_states)
            delta_phi = phi_next - phi
            intrinsic_reward = torch.einsum('bi,bi->b', delta_phi, skills).unsqueeze(-1)

            next_states_encoded = next_states
            mean, std = self.policy(next_states_encoded, skills)
            next_action, log_prob = self.sample_action(mean, std)

            target_q1 = self.q1_target_net(next_states_encoded, skills, next_action)
            target_q2 = self.q2_target_net(next_states_encoded, skills, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob

            q_target = 10 * intrinsic_reward + self.gamma * not_dones.unsqueeze(-1) * target_q

        states_encoded = states
        q1 = self.q1_net(states_encoded, skills, actions)
        q2 = self.q2_net(states_encoded, skills, actions)
        return F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)


    def get_policy_loss(self, states, skills):
        states_encoded = states
        mean, std = self.policy(states_encoded, skills)
        action, log_prob = self.sample_action(mean, std)
        
        q1 = self.q1_net(states_encoded, skills, action)
        q2 = self.q2_net(states_encoded, skills, action)
        q = torch.min(q1, q2)

        return (self.alpha * log_prob - q).mean()


    def get_entropy(self, states, skills):
        mean, std = self.policy(states, skills)
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)  # sum over action dimensions
        return entropy


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


class RepresentationFunction(nn.Module):
    def __init__(self, n_obs, n_skill, config):
        super().__init__()

        self.representation_func = nn.Sequential(
            nn.Linear(n_obs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_skill)
        ).to(device)

        self.optimizer = Adam(list(self.parameters()), lr=config.representation_lr)
        self.lambda_param = torch.tensor(config.lambda_param, requires_grad=True)
        self.lambda_optimizer = Adam([self.lambda_param], lr=config.lambda_lr)
        self.epsilon = config.epsilon
        self.update_steps = config.n_update_repr

    def get_latent_representation(self, state, normalize=True):
        x = self.representation_func(state)
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
    trajectories = {skill_id: [] for skill_id in range(n_skills)}

    temp_buffer = ReplayBuffer(capacity=steps_per_skill * n_skills)
    temp_buffer.batch_size = 512

    mean_intrinsic_reward = []
    norms = []
    mean_episode_length = []

    one_hot = torch.eye(n_skills, device=device)  # shape (n_skills, n_skills)
    mean = one_hot.mean(dim=0, keepdim=True)
    zero_centered = one_hot - mean  # subtract mean from each row

    for skill_id in range(n_skills): # For every z vector, I will perform ONLY the action related to that skill and get the reward
        env = gym.make(env_name, render_mode='rgb_array')
        positions = []

        obs, _ = env.reset()
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32, device=device), 0)

        # USING ONE HOT ENCODED SKILLS HERE SO THAT WE CAN FIND THE DIFFERENT SKILLS SEPARATELY
        skill = zero_centered[skill_id].unsqueeze(0)

        frames = []
        total_intrinsic_reward = 0
        step_count = 0
        print(f'evaluating skill {skill_id}')

        while step_count < steps_per_skill:
            with torch.no_grad():
                action = policy.get_action(obs, skill, eval=True).squeeze(0).cpu().numpy()
            next_obs, _, terminated, truncated, _ = env.step(action)
            phi = representation.get_latent_representation(obs)
            for vec in phi:
                visited_bins.add(get_state_bin(vec))

            pos = next_obs[:2]  # use [0] if only X
            positions.append(pos)
            trajectories[skill_id].append(np.array(positions))  # store this seed's path for this skill

            # next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).view(1, -1)
            next_obs_tensor = torch.unsqueeze(torch.tensor(next_obs, dtype=torch.float32, device=device), 0) ## Encoding doesn't matter because get_latent_representation encodes
            delta_phi = representation.get_latent_representation(next_obs_tensor) - representation.get_latent_representation(obs)
            norms.append(delta_phi.norm(dim=1))  
            intrinsic_reward = torch.einsum('bi,bi->b', delta_phi, skill).item()

            frame = env.render()
            frames.append(frame)
            temp_buffer.store(obs, skill, torch.tensor([action]), torch.tensor([intrinsic_reward]), next_obs_tensor, torch.tensor([~(terminated or truncated)]))

            total_intrinsic_reward += intrinsic_reward

            if terminated or truncated:
                break

            obs = next_obs_tensor
            step_count += 1

        env.close()

        video_path = os.path.join(video_dir, f"skill_{skill_id}.mp4")
        imageio.mimsave(video_path, frames, fps=30)

        del frames
        torch.cuda.empty_cache() 
        mean_intrinsic_reward.append(total_intrinsic_reward)
        mean_episode_length.append(step_count)
        # print(f"Skill {skill_id}: steps = {step_count}, intrinsic reward = {total_intrinsic_reward:.2f}, saved to {video_path}")
        if writer:
            writer.add_scalar(f"eval/intrinsic_reward_skill_{skill_id}", total_intrinsic_reward, global_step)
            writer.add_scalar(f"eval/episode_length_skill_{skill_id}", step_count, global_step)
    
    colors = plt.cm.get_cmap("tab10", n_skills)

    colors = cm.get_cmap("tab10", n_skills)
    plt.figure(figsize=(8, 6))
    for skill_id, traj_list in trajectories.items():
        # Flatten and concatenate all trajectories for this skill
        all_positions = np.concatenate(traj_list, axis=0)  # shape: (total_steps, 2)
        plt.plot(all_positions[:, 0], all_positions[:, 1], color=colors(skill_id), label=f'Skill {skill_id}', alpha=0.6)

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Ant-v5: X-Y Trajectories Colored by Skill (across seeds)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(f"{video_dir}/ant_trajectory_all_skills.png")
    plt.close()



    mean_norm = torch.cat(norms).mean().item()
    writer.add_scalar("representation/mean_delta_phi_norm", mean_norm, global_step)
    writer.add_scalar("eval/total_state_coverage_bins", len(visited_bins), global_step)

    mean_intrinsic_reward = sum(mean_intrinsic_reward) / n_skills
    mean_episode_length = sum(mean_episode_length) / n_skills

    writer.add_scalar(f"eval/mean_intrinsic_reward", mean_intrinsic_reward, global_step)
    writer.add_scalar(f"eval/mean_episode_length", mean_episode_length, global_step)
    
    visualize_representation(temp_buffer, representation, global_step, folder=video_dir, n_skills=n_skills)
    # plot_skill_action_distributions(policy, temp_buffer, n_skills, 9, save_path=f"{video_dir}/skill_action_distribution.png")

class GaussianSkillPolicyNet(nn.Module):
    def __init__(self, obs_dim, skill_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.skill_scale = 5

    def forward(self, obs, skill):
        skill = torch.tanh(skill * self.skill_scale)
        x = torch.cat([obs, skill], dim=-1)
        x = self.fc(x)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

def get_state_bin(phi, bin_size=0.1):
    """Discretize latent state into a tuple key using fixed-size bins."""
    return tuple((phi / bin_size).floor().int().tolist())






if __name__ == "__main__":
    env = gym.make('Ant-v5', ctrl_cost_weight=0.5)
    n_obs = env.observation_space.shape[0]
    n_actions = 8 # actually action_dim now

    config = Config(lambda_param=30.0, lambda_lr=1e-4, alpha=0.1, q_lr=1e-4,\
         policy_lr=1e-4, representation_lr=1e-4, timestamp=timestamp, epsilon=1e-3, n_skill=8,
         warmup_epochs=0, n_update_policy=50, n_update_repr=50)
    
    n_skill = config.n_skill

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

    if not os.path.exists(f"./model/SAC/{timestamp}"):
        os.makedirs(f"./model/SAC/{timestamp}")

    folder = f'./model/SAC/{timestamp}'

    for epoch in range(start_epoch, num_epochs + start_epoch):
        print(f'EPOCH: {epoch}')
        drl.rollout(policy, epoch, n_skill)
        representation.update(drl.replay_buffer, epoch)

        if epoch >= config.warmup_epochs:
            policy.update(drl.replay_buffer, epoch)
        if epoch % 10 == 0:
            ## EVALUATING SKILL
            evaluate_skills('Ant-v5', policy, representation,\
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
    }, f"./model/SAC/{timestamp}/checkpoint.pth")
