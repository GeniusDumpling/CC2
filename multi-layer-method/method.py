"""Shared graph encoder + subnet -> host -> action, trained with standard SB3 PPO.

Inspired by kasanari/incident-response-rl-gnn's local message passing and
node-first policy; implemented independently using PyTorch and standard SB3.
"""
import torch
from torch import nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import BasePolicy


class SubnetPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, width=64,
                 use_sde=False, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        if use_sde:
            raise ValueError('Discrete policy does not support gSDE')
        self.width = width
        self.embed = nn.Sequential(nn.Linear(4, width), nn.Tanh())
        self.messages = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.updates = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(2)])
        # Nonlinear mixing keeps shared context from cancelling under softmax.
        self.subnet_head = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))
        self.host_head = nn.Sequential(nn.Linear(3 * width, width), nn.Tanh(), nn.Linear(width, 1))
        self.action_head = nn.Sequential(nn.Linear(3 * width, width), nn.Tanh(), nn.Linear(width, 5))
        self.value_head = nn.Linear(width, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def joint_distribution(self, obs):
        x = self.embed(obs['nodes'].float())
        adjacency = obs['adjacency'].bool()
        # ponytail: dense O(N^2) aggregation for small CAGE-2 graphs; use sparse edges for large networks.
        for message, update in zip(self.messages, self.updates):
            neighbors = message(x)[:, None, :, :].expand(-1, x.shape[1], -1, -1)
            pooled = neighbors.masked_fill(~adjacency[..., None], -torch.inf).amax(2)
            x = x + torch.tanh(update(torch.cat([x, pooled], -1)))
        membership = obs['membership'].float()
        subnets = membership @ x / membership.sum(-1, keepdim=True).clamp_min(1)
        global_x = x[:, :-1].mean(1)  # omit the global-action dummy host
        global_s = global_x[:, None, :].expand_as(subnets)
        b, s, _ = subnets.shape
        n = x.shape[1]
        context = torch.cat([
            x[:, None].expand(-1, s, -1, -1),
            subnets[:, :, None].expand(-1, -1, n, -1),
            global_x[:, None, None].expand(-1, s, n, -1),
        ], -1)
        action_mask = obs['action_mask'].bool()[:, None].expand(-1, s, -1, -1)
        host_mask = membership.bool() & action_mask.any(-1)
        subnet_mask = host_mask.any(-1)
        if not subnet_mask.any(-1).all():
            raise ValueError('Observation contains no valid action')

        def probabilities(logits, mask):
            # Inactive conditional rows receive a dummy distribution, then zero mass.
            active = mask.any(-1, keepdim=True)
            safe_mask = mask | ~active
            return logits.masked_fill(~safe_mask, -torch.inf).softmax(-1) * active

        pz = probabilities(self.subnet_head(torch.cat([subnets, global_s], -1)).squeeze(-1), subnet_mask)
        ph = probabilities(self.host_head(context).squeeze(-1), host_mask)
        pa = probabilities(self.action_head(context), action_mask)
        joint = pz[:, :, None, None] * ph[:, :, :, None] * pa
        return joint, self.value_head(global_x)

    def forward(self, obs, deterministic=False):
        joint, values = self.joint_distribution(obs)
        rows = torch.arange(len(joint), device=joint.device)
        if deterministic:
            # Per-level marginal argmax can select a low-probability cell, so search the
            # joint instead. Illegal cells are exactly zero, so a flat argmax is
            # equivalent to restricting the search to the legal tuples.
            z, h, a = torch.unravel_index(joint.flatten(1).argmax(1), joint.shape[1:])
        else:
            # Ancestral sampling is an exact draw from the same joint distribution.
            z = Categorical(probs=joint.sum((2, 3))).sample()
            h = Categorical(probs=joint[rows, z].sum(-1)).sample()
            a = Categorical(probs=joint[rows, z, h]).sample()
        actions = torch.stack([z, h, a], -1)
        return actions, values, joint[rows, z, h, a].clamp_min(1e-30).log()

    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)[0]

    def evaluate_actions(self, obs, actions):
        joint, values = self.joint_distribution(obs)
        z, h, a = actions.long().unbind(-1)
        log_prob = joint[torch.arange(len(joint), device=joint.device), z, h, a].clamp_min(1e-30).log()
        entropy = -(joint * joint.clamp_min(1e-30).log()).sum((1, 2, 3))
        return values, log_prob, entropy

    def predict_values(self, obs):
        return self.joint_distribution(obs)[1]

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(), width=self.width, lr_schedule=self._dummy_schedule)
