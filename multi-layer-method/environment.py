"""CAGE-2 Blue observation adapter; no hidden compromise state is exposed."""
from pathlib import Path
import random
import sys

import gymnasium as gym
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1] / 'repos' / 'cage-challenge-2'
sys.path.insert(0, str(REPO / 'CybORG'))
import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Shared.Actions import Sleep, Monitor, Analyse, Remove, Restore

ACTION_NAMES = ['Sleep', 'Monitor', 'Analyse', 'Remove', 'Restore']
ACTIONS = [Sleep, Monitor, Analyse, Remove, Restore]
RED_AGENTS = {'b_line': B_lineAgent, 'meander': RedMeanderAgent, 'sleep': SleepAgent}


class SubnetDefenseEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scenario=None, red_agent='meander', max_steps=50):
        super().__init__()
        if max_steps < 1:
            raise ValueError('max_steps must be positive')
        if not Path(CybORG.__file__).resolve().is_relative_to(REPO):
            raise RuntimeError('Another CybORG was imported first; run from a fresh Python process')
        self.scenario = Path(scenario or REPO / 'CybORG/CybORG/Shared/Scenarios/Scenario2.yaml').resolve()
        config = yaml.safe_load(self.scenario.read_text())
        self.hosts = sorted(config['Agents']['Blue']['INT']['Hosts'])
        self.subnets = sorted(config['Subnets'])
        self.n_hosts, self.n_subnets = len(self.hosts), len(self.subnets)
        n, s = self.n_hosts + 1, self.n_subnets + 1
        self.membership = np.zeros((s, n), np.float32)
        for z, name in enumerate(self.subnets):
            for host in config['Subnets'][name]['Hosts']:
                if host in self.hosts:
                    self.membership[z, self.hosts.index(host)] = 1
        if not np.all(self.membership[:, :-1].sum(0) == 1):
            raise ValueError('Each observed host must belong to exactly one subnet')
        self.membership[-1, -1] = 1  # canonical global branch for Sleep/Monitor
        self.adjacency = np.eye(n, dtype=np.float32)
        # Coarse declared subnet connectivity, not exploit reachability or hidden state.
        for i in range(self.n_hosts):
            zi = self.subnets[int(self.membership[:, i].argmax())]
            for j in range(self.n_hosts):
                zj = self.subnets[int(self.membership[:, j].argmax())]
                nacls = config['Subnets'][zi].get('NACLs', {})
                rule = nacls.get(zj, nacls.get('all', {}))
                self.adjacency[i, j] = float(zi == zj or rule.get('out') == 'all')
        self.footholds = {x['hostname'] for x in config['Agents']['Red']['starting_sessions']}
        self.simulator = CybORG.CybORG(str(self.scenario), 'sim', agents={'Red': RED_AGENTS[red_agent]})
        self.table = BlueTableWrapper(self.simulator, output_mode='table')
        self.max_steps, self.steps = max_steps, 0
        self.observation_space = gym.spaces.Dict({
            'nodes': gym.spaces.Box(0, 1, (n, 4), np.float32),
            'adjacency': gym.spaces.Box(0, 1, (n, n), np.float32),
            'membership': gym.spaces.Box(0, 1, (s, n), np.float32),
            'action_mask': gym.spaces.Box(0, 1, (n, 5), np.float32),
        })
        self.action_space = gym.spaces.MultiDiscrete([s, n, 5])

    def _observation(self, result):
        activity = {'None': [0, 0], 'Scan': [1, 0], 'Exploit': [1, 1]}
        access = {'No': [0, 0], 'Unknown': [1, 0], 'User': [0, 1], 'Privileged': [1, 1]}
        rows = {row[2]: row for row in result.observation.rows}
        nodes = np.zeros((self.n_hosts + 1, 4), np.float32)
        mask = np.zeros((self.n_hosts + 1, 5), np.float32)
        allowed = self.simulator.get_action_space('Blue')
        for i, name in enumerate(self.hosts):
            row = rows[name]
            nodes[i] = activity[row[3]] + access[row[4]]
            if name not in self.footholds and allowed['hostname'].get(name, False):
                for a in range(2, 5):
                    mask[i, a] = bool(allowed['action'].get(ACTIONS[a], False))
        mask[-1, 0] = 1
        mask[-1, 1] = bool(allowed['action'].get(Monitor, False))
        self.last_obs = {'nodes': nodes, 'adjacency': self.adjacency.copy(),
                         'membership': self.membership.copy(), 'action_mask': mask}
        return self.last_obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('No reset options supported')
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.simulator.set_seed(seed)
            self.action_space.seed(seed)
        self.steps = 0
        return self._observation(self.table.reset(agent='Blue')), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action indices: {action}')
        z, h, a = map(int, action)
        if not self.membership[z, h] or not self.last_obs['action_mask'][h, a]:
            raise ValueError(f'Masked subnet/host/action: {action}')
        if a == 0:
            command = Sleep()
        elif a == 1:
            command = Monitor(session=0, agent='Blue')
        else:
            command = ACTIONS[a](session=0, agent='Blue', hostname=self.hosts[h])
        result = self.table.step(agent='Blue', action=command)
        self.steps += 1
        terminated = bool(result.done)
        info = {'subnet': self.subnets[z] if z < self.n_subnets else 'global',
                'host': self.hosts[h] if h < self.n_hosts else None, 'action': ACTION_NAMES[a]}
        return self._observation(result), float(result.reward), terminated, self.steps >= self.max_steps and not terminated, info

    def close(self):
        self.simulator.shutdown()
