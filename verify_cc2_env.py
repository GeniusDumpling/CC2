"""Run with either cc2-native or cc2-plus; no repository changes required.

Default: Scenario2, Blue vs B_line, 100-step episodes, 512 PPO steps.
Use --raw-check to diagnose the repository's unadapted ChallengeWrapper.
This is an integration smoke test, not a policy performance evaluation.
"""

import argparse
from pathlib import Path
import sys

import CybORG
import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed


class CC2Compatibility(gym.Env):
    """Adapt only the flat Blue ChallengeWrapper interface used in this test."""

    metadata = {"render_modes": []}

    def __init__(self, simulator, challenge):
        super().__init__()
        self.simulator = simulator
        self.challenge = challenge
        self.modern_api = isinstance(challenge, gym.Env)
        space = challenge.observation_space
        self.observation_space = gym.spaces.Box(
            low=space.low, high=space.high, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(challenge.action_space.n)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("This smoke test does not support reset options")
        if seed is not None:
            self.simulator.set_seed(seed)
            np.random.seed(seed)
            self.action_space.seed(seed)
        result = self.challenge.reset()
        obs, info = result if self.modern_api else (result, {})
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        result = self.challenge.step(int(action))
        if self.modern_api:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, info = result
            truncated = False
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward), bool(terminated), bool(truncated), info,
        )

    def close(self):
        self.simulator.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--raw-check", action="store_true")
    args = parser.parse_args()
    print("Python:", sys.executable, flush=True)
    print("CybORG:", CybORG.__file__, flush=True)
    print("SB3:", sb3.__version__, "Torch:", torch.__version__, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if torch.cuda.is_available():
        value = (torch.ones(4, device="cuda") * 2).sum().item()
        assert value == 8, "CUDA tensor calculation failed"
        print("[PASS] CUDA tensor calculation:", torch.cuda.get_device_name(0), flush=True)

    set_random_seed(42)
    torch.set_num_threads(1)
    scenario = Path(CybORG.__file__).parent / "Shared/Scenarios/Scenario2.yaml"
    if not scenario.is_file():
        raise FileNotFoundError(scenario)
    simulator = CybORG.CybORG(str(scenario), "sim", agents={"Red": B_lineAgent})
    # Leave the repository's time limit disabled: it conflates termination
    # and truncation. Gymnasium TimeLimit handles the horizon separately.
    challenge = ChallengeWrapper("Blue", simulator, max_steps=None)
    env = gym.wrappers.TimeLimit(CC2Compatibility(simulator, challenge), 100)
    try:
        if args.raw_check:
            check_env(challenge)
            print("[PASS] Unadapted repository interface", flush=True)
            return

        check_env(env, warn=True)
        print("[PASS] SB3 check_env (compatibility adapter enabled)", flush=True)
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        endings = 0
        for _ in range(150):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs), "Observation outside declared space"
            assert np.isfinite(reward), "Non-finite reward"
            assert isinstance(info, dict)
            if terminated or truncated:
                endings += 1
                obs, info = env.reset()
                assert env.observation_space.contains(obs)
        assert endings > 0, "Episode boundary was not exercised"
        print("[PASS] 150 random steps, episode endings:", endings, flush=True)

        # Explicitly test horizon semantics independent of a long rollout.
        env.reset(seed=42)
        env._elapsed_steps = 99
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        assert truncated, "Time limit must set truncated=True"

        model = sb3.PPO(
            "MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=2,
            seed=42, device=args.device, verbose=0,
        )
        before = [p.detach().clone() for p in model.policy.parameters()]
        model.learn(total_timesteps=512)
        assert model.num_timesteps == 512
        assert all(torch.isfinite(p).all().item() for p in model.policy.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before, model.policy.parameters())), (
            "PPO did not update its parameters"
        )
        obs, _ = env.reset(seed=43)
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
        obs, reward, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        print("[PASS] PPO: 512 steps, finite updated parameters, prediction + step", flush=True)
        print("ALL CHECKS PASSED; training device:", model.device, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
