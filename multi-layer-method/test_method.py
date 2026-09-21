"""Runnable integration check: python test_method.py (no pytest required)."""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys

assert importlib.util.find_spec('method') is not None, 'Missing three-level policy implementation'
from environment import SubnetDefenseEnv
from method import SubnetPolicy
import numpy as np
import torch
from stable_baselines3 import PPO


def main():
    torch.set_num_threads(1)
    env = SubnetDefenseEnv(max_steps=8)
    try:
        obs, _ = env.reset(seed=7)
        model = PPO(SubnetPolicy, env, n_steps=16, batch_size=8, n_epochs=2,
                    policy_kwargs={'width': 16}, seed=7, device='cpu')
        policy = model.policy
        tensors, _ = policy.obs_to_tensor(obs)
        joint, values = policy.joint_distribution(tensors)
        assert torch.allclose(joint.sum((1, 2, 3)), torch.ones(1))
        legal = obs['membership'][:, :, None] * obs['action_mask'][None, :, :]
        assert (joint[0].detach().numpy()[legal == 0] == 0).all()
        # Enumerating the small action space provides an independent entropy check.
        nz = torch.nonzero(joint[0] > 0)
        batch = {k: v.expand(len(nz), *v.shape[1:]) for k, v in tensors.items()}
        _, logs, entropy = policy.evaluate_actions(batch, nz)
        probabilities = joint[0][tuple(nz.T)]
        assert torch.allclose(logs.exp(), probabilities, atol=1e-6)
        expected = -(probabilities * probabilities.log()).sum()
        assert torch.allclose(entropy, expected.expand_as(entropy), atol=1e-6)
        for _ in range(12):
            action, _ = model.predict(obs)
            assert legal[tuple(action)] == 1
        a, _ = model.predict(obs, deterministic=True)
        b, _ = model.predict(obs, deterministic=True)
        assert np.array_equal(a, b)
        # Deterministic decoding must be the joint argmax, not a per-level marginal argmax.
        joint, _ = policy.joint_distribution(tensors)
        expected = np.unravel_index(int(joint[0].argmax()), joint[0].shape)
        assert tuple(int(i) for i in a) == expected, 'Deterministic action is not the joint argmax'
        # Global Sleep has exactly one representation and consumes one step.
        sleep = np.array([env.n_subnets, env.n_hosts, 0])
        for i in range(8):
            _, _, terminated, truncated, _ = env.step(sleep)
            assert truncated == (i == 7) and not terminated
        env.reset(seed=7)
        bad = sleep.copy(); bad[1] = 0
        try:
            env.step(bad)
            raise AssertionError('Cross-subnet action accepted')
        except ValueError:
            pass
        before = [p.detach().clone() for p in policy.parameters()]
        model.learn(32)
        assert any(not torch.equal(x, y) for x, y in zip(before, policy.parameters()))
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        obs, _ = env.reset(seed=8)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp) / 'model')
            restored = PPO.load(Path(tmp) / 'model', env=env, device='cpu')
            a, _ = model.predict(obs, deterministic=True)
            b, _ = restored.predict(obs, deterministic=True)
            assert np.array_equal(a, b)
        for opponent in ('b_line', 'meander', 'sleep'):
            other = SubnetDefenseEnv(red_agent=opponent, max_steps=3)
            try:
                state, _ = other.reset(seed=11)
                for _ in range(3):
                    action, _ = model.predict(state)
                    state, reward, _, _, _ = other.step(action)
                    assert other.observation_space.contains(state) and np.isfinite(reward)
            finally:
                other.close()
        print('PASS: masks, joint probabilities/entropy, deterministic prediction, horizon, PPO update, save/load, 3 opponents')
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')),
                            '--steps', '256', '--horizon', '8', '--eval-episodes', '1',
                            '--output', str(output)], check=True, capture_output=True, text=True)
            events = list((output / 'tensorboard').rglob('events.out.tfevents.*'))
            assert events, 'Training did not create TensorBoard events'
            log = EventAccumulator(str(events[0].parent)).Reload()
            for tag in ('rollout/ep_rew_mean', 'rollout/ep_len_mean',
                        'train/value_loss', 'train/policy_gradient_loss',
                        'train/entropy_loss', 'train/approx_kl'):
                samples = log.Scalars(tag)
                assert samples and all(np.isfinite(x.value) for x in samples), tag
                assert samples[-1].step == 256, tag
        print('PASS: training CLI writes finite TensorBoard metrics through final update')
    finally:
        env.close()


if __name__ == '__main__':
    main()
