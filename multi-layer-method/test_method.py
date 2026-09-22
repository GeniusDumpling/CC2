"""Runnable integration check: python test_method.py (no pytest required)."""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import json

assert importlib.util.find_spec('method') is not None, 'Missing three-level policy implementation'
from environment import SubnetDefenseEnv
from method import SubnetPolicy
import numpy as np
import torch
from stable_baselines3 import PPO


def main():
    torch.set_num_threads(1)
    from ablate_graph import GraphAblationEnv, VARIANTS
    reference = SubnetDefenseEnv(max_steps=3)
    try:
        initial, _ = reference.reset(seed=7)
        for variant in VARIANTS:
            candidate = GraphAblationEnv(variant=variant, max_steps=3)
            try:
                observation, _ = candidate.reset(seed=7)
                assert np.array_equal(observation['action_mask'], initial['action_mask'])
                assert candidate.observation_space.contains(observation)
                u, op = candidate.hosts.index('User1'), candidate.hosts.index('Op_Server0')
                if variant in ('nacl', 'nacl_suppress'):
                    assert observation['adjacency'][op, u] == 0
                    assert observation['adjacency'][u, op] == 1
                if variant in ('suppress', 'nacl_suppress'):
                    h = candidate.hosts.index('User0')
                    assert observation['membership'][:, h].sum() == 0
                    assert observation['adjacency'][h].sum() == 1
                    assert observation['adjacency'][:, h].sum() == 1
                    changed = {k: v.copy() for k, v in initial.items()}
                    changed['nodes'][h] = 1
                    transformed = candidate.transform_observation(changed)
                    assert not transformed['nodes'][h].any()
                    assert changed['nodes'][h].all(), 'Transform mutated source observation'
                if variant == 'baseline':
                    assert all(np.array_equal(observation[k], initial[k]) for k in initial)
                reference.reset(seed=7)
                for _ in range(3):
                    action = np.array([candidate.n_subnets, candidate.n_hosts, 0])
                    _, reward, terminated, truncated, _ = candidate.step(action)
                    _, expected, end, cutoff, _ = reference.step(action)
                    assert (reward, terminated, truncated) == (expected, end, cutoff)
            finally:
                candidate.close()
    finally:
        reference.close()
    print('PASS: ablation edge direction, masks, suppression, identity and reward invariants')
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
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'model.zip'
            model.save(checkpoint)
            output = Path(tmp) / 'evaluation'
            subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate.py')),
                            '--model', str(checkpoint), '--episodes', '1',
                            '--output', str(output)], check=True, capture_output=True, text=True)
            report = json.loads((output / 'evaluation.json').read_text())
            assert len(report['results']) == 9
            assert {(r['horizon'], r['opponent']) for r in report['results']} == {
                (h, opponent) for h in (30, 50, 100) for opponent in ('b_line', 'meander', 'sleep')}
            for result in report['results']:
                assert result['mean'] == result['returns'][0]
                assert sum(result['action_counts'].values()) == result['horizon']
                assert result['std'] is None
            assert np.isclose(report['total_score'], sum(r['mean'] for r in report['results']))
        print('PASS: standalone evaluation covers 9 configurations and sums their means')
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'model.zip'
            model.save(checkpoint)
            output = Path(tmp) / 'diagnosis'
            subprocess.run([sys.executable, str(Path(__file__).with_name('diagnose_restore.py')),
                            '--model', str(checkpoint), '--episodes', '1', '--horizons', '8',
                            '--output', str(output)], check=True)
            report = json.loads((output / 'summary.json').read_text())
            records = [json.loads(line) for line in (output / 'steps.jsonl').read_text().splitlines()]
            assert len(report['results']) == 6 and len(records) == 48
            for record in records:
                assert np.isclose(record['reward'], record['action_cost'] + record['state_reward'])
                assert np.isclose(sum(record['action_marginal']), 1)
                assert np.isclose(sum(record['conditional_actions']), 1)
                assert np.isclose(record['joint_probability'], record['subnet_probability'] *
                                  record['host_conditional_probability'] * record['conditional_actions'][record['indices'][2]])
                if record['mode'] == 'deterministic':
                    assert np.isclose(record['joint_probability'], record['max_joint_probability'])
            assert report['checkpoint_unchanged']
        print('PASS: diagnostic traces, probability factorization, reward split, checkpoint integrity')
    finally:
        env.close()


if __name__ == '__main__':
    main()
