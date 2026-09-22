"""Read-only checkpoint diagnosis: policy probabilities, observations and reward costs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from environment import SubnetDefenseEnv, RED_AGENTS, ACTION_NAMES
import numpy as np
import torch
from stable_baselines3 import PPO


def probe_observations(model, output):
    """Input sensitivity only: these modified observations are not environment rollouts."""
    report = json.loads((output / 'summary.json').read_text())
    examples = {}
    with (output / 'steps.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['mode'] == 'deterministic' and row['action'] == 'Restore':
                key = (row['horizon'], row['opponent'], tuple(row['target_features_before']))
                examples.setdefault(key, row)
    results = []
    # User0 is the fixed initial foothold in the evaluated Scenario2 (not actionable).
    foothold = report['hosts'].index('User0')
    for row in examples.values():
        variants = {}
        for variant in ('original', 'clear_foothold', 'clear_selected_host', 'clear_all',
                        'disconnect_foothold', 'exclude_foothold_from_pool', 'exclude_foothold_from_both'):
            nodes = np.array(row['nodes_before'], dtype=np.float32)
            if variant == 'clear_foothold':
                nodes[foothold] = 0
            elif variant == 'clear_selected_host':
                nodes[row['indices'][1]] = 0
            elif variant == 'clear_all':
                nodes[:] = 0
            obs = {k: np.array(v, dtype=np.float32) for k, v in report['static_observation'].items()}
            obs.update(nodes=nodes, action_mask=np.array(row['action_mask'], dtype=np.float32))
            if variant in ('disconnect_foothold', 'exclude_foothold_from_both'):
                obs['adjacency'][foothold, :] = 0
                obs['adjacency'][:, foothold] = 0
                obs['adjacency'][foothold, foothold] = 1
            if variant in ('exclude_foothold_from_pool', 'exclude_foothold_from_both'):
                obs['membership'][:, foothold] = 0
            with torch.no_grad():
                joint = model.policy.joint_distribution(model.policy.obs_to_tensor(obs)[0])[0][0].cpu().numpy()
            z, h, a = np.unravel_index(joint.argmax(), joint.shape)
            selected_z, selected_h, _ = row['indices']
            variants[variant] = {'host': report['hosts'][h], 'action': ACTION_NAMES[a],
                                 'best_joint_per_action': joint.max(axis=(0, 1)).tolist(),
                                 'subnet_probabilities': joint.sum(axis=(1, 2)).tolist(),
                                 'original_target_conditional_actions':
                                 (joint[selected_z, selected_h] / joint[selected_z, selected_h].sum()).tolist()}
        results.append({'horizon': row['horizon'], 'opponent': row['opponent'], 'episode': row['episode'],
                        'step': row['step'], 'host': row['host'],
                        'target_features': row['target_features_before'],
                        'nonzero_hosts': {report['hosts'][h]: features for h, features in enumerate(row['nodes_before']) if any(features)},
                        'variants': variants})
    (output / 'observation_probes.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--horizons', type=int, nargs='+', default=[50, 100])
    parser.add_argument('--seed', type=int, default=153)
    args = parser.parse_args()
    if args.episodes < 1 or min(args.horizons) < 1 or not 0 <= args.seed <= 2**32 - args.episodes:
        parser.error('Invalid episode count, horizon or seed')
    torch.set_num_threads(1)
    model = PPO.load(args.model, device='cpu')
    model.policy.set_training_mode(False)
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'model': str(args.model.resolve()), 'sha256': digest, 'episodes': args.episodes,
              'seed': args.seed, 'device': 'cpu', 'results': [],
              'notes': 'Same reset seeds across modes; different actions can change environment RNG consumption. '
                       'Hidden session audit is logged only, never supplied to the policy. '
                       'Activity/compromise labels are BlueTable observations, not ground truth.'}
    with (args.output / 'steps.jsonl').open('w', encoding='utf-8') as stream:
        for horizon in args.horizons:
            for opponent in RED_AGENTS:
                for mode in ('deterministic', 'stochastic'):
                    env = SubnetDefenseEnv(red_agent=opponent, max_steps=horizon)
                    model.set_env(env)
                    report['hosts'] = env.hosts + ['global']
                    report['subnets'] = env.subnets + ['global']
                    report['action_names'] = ACTION_NAMES
                    report['static_observation'] = {'adjacency': env.adjacency.tolist(),
                                                    'membership': env.membership.tolist()}
                    counts, restore_stats, targets = Counter(), Counter(), Counter()
                    returns, costs, states, restore_mass = [], [], [], []
                    try:
                        for episode in range(args.episodes):
                            obs, _ = env.reset(seed=args.seed + episode)
                            torch.manual_seed(args.seed + episode)
                            last_restore = {}
                            total = cost_total = state_total = 0.0
                            for step in range(horizon):
                                tensors, _ = model.policy.obs_to_tensor(obs)
                                with torch.no_grad():
                                    joint = model.policy.joint_distribution(tensors)[0][0].cpu().numpy()
                                    action, _ = model.predict(obs, deterministic=mode == 'deterministic')
                                z, h, a = map(int, action)
                                node = obs['nodes'][h].copy()
                                gap = step - last_restore[h] if h in last_restore else None
                                host_mass = float(joint[z, h].sum())
                                subnet_mass = float(joint[z].sum())
                                conditional = joint[z, h] / host_mass
                                marginal = joint.sum(axis=(0, 1))
                                # Audit only, after action selection. Includes inactive sessions explicitly.
                                sessions = [{'host': s.host, 'username': s.username, 'active': s.active}
                                            for s in env.simulator.environment_controller.state.sessions['Red'].values()]
                                host = env.hosts[h] if h < env.n_hosts else 'global'
                                before = obs
                                obs, reward, terminated, truncated, info = env.step(action)
                                cost = float(env.simulator.get_last_action('Blue').cost)
                                state_reward = reward - cost  # Matches EnvironmentController reward + cost.
                                counts[info['action']] += 1
                                restore_mass.append(float(marginal[4]))
                                if a == 4:
                                    targets[host] += 1
                                    restore_stats['total'] += 1
                                    restore_stats['target_no_activity'] += int(not node[:2].any())
                                    restore_stats['target_no_access_evidence'] += int(not node[2:].any())
                                    restore_stats['target_all_zero'] += int(not node.any())
                                    restore_stats['same_host_previous_step'] += int(gap == 1)
                                    restore_stats['same_host_within_5_steps'] += int(gap is not None and gap <= 5)
                                    restore_stats['no_red_session_before'] += int(not any(s['host'] == host for s in sessions))
                                    restore_stats['post_target_all_zero'] += int(not obs['nodes'][h].any())
                                    last_restore[h] = step
                                record = {'horizon': horizon, 'opponent': opponent, 'mode': mode,
                                          'episode': episode, 'seed': args.seed + episode, 'step': step + 1,
                                          'indices': [z, h, a], 'host': host, **info,
                                          'nodes_before': before['nodes'].tolist(), 'nodes_after': obs['nodes'].tolist(),
                                          'action_mask': before['action_mask'].tolist(),
                                          'target_features_before': node.tolist(), 'steps_since_restore': gap,
                                          'subnet_probability': subnet_mass,
                                          'host_conditional_probability': host_mass / subnet_mass,
                                          'conditional_actions': conditional.tolist(),
                                          'action_marginal': marginal.tolist(),
                                          'best_joint_per_action': joint.max(axis=(0, 1)).tolist(),
                                          'joint_probability': float(joint[z, h, a]),
                                          'max_joint_probability': float(joint.max()),
                                          'joint_ties': int((joint == joint.max()).sum()),
                                          'reward': reward, 'action_cost': cost, 'state_reward': state_reward,
                                          'audit_red_sessions_before': sessions}
                                stream.write(json.dumps(record, allow_nan=False) + '\n')
                                total += reward; cost_total += cost; state_total += state_reward
                                if terminated or truncated:
                                    break
                            returns.append(total); costs.append(cost_total); states.append(state_total)
                    finally:
                        env.close()
                    result = {'horizon': horizon, 'opponent': opponent, 'mode': mode, 'returns': returns,
                              'mean': mean(returns), 'std': stdev(returns) if len(returns) > 1 else None,
                              'mean_action_cost': mean(costs), 'mean_state_reward': mean(states),
                              'action_counts': dict(counts), 'restore_stats': dict(restore_stats),
                              'restore_hosts': dict(targets), 'mean_restore_probability': mean(restore_mass)}
                    report['results'].append(result)
                    stream.flush()
                    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                    print(f'{horizon} / {opponent} / {mode}: mean={result["mean"]:.3f}, actions={dict(counts)}', flush=True)
    report['checkpoint_unchanged'] = hashlib.sha256(args.model.read_bytes()).hexdigest() == digest
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    probe_observations(model, args.output)


if __name__ == '__main__':
    main()
