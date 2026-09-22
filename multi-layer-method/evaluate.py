"""CAGE-2 score: sum of nine mean episode returns (3 horizons x 3 opponents)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from environment import SubnetDefenseEnv, RED_AGENTS, ACTION_NAMES
import torch
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Saved PPO .zip checkpoint')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--episodes', type=int, default=1000, help='Episodes per configuration')
    parser.add_argument('--seed', type=int, default=153)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--ablation', choices=('baseline', 'nacl', 'suppress', 'nacl_suppress'), default='baseline')
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.seed <= 2**32 - args.episodes:
        parser.error('episodes must be positive and episode seeds must fit uint32')
    if not args.model.is_file():
        parser.error(f'Checkpoint does not exist: {args.model}')
    torch.set_num_threads(1)
    model = PPO.load(args.model, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        'model': str(args.model.resolve()),
        'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'training_timesteps': model.num_timesteps,
        'episodes_per_configuration': args.episodes,
        'seed': args.seed, 'device': str(model.device),
        'seed_protocol': 'reset(seed=seed+episode_index), reused for each configuration',
        'decoding': 'deterministic joint argmax',
        'score_definition': 'sum of nine mean undiscounted episode returns',
        'scope': 'CAGE-2 scoring aggregation; five-action policy, not official validated ranking',
        'ablation': args.ablation,
        'results': [],
    }
    # Persist each completed group; a final total is written only after all nine finish.
    path = args.output / 'evaluation.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for horizon in (30, 50, 100):
        for opponent in RED_AGENTS:
            from ablate_graph import GraphAblationEnv
            env = GraphAblationEnv(variant=args.ablation, red_agent=opponent, max_steps=horizon)
            returns, counts = [], Counter({name: 0 for name in ACTION_NAMES})
            costs, state_rewards = [], []
            restore_zero = 0
            try:
                # Check checkpoint spaces against the actual adapter before evaluating.
                model.set_env(env)
                for episode in range(args.episodes):
                    obs, _ = env.reset(seed=args.seed + episode)
                    total = 0.0
                    cost_total = 0.0
                    for step in range(horizon):
                        action, _ = model.predict(obs, deterministic=True)
                        if int(action[2]) == 4 and not obs['nodes'][int(action[1])].any():
                            restore_zero += 1
                        obs, reward, terminated, truncated, info = env.step(action)
                        total += reward
                        cost_total += float(env.simulator.get_last_action('Blue').cost)
                        counts[info['action']] += 1
                        if terminated or (truncated and step + 1 != horizon):
                            raise RuntimeError('Unexpected early end; cannot score a full fixed-length episode')
                    returns.append(total)
                    costs.append(cost_total)
                    state_rewards.append(total - cost_total)
                    if (episode + 1) % 100 == 0:
                        print(f'{horizon} steps / {opponent}: {episode + 1}/{args.episodes}', flush=True)
            finally:
                env.close()
            result = {'horizon': horizon, 'opponent': opponent, 'returns': returns,
                      'mean': mean(returns), 'std': stdev(returns) if len(returns) > 1 else None,
                      'action_counts': dict(counts), 'mean_action_cost': mean(costs),
                      'mean_state_reward': mean(state_rewards), 'restore_zero_target': restore_zero}
            report['results'].append(result)
            path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            print(f'{horizon} steps / {opponent}: mean={result["mean"]:.4f}, std={result["std"]}', flush=True)
    report['total_score'] = sum(result['mean'] for result in report['results'])
    report['checkpoint_unchanged'] = hashlib.sha256(args.model.read_bytes()).hexdigest() == report['model_sha256']
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Total score (sum of 9 means): {report["total_score"]:.4f}\nSaved: {path}', flush=True)


if __name__ == '__main__':
    main()
