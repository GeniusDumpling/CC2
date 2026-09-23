"""Train once, save, then evaluate on all three CAGE-2 opponents."""
import argparse
import json
from pathlib import Path

from environment import RED_AGENTS
from ablate_graph import GraphAblationEnv, VARIANTS
from method import SubnetPolicy
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def main(default_ablation='baseline', default_run='v1'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--opponent', choices=RED_AGENTS, default='meander')
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'runs' / default_run)
    parser.add_argument('--ablation', choices=VARIANTS, default=default_ablation,
                        help='Observation transform used for both training and evaluation')
    parser.add_argument('--load', type=Path, help='Evaluate an existing checkpoint without training')
    args = parser.parse_args()
    if args.steps < 1 or args.horizon < 1 or args.eval_episodes < 1:
        parser.error('steps, horizon and eval-episodes must be positive')
    torch.set_num_threads(1)
    # Refuse to overwrite an earlier experiment accidentally.
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    env = Monitor(GraphAblationEnv(variant=args.ablation, red_agent=args.opponent, max_steps=args.horizon),
                  filename=str(args.output / 'train'))
    try:
        if args.load:
            model = PPO.load(args.load, env=env, device=args.device)
        else:
            model = PPO(SubnetPolicy, env, n_steps=128, batch_size=64, n_epochs=4,
                        learning_rate=3e-4, ent_coef=0.01, seed=args.seed,
                        tensorboard_log=str(args.output / 'tensorboard'),
                        device=args.device, verbose=1)
            model.learn(args.steps)
            # PPO normally dumps before updating; also write the final update's metrics.
            model.logger.dump(step=model.num_timesteps)
            model.logger.close()
            model.save(args.output / 'model')
        results = {}
        for opponent in RED_AGENTS:
            evaluation = GraphAblationEnv(variant=args.ablation, red_agent=opponent, max_steps=args.horizon)
            returns, counts = [], {}
            try:
                for episode in range(args.eval_episodes):
                    obs, _ = evaluation.reset(seed=args.seed + 10_000 + episode)
                    total = 0.0
                    for _ in range(args.horizon):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = evaluation.step(action)
                        total += reward
                        counts[info['action']] = counts.get(info['action'], 0) + 1
                        if terminated or truncated:
                            break
                    returns.append(total)
            finally:
                evaluation.close()
            results[opponent] = {'returns': returns, 'mean': float(np.mean(returns)),
                                 'std': float(np.std(returns)), 'action_counts': counts}
        report = {'timesteps': model.num_timesteps, 'deterministic': True,
                  'ablation': args.ablation, 'opponents': results}
        (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
