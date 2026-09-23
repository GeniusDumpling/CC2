"""Audit frozen v1-300k against Meander; hidden state never enters the policy."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

from environment import SubnetDefenseEnv, ACTION_NAMES
import numpy as np
import torch
from stable_baselines3 import PPO


def snapshot(controller, host):
    state = controller.state
    target = state.hosts[host]
    return {
        'sessions': {str(k): {'user': s.username, 'active': s.active, 'pid': s.pid}
                     for k, s in state.sessions['Red'].items() if s.host == host},
        'processes': sorted((p.pid, p.user, str(p.name)) for p in target.processes),
        'services': repr(target.services),
        'files': repr([f.get_state() for f in target.files]),
    }


def main():
    root = Path(__file__).parent
    checkpoint = root / 'runs/v1-300k/model.zip'
    output = root / 'runs/v1-300k-action-audit'
    output.mkdir(exist_ok=False)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    model = PPO.load(checkpoint, device='cpu')
    report = {'model_sha256': digest, 'episodes': 100, 'horizon': 50, 'seed': 10000,
              'opponent': 'meander', 'decoding': 'joint argmax', 'results': {},
              'scope': 'Immediate host-state audit plus whole-policy Remove-to-Sleep intervention. '
                       'Same reset seeds, not identical later random streams. '
                       'No state change is not proof of counterfactual uselessness.'}
    with (output / 'steps.jsonl').open('w', encoding='utf-8') as stream:
        for mode in ('baseline', 'remove_to_sleep'):
            env = SubnetDefenseEnv(red_agent='meander', max_steps=50)
            controller = env.simulator.environment_controller
            original_execute = controller.execute_action
            captured = []

            def execute(command):
                name = type(command).__name__
                if name not in ('Restore', 'Remove') or command.agent != 'Blue':
                    return original_execute(command)
                before = snapshot(controller, command.hostname)
                result = original_execute(command)
                after = snapshot(controller, command.hostname)
                captured.append({'before': before, 'after_blue': after,
                                 'success': str(result.success)})
                return result

            controller.execute_action = execute
            counts, targets = Counter(), Counter()
            stats = {name: Counter() for name in ('Restore', 'Remove')}
            returns, costs = [], []
            try:
                for episode in range(100):
                    obs, _ = env.reset(seed=10000 + episode)
                    total = cost_total = 0.0
                    pending = {}
                    for step in range(1, 51):
                        action, _ = model.predict(obs, deterministic=True)
                        proposed = ACTION_NAMES[int(action[2])]
                        if mode == 'remove_to_sleep' and proposed == 'Remove':
                            action = np.array([env.n_subnets, env.n_hosts, 0])
                        host = env.hosts[int(action[1])] if int(action[1]) < env.n_hosts else None
                        node = obs['nodes'][int(action[1])].tolist()
                        captured.clear()
                        obs, reward, terminated, truncated, info = env.step(action)
                        cost = float(env.simulator.get_last_action('Blue').cost)
                        total += reward
                        cost_total += cost
                        counts[info['action']] += 1
                        row = {'mode': mode, 'episode': episode, 'seed': 10000 + episode,
                               'step': step, **info, 'proposed': proposed, 'features': node,
                               'red_action': str(env.simulator.get_last_action('Red')),
                               'reward': reward, 'cost': cost}
                        if info['action'] in stats:
                            assert len(captured) == 1
                            row.update(captured[0])
                            row['after_round'] = snapshot(controller, host)
                            stat = stats[info['action']]
                            before, after = row['before'], row['after_blue']
                            removed = set(before['sessions']) - set(after['sessions'])
                            stat['total'] += 1
                            stat['has_red_before'] += bool(before['sessions'])
                            stat['has_active_red_before'] += any(s['active'] for s in before['sessions'].values())
                            stat['removed_any_session'] += bool(removed)
                            stat['removed_session_count'] += len(removed)
                            cleared = bool(before['sessions']) and not after['sessions']
                            stat['cleared_all_sessions'] += cleared
                            stat['no_red_before'] += not before['sessions']
                            stat['no_tracked_host_change'] += before == after
                            stat['processes_changed'] += before['processes'] != after['processes']
                            stat['privileged_red_before'] += any(s['user'] in ('root', 'SYSTEM') for s in before['sessions'].values())
                            stat['same_round_reinfection'] += cleared and bool(row['after_round']['sessions'])
                            targets[f'{info["action"]}:{host}'] += 1
                            if cleared:
                                pending[host] = (step, info['action'])
                        for target, (cleared_step, name) in list(pending.items()):
                            if any(s.host == target for s in controller.state.sessions['Red'].values()):
                                stats[name][f'reinfection_gap_{step-cleared_step}'] += 1
                                del pending[target]
                        stream.write(json.dumps(row, default=str) + '\n')
                        assert not terminated and truncated == (step == 50)
                    for _, name in pending.values():
                        stats[name]['not_reinfected_before_episode_end'] += 1
                    returns.append(total)
                    costs.append(cost_total)
                    if (episode + 1) % 25 == 0:
                        print(mode, episode + 1, 'mean', mean(returns), flush=True)
            finally:
                controller.execute_action = original_execute
                env.close()
            report['results'][mode] = {'returns': returns, 'mean': mean(returns),
                'mean_cost': mean(costs), 'mean_state_reward': mean(returns) - mean(costs),
                'actions': dict(counts), 'targets': dict(targets),
                'audit': {k: dict(v) for k, v in stats.items()}}
            if mode == 'baseline':
                previous = json.loads((root / 'runs/v1-300k/evaluation.json').read_text())
                expected = previous['opponents']['meander']['returns']
                assert np.allclose(returns[:10], expected, rtol=0, atol=1e-10), 'Audit changed baseline trajectory'
            (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    report['checkpoint_unchanged'] = True
    report['baseline_first10_reproduced'] = True
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'returns'}
                      for k, v in report['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
