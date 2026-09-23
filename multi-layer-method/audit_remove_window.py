"""Frozen v1 Meander audit: exploit privilege, visible alerts, isolated Remove probes."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from statistics import mean

from environment import SubnetDefenseEnv, ACTION_NAMES
from CybORG.Shared.Actions import Remove
import numpy as np
import torch
from stable_baselines3 import PPO


def sessions(state, host=None):
    return {str(k): {'host': s.host, 'user': s.username, 'pid': s.pid, 'active': s.active}
            for k, s in state.sessions['Red'].items() if host is None or s.host == host}


def main():
    root = Path(__file__).parent
    checkpoint = root / 'runs/v1-300k/model.zip'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = root / 'runs/v1-300k-remove-window'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    model = PPO.load(checkpoint, device='cpu')
    env = SubnetDefenseEnv(red_agent='meander', max_steps=50)
    controller = env.simulator.environment_controller
    execute_original = controller.execute_action
    red_events = []
    totals, new_counts, features, reactions, exploits = Counter(), Counter(), Counter(), Counter(), Counter()
    returns = []

    def execute(action):
        if getattr(action, 'agent', None) != 'Red':
            return execute_original(action)
        before = sessions(controller.state)
        result = execute_original(action)
        after = sessions(controller.state)
        red_events.append({'action': str(action), 'type': type(action).__name__,
                           'subaction': type(getattr(action, 'sub_action', None)).__name__,
                           'success': str(result.success),
                           'new': {k: v for k, v in after.items() if k not in before},
                           'escalated': [k for k, v in after.items() if k in before and
                               before[k]['user'] not in ('root', 'SYSTEM') and v['user'] in ('root', 'SYSTEM')]})
        return result

    controller.execute_action = execute
    try:
        with (output / 'steps.jsonl').open('w', encoding='utf-8') as stream:
            for episode in range(100):
                obs, _ = env.reset(seed=10000 + episode)
                total = 0.0
                for step in range(1, 51):
                    action, _ = model.predict(obs, deterministic=True)
                    # Ground truth is used only after action selection for diagnostics.
                    state = controller.state
                    candidates = []
                    for h, host in enumerate(env.hosts):
                        if not obs['action_mask'][h, 3]:
                            continue
                        host_sessions = sessions(state, host)
                        if not host_sessions:
                            continue
                        pids = state.sessions['Blue'][0].sus_pids.get(host, [])
                        ordinary = {k: s for k, s in host_sessions.items() if s['user'] not in ('root', 'SYSTEM')}
                        removable = {k: s for k, s in ordinary.items() if s['pid'] in pids and
                            state.hosts[host].get_process(s['pid']) is not None and
                            state.hosts[host].get_process(s['pid']).user not in ('root', 'SYSTEM')}
                        totals['compromised_host_decisions'] += 1
                        totals['ordinary_host_decisions'] += bool(ordinary)
                        totals['ordinary_without_matching_pid'] += bool(ordinary) and not removable
                        if not removable:
                            continue
                        # Execute only in a copy; restore RNG consumed by Remove's choice().
                        py_rng, np_rng = random.getstate(), np.random.get_state()
                        clone = deepcopy(state)
                        try:
                            Remove(session=0, agent='Blue', hostname=host).sim_execute(clone)
                        finally:
                            random.setstate(py_rng)
                            np.random.set_state(np_rng)
                        removed = set(host_sessions) - set(sessions(clone, host))
                        assert removed, 'Predicted removable session survived isolated Remove'
                        totals['removable_host_decisions'] += 1
                        selected = int(action[1]) == h
                        reaction = ACTION_NAMES[int(action[2])] if selected else 'other_host_or_global'
                        reactions[reaction] += 1
                        key = str(obs['nodes'][h].astype(int).tolist())
                        features[key] += 1
                        candidates.append({'host': host, 'features': obs['nodes'][h].tolist(),
                            'sessions': host_sessions, 'sus_pids': list(pids),
                            'probe_removed': sorted(removed), 'policy_reaction': reaction})
                    totals['steps_with_removable_target'] += bool(candidates)
                    red_events.clear()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    assert len(red_events) == 1
                    red = red_events[0]
                    totals['actual_escalated_sessions'] += len(red['escalated'])
                    for sid, s in red['new'].items():
                        if s['host'] in env.footholds:
                            continue
                        host = s['host']; h = env.hosts.index(host)
                        privileged = s['user'] in ('root', 'SYSTEM')
                        category = 'privileged' if privileged else 'ordinary'
                        new_counts[category] += 1
                        new_counts[category + '_pid_known'] += s['pid'] in controller.state.sessions['Blue'][0].sus_pids.get(host, [])
                        new_counts[category + '_exploit_alert'] += bool(obs['nodes'][h, 1])
                        exploits[red['subaction'] + ':' + s['user']] += 1
                        s['blue_features_after'] = obs['nodes'][h].tolist()
                        s['pid_known'] = s['pid'] in controller.state.sessions['Blue'][0].sus_pids.get(host, [])
                    stream.write(json.dumps({'seed': 10000 + episode, 'step': step, **info,
                        'candidates': candidates, 'red': red, 'reward': reward}) + '\n')
                    assert not terminated and truncated == (step == 50)
                returns.append(total)
                if (episode + 1) % 25 == 0:
                    print(episode + 1, 'episodes; windows', totals['removable_host_decisions'], flush=True)
    finally:
        controller.execute_action = execute_original
        env.close()
    previous = json.loads((root / 'runs/v1-300k-action-audit/summary.json').read_text())
    assert np.allclose(returns, previous['results']['baseline']['returns'], rtol=0, atol=1e-10)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    report = {'episodes': 100, 'horizon': 50, 'seed': 10000, 'mean_reward': mean(returns),
        'sha256': digest, 'baseline_100_returns_reproduced': True, 'checkpoint_unchanged': True,
        'totals': dict(totals), 'new_sessions': dict(new_counts), 'exploit_users': dict(exploits),
        'window_observations': dict(features), 'policy_reactions': dict(reactions),
        'notes': 'Host-decision counts are repeated opportunities, not unique infections. '
                 'Ground truth and cloned-state probes are audit only. No changed deployment or training.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
