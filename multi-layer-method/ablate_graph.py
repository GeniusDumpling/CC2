"""Frozen-v1 2x2 input ablation: directed NACL graph and foothold suppression."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from environment import SubnetDefenseEnv
import numpy as np
import yaml

VARIANTS = ('baseline', 'nacl', 'suppress', 'nacl_suppress')


class GraphAblationEnv(SubnetDefenseEnv):
    def __init__(self, variant='baseline', **kwargs):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown graph ablation: {variant}')
        super().__init__(**kwargs)
        self.variant = variant
        self.policy_adjacency = self.adjacency.copy()
        self.policy_membership = self.membership.copy()
        self.suppressed = []
        if variant in ('nacl', 'nacl_suppress'):
            subnets = yaml.safe_load(self.scenario.read_text())['Subnets']

            def allows(source, destination, direction):
                rules = subnets[source].get('NACLs', {})
                value = rules.get(destination, rules.get('all', {})).get(direction, 'None')
                if value not in ('all', 'None', None):
                    raise ValueError('This ablation supports only all/None subnet rules')
                return value == 'all'

            for receiver in range(self.n_hosts):
                dst = self.subnets[int(self.membership[:, receiver].argmax())]
                for sender in range(self.n_hosts):
                    src = self.subnets[int(self.membership[:, sender].argmax())]
                    # MPNN aggregates column j into row i: adjacency[receiver, sender].
                    self.policy_adjacency[receiver, sender] = float(src == dst or (
                        allows(src, dst, 'out') and allows(dst, src, 'in')))
        if variant in ('suppress', 'nacl_suppress'):
            self.suppressed = [i for i, host in enumerate(self.hosts) if host in self.footholds]
            for h in self.suppressed:
                self.policy_adjacency[h, :] = 0
                self.policy_adjacency[:, h] = 0
                self.policy_adjacency[h, h] = 1
                self.policy_membership[:, h] = 0

    def transform_observation(self, obs):
        transformed = {k: v.copy() for k, v in obs.items()}
        transformed['adjacency'] = self.policy_adjacency.copy()
        transformed['membership'] = self.policy_membership.copy()
        # Also clear features: self-loop nodes still enter the unchanged global mean.
        # This removes dynamic foothold information, not its constant embedding/count.
        transformed['nodes'][self.suppressed] = 0
        return transformed

    def _observation(self, result):
        return self.transform_observation(super()._observation(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=153)
    args = parser.parse_args()
    if not args.model.is_file() or args.episodes < 1 or not 0 <= args.seed <= 2**32 - args.episodes:
        parser.error('Invalid checkpoint, episode count or seed')
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for variant in VARIANTS:
        command = [sys.executable, '-u', str(Path(__file__).with_name('evaluate.py')),
                   '--model', str(args.model.resolve()), '--output', str(args.output / variant),
                   '--episodes', str(args.episodes), '--seed', str(args.seed), '--ablation', variant]
        with (args.output / f'{variant}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        reports[variant] = json.loads((args.output / variant / 'evaluation.json').read_text())
        print(f'{variant}: total={reports[variant]["total_score"]:.4f}', flush=True)
    baseline = reports['baseline']
    comparison = {'scope': 'Frozen checkpoint input ablation; no retraining; paired reset seeds, '
                           'not identical subsequent random streams. No claim of architectural superiority.',
                  'episodes_per_configuration': args.episodes, 'results': []}
    for variant, report in reports.items():
        assert report['model_sha256'] == baseline['model_sha256']
        differences = []
        for current, original in zip(report['results'], baseline['results']):
            assert (current['horizon'], current['opponent']) == (original['horizon'], original['opponent'])
            delta = np.array(current['returns']) - np.array(original['returns'])
            differences.append({'horizon': current['horizon'], 'opponent': current['opponent'],
                                'mean_delta': float(delta.mean()),
                                'paired_standard_error': float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None})
        comparison['results'].append({'variant': variant, 'total_score': report['total_score'],
                                     'delta_vs_baseline': report['total_score'] - baseline['total_score'],
                                     'paired_differences': differences})
    (args.output / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
