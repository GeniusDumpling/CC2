"""Check v2 wiring without starting training: python test_v2.py."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import train


def main():
    assert Path(__file__).with_name('train_v2.py').is_file(), 'Missing v2 entry point'
    import train_v2

    class TrainingIntercepted(Exception):
        pass

    def inspect_model(policy, env, **kwargs):
        raw = env.unwrapped
        assert raw.variant == 'suppress'
        obs, _ = env.reset(seed=0)
        h = raw.hosts.index('User0')
        assert not obs['nodes'][h].any()
        assert not obs['membership'][:, h].any()
        assert obs['adjacency'][h].sum() == obs['adjacency'][:, h].sum() == 1
        assert not obs['action_mask'][h].any()
        assert raw.max_steps == 50
        assert kwargs['seed'] == 0 and kwargs['n_steps'] == 128
        raise TrainingIntercepted

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / 'v2'
        with patch.object(sys, 'argv', ['train_v2.py', '--output', str(output)]), \
                patch.object(train, 'PPO', side_effect=inspect_model):
            try:
                train_v2.main()
            except TrainingIntercepted:
                pass
            else:
                raise AssertionError('Training environment was not inspected')
        config = json.loads((output / 'config.json').read_text())
        assert config['ablation'] == 'suppress' and config['load'] is None
        assert config['steps'] == 100000 and config['opponent'] == 'meander'
        assert not (output / 'model.zip').exists()

        # Exercise post-training evaluation with a stand-in predictor; forbid learning.
        class EvaluationOnly:
            num_timesteps = 0

            def predict(self, obs, deterministic):
                assert deterministic
                assert (obs['membership'].sum(axis=0) == 0).sum() == 1
                return np.array([obs['membership'].shape[0] - 1,
                                 obs['nodes'].shape[0] - 1, 0]), None

        evaluation = Path(tmp) / 'evaluation'
        with patch.object(sys, 'argv', ['train_v2.py', '--load', 'unused.zip',
                '--output', str(evaluation), '--horizon', '3', '--eval-episodes', '1']), \
                patch.object(train.PPO, 'load', return_value=EvaluationOnly()):
            train_v2.main()
        report = json.loads((evaluation / 'evaluation.json').read_text())
        assert report['ablation'] == 'suppress'
        assert len(report['opponents']) == 3
    print('PASS: v2 training/evaluation use suppression; no training was executed')


if __name__ == '__main__':
    main()
