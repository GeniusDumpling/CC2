"""Train v2 from scratch with User0 information suppressed in Scenario2."""
from train import main as train_main


def main():
    train_main(default_ablation='suppress', default_run='v2')


if __name__ == '__main__':
    main()
