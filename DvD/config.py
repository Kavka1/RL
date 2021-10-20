import argparse


def hyperparameters_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--env', type=str, default='')
    
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--noise_std', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--memory_size', type=int, default=1e5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--update_delay', type=int, default=4)

    parser.add_argument('--hidden_sizes', type=list, default=[256, 256])

    config = parser.parse_args()
    return config