import argparse
from ast import parse

def add_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--lr_a', type=float, default=4e-4)
    parser.add_argument('--lr_c', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--noise_eps', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--memory_size', type=int, default=1000000)

    parser.add_argument('--population_K', type=int, default=10)
    parser.add_argument('--evaluate_episode', type=int, default=1)
    parser.add_argument('--elite_frac', type=float, default=0.1)
    parser.add_argument('--mut_prob', type=float, default=0.9)
    parser.add_argument('--mut_frac', type=float, default=0.1)
    parser.add_argument('--super_mut_prob', type=float, default=0.05)
    parser.add_argument('--reset_prob', type=float, default=0.05)
    parser.add_argument('--mut_strength', type=float, default=0.1)

    parser.add_argument('--sync_interval', type=int, default=10)
    parser.add_argument('--generation_episode', type=int, default=100000)

    #SAC parameter
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)

    args = parser.parse_args()
    return args