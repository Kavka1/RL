import argparse


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FetchPickAndPlace-v1')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--load_model_remark', type=str, default='FetchReach-v1_123_150')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2_coefficient', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--noise_eps', type=float, default=0.1)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--max_episode', type=int, default=100000)
    parser.add_argument('--save_model_interval', type=int, default=500)#500
    parser.add_argument('--save_model_start', type=int, default=25000)#25000
    parser.add_argument('--update_N', type=int, default=40)
    parser.add_argument('--evaluate_episode', type=int, default=20)
    parser.add_argument('--evaluate_interval', type=int, default=250)

    args = parser.parse_args()
    return args


