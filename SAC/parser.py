import argparse


def add_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load_model_remark', type=str, default='HalfCheetah-v2_123_1000')

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--max_episode', type=int, default=500000)
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--train_start_step', type=int, default=1000)
    parser.add_argument('--save_model_interval', type=int, default=1000)

    args = parser.parse_args()
    return args