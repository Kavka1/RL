import argparse


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load_model_remark', type=str, default='10500')
    parser.add_argument('--log_env_idx', type=int, default=0)

    parser.add_argument('--num_worker', type=int, default=16)

    parser.add_argument('--save_model_interval', type=int, default=500)
    parser.add_argument('--max_epoch', type=int, default=100000)
    parser.add_argument('--max_episode_step', type=int, default=10000)
    parser.add_argument('--initialize_episode', type=int, default=1)
    parser.add_argument('--update_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rollout_len', type=int, default=256)

    parser.add_argument('--r_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma_e', type=float, default=0.999)
    parser.add_argument('--gamma_i', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.95)
    parser.add_argument('--entropy_coef', type=float, default=0.001)
    parser.add_argument('--ex_coef', type=int, default=2)
    parser.add_argument('--in_coef', type=int, default=1)
    parser.add_argument('--clip_eps', type=float, default=0.15)
    parser.add_argument('--obs_clip', type=float, default=5)
    parser.add_argument('--update_proportion', type=float, default=0.25)

    args = parser.parse_args()
    return args