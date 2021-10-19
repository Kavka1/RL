import argparse


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gym_lichter:FreeAnt-v0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--N_iter', type=int, default=500)
    parser.add_argument('--goal_num_per_iter', type=int, default=300)
    parser.add_argument('--load_model_remark', type=str, default='gym_lichter:FreeAnt-v0_123_iter50')

    parser.add_argument('--GAN_update_iteration', type=int, default=200)
    parser.add_argument('--goal_coverage_scale', type=float, default=5)
    parser.add_argument('--initialize_goal_num', type=int, default=100)
    parser.add_argument('--noise_dim', type=int, default=4)
    parser.add_argument('--lr_G', type=float, default=5e-4)
    parser.add_argument('--lr_D', type=float, default=5e-4)
    parser.add_argument('--r_min', type=float, default=0.1)
    parser.add_argument('--r_max', type=float, default=0.7)

    parser.add_argument('--evaluate_episodes', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.998)
    parser.add_argument('--lamda', type=float, default=0.995)
    parser.add_argument('--lr_pi', type=float, default=1e-3)
    parser.add_argument('--lr_v', type=float, default=1e-3)
    parser.add_argument('--action_var', type=float, default=0.2)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--temperature_coef', type=float, default=0.05)
    parser.add_argument('--K_updates', type=int, default=100)

    parser.add_argument('--lr_pi_TD3', type=float, default=5e-4)
    parser.add_argument('--lr_q', type=float, default=5e-4)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--noise_std', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--policy_update_interval', type=int, default=2)
    parser.add_argument('--K_updates_TD3', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--memory_capacity', type=int, default=10000000)
    
    args = parser.parse_args()
    return args


