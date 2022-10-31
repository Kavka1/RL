from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    # General.
    device: str = 'auto'
    logdir: str = './output/'
    comment: str = ''
    seed: Optional[int] = None
    deterministic: bool = False  # True if you getting OOM
    steps: int = int(1e6)
    eval_every: int = int(1e4)
    video_every: int = int(1e4)
    save_every: int = int(1e4)
    eval_episodes: int = 5
    log_every: int = int(1e3)
    log_scalars: bool = True
    log_images: bool = True
    # Environment.
    task: str = 'cartpole_balance'
    envs: int = 1
    action_repeat: int = 2
    time_limit: int = 1000
    prefill: int = 5000
    eval_noise: float = 0.0
    clip_rewards: str = 'none'
    # Model.
    deter_size: int = 200
    stoch_size: int = 30
    num_units: int = 400
    dense_act: str = 'elu'
    cnn_act: str = 'relu'
    cnn_depth: int = 32
    pcont: bool = False
    free_nats: float = 3.0
    kl_scale: float = 1.0
    pcont_scale: float = 10.0
    weight_decay: float = 0.0
    # Training.
    batch_size: int = 50
    batch_length: int = 50
    train_every: int = 1000
    train_steps: int = 100
    model_lr: float = 6e-4
    value_lr: float = 8e-5
    actor_lr: float = 8e-5
    grad_clip: float = 100.0
    dataset_balance: bool = False
    # Behavior.
    discount: float = 0.99
    disclam: float = 0.95
    horizon: int = 15
    action_dist: str = 'tanh_normal'
    action_init_std: float = 5.0
    expl: str = 'additive_gaussian'
    expl_amount: float = 0.3
    expl_decay: float = 0.0
    expl_min: float = 0.0
    # Ablations.
    update_horizon: Optional[int] = None  # policy value after this horizon are not updated
    single_step_q: bool = False  # Use 1-step target as an estimate of q.
    # Additional
    tf_init: bool = False