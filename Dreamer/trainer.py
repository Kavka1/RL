from typing     import List, Dict, Optional, Tuple, Union
from pathlib    import Path
from termcolor  import colored
from omegaconf  import OmegaConf

import argparse
import json
import time
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from RL.Dreamer.config       import Config
from RL.Dreamer.buffer       import ReplayBuffer
from RL.Dreamer.utils        import set_seed, Timer
from RL.Dreamer.env          import make_env
from RL.Dreamer.dreamer      import Dreamer


class Trainer:
    def __init__(self, config: Config) -> None:
        self.c = config
        self.setup()

    def setup(self) -> None:
        set_seed(self.c.seed, self.c.deterministic)
        # log
        name = self.c.task
        if self.c.comment:
            name = f'{name}-{self.c.comment}'
        name = name + '-' + time.strftime('%Y-%m-%d_%H-%M-%S')

        self.logdir = Path(self.c.logdir) / name
        print('Logdir', self.logdir)
        self.video_dir = self.logdir / 'video'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.logdir/'tb'))

        # Create environments.
        print('Creating environment...')
        self.train_env = make_env(task=self.c.task, action_repeat=self.c.action_repeat, timelimit=self.c.time_limit)
        self.test_env = make_env(task=self.c.task, action_repeat=self.c.action_repeat, timelimit=self.c.time_limit)
        
        # Replay
        self.replay_buffer = ReplayBuffer(action_space=self.train_env.action_space, balance=self.c.dataset_balance)

        # Device
        if self.c.device == 'auto':
            self.c.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Agent
        print('Creating agent...')
        self.agent = Dreamer(self.c, self.train_env.action_space, self.writer, self.logdir).to(self.c.device)

    def train(self) -> None:
        self.global_steps   = 0
        self.episodes       = 0

        obs = self.train_env.reset()
        self.replay_buffer.start_episode(obs)
        agent_state = None
        self.timer  = Timer()
        self.last_frames = 0

        # Initialize evaluation
        self.eval()
        print('Start training...')
        while self.global_frames < self.c.steps:
            # get action
            if self.global_frames < self.c.prefill:
                action = self.train_env.action_space.sample()
            else:
                action, agent_state = self.agent.get_action(obs, agent_state, training=True)
            # step
            obs, reward, done, info = self.train_env.step(action)
            self.replay_buffer.add(obs, action, reward, done, info)
            self.global_steps += 1

            # End of training episode logging
            if done:
                assert 'episode' in info
                self.episodes += 1
                self.log(float(info['episode']['r']), float(info['episode']['l']), prefix='train')
                # Reset
                obs = self.train_env.reset()
                self.replay_buffer.start_episode(obs)
                agent_state = None
            
            # Training
            if self.global_frames > self.c.prefill and self.global_frames % self.c.train_every == 0:
                for train_step in range(self.c.train_steps):
                    log_images = self.c.log_images and self.global_frames % self.c.video_every == 0 and train_step == 0
                    self.agent.update(self.replay_buffer, log_images=log_images, video_path=self.video_dir / 'model' / f'{self.global_frames_str}.gif')
                if self.global_frames % self.c.log_every == 0:
                    self.agent.write_log(self.global_frames)

            # Evaluation
            if self.global_frames % self.c.eval_every == 0:
                self.eval()
            
            # Saving
            if self.global_frames % self.c.save_every == 0:
                self.agent.save(self.logdir / 'checkpoint.pth')
                with (self.logdir / 'checkpoint_step.txt').open('w') as f:
                    print(self.global_frames, file=f)

    def eval(self) -> None:
        print("Start evaluation")
        video_path = self.video_dir / 'interaction' / f'{self.global_frames_str}.mp4'
        video_path.parent.mkdir(exist_ok=True)
        returns = []
        lengths = []
        for i in range(self.c.eval_episodes):
            record_video = (i == 0)
            if record_video:
                self.test_env.start_recording()

            obs     = self.test_env.reset()
            done    = False
            agent_state = None
            while not done:
                action, agent_state = self.agent.get_action(obs, agent_state, training=False)
                obs, reward, done, info = self.test_env.step(action)
                if done:
                    assert 'episode' in info
            returns.append(info['episode']['r'])
            lengths.append(info['episode']['l'])

            if record_video:
                self.test_env.end_and_save(path=video_path)
        
        avg_return = float(np.mean(returns))
        avg_length = float(np.mean(lengths))
        # Eval logging
        self.log(avg_return, avg_length, prefix='test')
        self.writer.flush()

    def log(self, avg_return: float, avg_length: float, prefix: str) -> None:
        colored_prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        elapsed_time = self.timer.split()
        total_time = datetime.timedelta(seconds=int(self.timer.total()))
        fps = (self.global_frames - self.last_frames) / elapsed_time
        self.last_frames = self.global_frames
        print(f'{colored_prefix:<14} | F: {self.global_frames} | E: {self.episodes} | R: {avg_return:.2f} | L: {avg_length:.2f} | FPS: {fps:.2f} | T: {total_time}')
        metrics = [
            (f'{prefix}/return', avg_return),
            (f'{prefix}/length', avg_length),
            (f'{prefix}/episodes', self.episodes)
        ]
        for k, v in metrics:
            self.writer.add_scalar(k, v, global_step=self.global_frames)
        with (self.logdir / f'{prefix}_metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict([('step', self.global_frames)] + metrics)) + '\n')

    @property
    def global_frames(self):
        return self.global_steps * self.c.action_repeat

    @property
    def global_frames_str(self):
        length = len(str(self.c.steps))
        length = len(str(self.c.steps))
        return f'{self.global_frames:0{length}d}'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='config file')
    args, remaining = parser.parse_known_args()

    # Load YAML config
    config = OmegaConf.structured(Config)
    if args.config:
        config = OmegaConf.merge(config, OmegaConf.load(args.config))

    # Load command line configuration
    config = OmegaConf.merge(config, OmegaConf.from_cli(remaining))

    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()