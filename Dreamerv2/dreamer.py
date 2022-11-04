from typing     import List, Dict, Optional, Tuple, Union
from pathlib    import Path
from gym        import spaces

import json
import imageio
import collections
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.distributions     import kl_divergence

from RL.Dreamer.model.module import ConvEncoder, ConvDecoder, DenseDecoder, ActionDecoder
from RL.Dreamer.model.rssm   import RSSM
from RL.Dreamer.config       import Config
from RL.Dreamer.buffer       import ReplayBuffer
from RL.Dreamer.utils        import AverageMeter, AttrDict, freeze


act_dict = {
    'relu': nn.ReLU,
    'elu':  nn.ELU
}


class Dreamer(nn.Module):
    def __init__(
        self,
        config:         Config,
        action_space:   spaces.Box,
        writer:         SummaryWriter,
        logdir:         Path
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.a_dim        = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        self.c            = config
        self.metrics      = collections.defaultdict(AverageMeter)

        self.writer       = writer
        self.logdir       = logdir
        self.build_model()

    def build_model(self) -> None:
        # activation
        cnn_activation = act_dict[self.c.cnn_act]
        act            = act_dict[self.c.dense_act]
        # model
        feature_size = self.c.stoch_size + self.c.deter_size
        self.encoder   = ConvEncoder(
            depth       = self.c.cnn_depth, 
            activation  = cnn_activation
        )
        self.decoder = ConvDecoder(
            feature_dim = feature_size,
            depth       = self.c.cnn_depth,
            activation  = cnn_activation
        )
        self.dynamics  = RSSM(
            a_dim   =   self.a_dim,
            stoch   =   self.c.stoch_size,
            deter   =   self.c.deter_size,
            hidden  =   self.c.deter_size
        )
        self.reward = DenseDecoder(
            input_dim   =   feature_size,
            shape       =   (),
            layers      =   2,
            units       =   self.c.num_units,
            activation  =   act
        )
        if self.c.pcont:
            self.pcont  =   DenseDecoder(
                input_dim   =   feature_size,
                shape       =   (),
                layers      =   3,
                units       =   self.c.num_units,
                activation  =   act
            )
        # actor critic
        self.actor = ActionDecoder(
            input_dim   =   feature_size,
            a_dim       =   self.a_dim,
            layers      =   4,
            units       =   self.c.num_units,
            dist        =   self.c.action_dist,
            activation  =   act,
            init_std    =   self.c.action_init_std,
        )
        self.value = DenseDecoder(
            input_dim   =   feature_size,
            shape       =   (),
            layers      =   3,
            units       =   self.c.num_units,
            activation  =   act
        )

        self.model_modules = nn.ModuleList([
            self.encoder,
            self.dynamics,
            self.decoder,
            self.reward
        ])
        if self.c.pcont:
            self.model_modules.append(self.pcont)

        self.model_optimizer = optim.Adam(self.model_modules.parameters(), lr=self.c.model_lr, weight_decay=self.c.weight_decay)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.c.value_lr, weight_decay=self.c.weight_decay)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.c.actor_lr, weight_decay=self.c.weight_decay)

        # tensorflow initialization
        if self.c.tf_init:
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_uniform_(m.weight.data)
                    if hasattr(m.bias, 'data'):
                        m.bias.data.fill_(0.0)

    def preprocess_observation(self, obs: Dict[str, np.ndarray]) -> Dict:
        obs = {k: torch.as_tensor(v, device=self.c.device, dtype=torch.float) for k, v in obs.items()}
        obs['image'] = obs['image'] / 255.0 - 0.5
        return obs

    def preprocess_batch(self, data: Dict[str, np.ndarray]) -> Dict:
        data = {k: torch.as_tensor(v, device=self.c.device, dtype=torch.float) for k, v in data.items()}
        data['image'] = data['image'] / 255.0 - 0.5
        clip_rewards  = dict(none=lambda x: x, tanh=torch.tanh)[self.c.clip_rewards]
        data['reward']= clip_rewards(data['reward'])
        return data

    @torch.no_grad()
    def get_action(self, obs: Dict[str, np.ndarray], state: Optional[Tensor] = None, training: bool = True) -> Tuple[np.ndarray, Optional[Tensor]]:
        """
        Corresponds to Dreamer.__call__, but without training.
        Args:
            obs: obs['image'] shape (C, H, W), uint8
            state: None, or Tensor
        Returns:
            action: (D)
            state: None, or Tensor
        """
        # unsqueeze for T and B
        obs['image']    = obs['image'][None, None, ...]
        action, state   = self.policy(obs, state, training)
        action          = action.squeeze(axis=0)
        return action, state

    def policy(self, obs: Tensor, state: Tensor, training: bool) -> Tensor:
        """
        Args:
            obs: (B, C, H, W)
            state: (B, D)
        Returns:
            action: (B, D)
            state: (B, D)
        """
        if state is None:
            latent = self.dynamics.initial(len(obs['image']))
            action = torch.zeros((len(obs['image']), self.a_dim), dtype=torch.float32).to(self.c.device)
        else:
            latent, action = state
        embed = self.encoder(self.preprocess_observation(obs))
        embed = embed.squeeze(0)
        latent, _ = self.dynamics.obs_step(latent, action, embed)       # get the posterior state
        feat  = self.dynamics.get_feat(latent)
        if training:
            action = self.actor(feat).sample()                              # stochastic action
        else:
            action = torch.tanh(self.actor(feat).base_dist.base_dist.mean)  # deterministic action
        action = self.exploration(action, training)
        state  = (latent, action)
        action = action.cpu().detach().numpy()
        action = np.array(action, dtype=np.float32)
        return action, state

    def exploration(self, action: Tensor, training: bool) -> Tensor:
        """
        Args:
            action: (B, D)
        Returns:
            action: (B, D)
        """
        if training:
            amount = self.c.expl_amount
            if self.c.expl_decay:
                amount *= 0.5 ** (self.step / self.c.expl_decay)
            if self.c.expl_min:
                amount = max(self.c.expl_min, amount)
            self.metrics['expl_amount'].update_state(amount)
        elif self.c.eval_noise:
            amount = self.c.eval_noise
        else:
            return action
        if self.c.expl == 'additive_gaussian':
            return torch.clamp(torch.normal(action, amount), -1, 1)
        if self.c.expl == 'completely_random':
            return torch.rand(action.shape, -1, 1)
        if self.c.expl == 'epsilon_greedy':
            indices = torch.distributions.Categorical(0 * action).sample()
            return torch.where(
                torch.rand(action.shape[:1], 0, 1) < amount,
                torch.one_hot(indices, action.shape[-1], dtype=self.float),
                action)
        raise NotImplementedError(self.c.expl)

    def imagine_ahead(self, post: dict) -> Tensor:
        """
        Starting from a posterior, do rollout using your currenct policy.

        Args:
            post: dictionary of posterior state. Each (B, T, D)
        Returns:
            imag_feat: (T, B, D). concatenation of imagined posteiror states. 
        """
        if self.c.pcont:
            # (B, T, D)
            # last state may be terminal. Terminal's next discount prediction is not trained.
            post = {k: v[:, :-1] for k, v in post.items()}
        # (B, T, D) -> (BT, D)
        flatten = lambda x: x.reshape(-1, *x.size()[2:])
        start   = {k: flatten(v).detach() for k, v in post.items()}
        state   = start

        state_list = [start]
        for i in range(self.c.horizon):
            if self.c.update_horizon is not None and i >= self.c.update_horizon:
                with torch.no_grad():
                    action = self.actor(self.dynamics.get_feat(state).detach()).rsample()
            else:
                action = self.actor(self.dynamics.get_feat(state).detach()).rsample()

            state = self.dynamics.img_step(state, action)
            state_list.append(state)
            if self.c.single_step_q:
                # Necessary, if you are using single step q estimate
                state = {k: v.detach() for k, v in state.items()}
        # (H, BT, D)
        states = {k: torch.stack([state[k] for state in state_list], dim=0) for k in state_list[0]}
        imag_feat = self.dynamics.get_feat(states)
        return imag_feat

    def update(self, replay_buffer: ReplayBuffer, log_images: bool, video_path: Path) -> None:
        """
        Corresponds to Dreamer._train.

        Update the model and policy/value. Log metrics and video.
        """
        data = replay_buffer.sample(self.c.batch_size, self.c.batch_length)
        data = self.preprocess_batch(data)
        # (B, T, D)
        embed = self.encoder(data)
        post, prior = self.dynamics.observe(embed, data['action'])
        # (B, T, D)
        feat  = self.dynamics.get_feat(post)
        # (B, T, 3, H, W), std = 1.0
        image_pred  = self.decoder(feat)
        # (B, T, 1)
        reward_pred = self.reward(feat)
        likes = AttrDict()
        # mean over batch and time, sum over pixel
        likes.image = image_pred.log_prob(data['image']).mean(dim=[0,1])
        likes.reward= reward_pred.log_prob(data['reward']).mean(dim=[0,1])
        if self.c.pcont:
            pcont_pred      = self.pcont(feat)
            pcont_target    = self.c.discount * data['discount']
            likes.pcont     = torch.mean(pcont_pred.log_prob(pcont_target), dim=[0,1])
            likes.pcont     *= self.c.pcont_scale
        prior_dist = self.dynamics.get_dist(prior)
        post_dist  = self.dynamics.get_dist(post)
        div = kl_divergence(post_dist, prior_dist).mean(dim=[0,1])
        div = torch.clamp(div, min=self.c.free_nats)
        model_loss = self.c.kl_scale * div - sum(likes.values())

        # Actor Loss
        with freeze(nn.ModuleList([self.model_modules, self.value])):
            # (H + 1, BT, D), indexed t = 0 to H, includes the start state 
            # unlike original implementation
            imag_feat = self.imagine_ahead(post)
            reward    = self.reward(imag_feat[1:]).mean
            if self.c.pcont:
                pcont = self.pcont(imag_feat[1:]).mean
            else:
                pcont = self.c.discount * torch.ones_like(reward)
            
            value = self.value(imag_feat[1:]).mean
            # The original implementation seems to be incorrect (off by one error)
            # This one should be correct
            # For t = 0 to H - 1
            returns = torch.zeros_like(value)
            last = value[-1]
            for t in reversed(range(self.c.horizon)):
                # V_t = r_t + gamma * ((1 - lambda) * v_t + lambda * V_t+1)
                returns[t] = (
                    reward[t] + pcont[t] * (
                        (1. - self.c.disclam)   * value[t] + \
                        self.c.disclam          * last
                    )
                )
                last = returns[t]
            # [H, BT, D]
            with torch.no_grad():
                # mask[t] -> state[t] is terminal or after a terminal state
                mask = torch.cat([torch.ones_like(pcont[:1]), torch.cumprod(pcont, dim=0)[:-1]], dim=0)
            
            if not self.c.single_step_q:
                actor_loss = - (mask * returns).mean(dim=[0,1])

        # Value loss
        target = returns.detach()
        if self.c.update_horizon is None:
            value_pred = self.value(imag_feat[:-1].detach())
            value_loss = torch.mean( - value_pred.log_prob(target) * mask, dim=[0, 1])
        else:
            value_pred = self.value(imag_feat[:self.c.update_horizon].detach())
            value_loss = torch.mean(
                - value_pred.log_prob(target[:self.c.update_horizon]) * \
                    mask[:self.c.update_horizon], 
                dim=[0,1]
            )

        self.model_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)

        (value_loss + model_loss + actor_loss).backward()

        actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.c.grad_clip)
        value_norm = nn.utils.clip_grad_norm_(self.value.parameters(), self.c.grad_clip)
        model_norm = nn.utils.clip_grad_norm_(self.model_modules.parameters(), self.c.grad_clip)
        self.actor_optimizer.step()
        self.model_optimizer.step()
        self.value_optimizer.step()

        if self.c.log_scalars:
            self.scalar_summaries(
                data, feat, prior_dist, post_dist, likes,
                div, model_loss, value_loss, actor_loss, 
                model_norm, value_norm, actor_norm
            )
        if log_images:
            self.image_summaries(data, embed, image_pred, video_path)

    def write_log(self, step: int):
        """
        Corresponds to Dreamer._write_summaries
        """
        metrics = [(k, float(v.result())) for k, v in self.metrics.items()]
        [m.reset_states() for m in self.metrics.values()]
        with (self.logdir / 'agent_metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        for k, m in metrics:
            self.writer.add_scalar('agent/' + k, m, global_step=step)
        # print(colored(f'[{step}]', 'red'), ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self.writer.flush()

    @torch.no_grad()
    def scalar_summaries(
          self, data, feat, prior_dist, post_dist, likes, div,
          model_loss, value_loss, actor_loss, model_norm, value_norm,
          actor_norm):
        self.metrics['model_grad_norm'].update_state(model_norm)
        self.metrics['value_grad_norm'].update_state(value_norm)
        self.metrics['actor_grad_norm'].update_state(actor_norm)
        self.metrics['prior_ent'].update_state(prior_dist.entropy().mean())
        self.metrics['post_ent'].update_state(post_dist.entropy().mean())
        for name, logprob in likes.items():
          self.metrics[name + '_loss'].update_state(-logprob)
        self.metrics['div'].update_state(div)
        self.metrics['model_loss'].update_state(model_loss)
        self.metrics['value_loss'].update_state(value_loss)
        self.metrics['actor_loss'].update_state(actor_loss)
        self.metrics['action_ent'].update_state(self.actor(feat).base_dist.base_dist.entropy().sum(dim=-1).mean())

    @torch.no_grad()
    def image_summaries(self, data, embed, image_pred, video_path):
        # Take the first 6 sequences in the batch
        B, T, C, H, W = image_pred.mean.size()
        B = 6
        truth = data['image'][:6] + 0.5
        recon = image_pred.mean[:6]
        init, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data['action'][:6, 5:], init)
        openl = self.decoder(self.dynamics.get_feat(prior)).mean
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], dim=1)
        error = (model - truth + 1) / 2
        # (B, T, 3, 3H, W)
        openl = torch.cat([truth, model, error], dim=3)
        # (T, 3H, B * W, 3)
        openl = openl.permute(1, 3, 0, 4, 2).reshape(T, 3 * H, B * W, C).cpu().numpy()
        openl = (openl * 255.).astype(np.uint8)
        # video_path = self.video_dir / 'model' / f'{self.global_frames_str}.gif'
        video_path.parent.mkdir(exist_ok=True)
        # imageio.mimsave(video_path, openl, fps=20, ffmpeg_log_level='error')
        imageio.mimsave(video_path, openl, fps=30)

        # self.writer.add_image()
        # tools.graph_summary(
            # self._writer, tools.video_summary, 'agent/openl', openl)

    def load(self, path: Union[str, Path], device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        path = Path(path)
        with path.open('wb') as f:
            self.load_state_dict(torch.load(f, map_location=device))

    # Change to state dict if we just want to save the weights
    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('wb') as f:
            torch.save(self.state_dict(), f)