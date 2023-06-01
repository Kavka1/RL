import torch

from RL.DT.training.trainer import Trainer

class ActTrainer(Trainer):
    def train_step(self):
        states, actions, rewards, dones, rtg, _, attention_masks    =   self.get_batch(self.batch_size)
        state_target, action_target, reward_target                  =   torch.clone(states), torch.clone(actions), torch.clone(rewards)
        
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, target_return=rtg[:,0], attention_masks=attention_masks,
        )
        
        a_dim   =   action_preds.shape[2]

        action_preds    =   action_preds.reshape(-1, a_dim)
        action_target   =   action_target[:,-1].reshape(-1, a_dim)

        loss    =   self.loss_fn(
            state_preds, action_preds, reward_preds, 
            state_target, action_target, reward_target
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()