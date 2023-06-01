import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.modeling_utils   import (
    Conv1D,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)


class Attention(nn.Module):
    def __init__(
        self,
        nx:     int,
        n_ctx:  int,
        config,
        scale               = False,
        is_cross_attention  = False
    ) -> None:
        super().__init__()

        n_state = nx    # in Attention: n_state = 768 (nx = n_embd)
        assert n_state % config.n_head == 0
        
        # lower triangular matrix of [n_ctx, n_ctx] Ones
        self.register_buffer(
            'bias', torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)  
        )
        self.register_buffer(
            'masked_bias', torch.tensor(-1e4)
        )

        self.n_head                 = config.n_head
        self.split_size             = n_state
        self.scale                  = scale
        self.is_cross_attention     = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj     = Conv1D(n_state, nx)

        self.attn_dropout   =   nn.Dropout(config.attn_pdrop)
        self.resid_dropout  =   nn.Dropout(config.resid_pdrop)
        self.pruned_heads   =   set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return

        heads, index    =   find_pruneable_heads_and_indices(
            heads       =   heads,
            n_heads     =   self.n_head,
            head_size   =   self.split_size // self.n_head,
            already_pruned_heads=   self.prune_heads    
        )

        index_attn      =   torch.cat([
            index, 
            index + self.split_size, 
            index + (2 * self.split_size)
        ])
        
        # Prune conv1d layers
        self.c_attn     =   prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj     =   prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size =   (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head     =   self.n_head - len(heads)
        self.prune_heads=   self.prune_heads.union(heads)
    
    def _attn(self, q, k, v, attention_mask = None, head_mask=None, output_attentions=False) -> torch.Tensor:
        w = torch.matmul(q, k)
        if self.scale:
            w   = w / (float(v.size(-1)) ** 0.5)
        nd, ns  = w.size(-2), w.size(-1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            mask = self.bias[:, :, ns - nd: ns, :ns]
            w    = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            w = w + attention_mask
        
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)

        return outputs

    def merge_heads(self, x):
        x           = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1), )
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x           = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)    #   (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)    #   (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        layer_past              =   None,
        attention_mask          =   None,
        head_mask               =   None,
        encoder_hidden_states   =   None,
        encoder_attention_mask  =   None,
        use_cache               =   False,
        output_attentions       =   False
    ):
        if encoder_hidden_states is not None:
            assert hasattr(
                self, 'q_attn'
            ), "If class is used as cross attention, the weights 'q_attn' has to be defined"
            query           =   self.q_attn(hidden_states)
            key, value      =   self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask  =   encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        query   =   self.split_heads(query)
        key     =   self.split_heads(key, k=True)
        value   =   self.split_heads(value)

        if layer_past is not None:
            past_key, past_value    =   layer_past[0].transpose(-2, -1), layer_past[1]  #   transpose back cf below
            key                     =   torch.cat([past_key, key], dim=-1)
            value                   =   torch.cat([past_value, value], dim=-2)
        
        if use_cache is True:
            present     =   torch.stack((key.transpose(-2, -1), value))     #   transpose to have same shapes for stacking
        else:
            present     =   (None,)

        attn_outputs    =   self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a               =   attn_outputs[0]             #   [batch, head, seq_length, head_features]

        a               =   self.merge_heads(a)         #   [batch, seq_length, n_state]
        a               =   self.c_proj(a)              #   [batch, seq_length, n_state]
        a               =   self.resid_dropout(a)

        outputs         =   [a, present] + attn_outputs[1:]
        return outputs


class MLP(nn.Module):
    def __init__(self, n_state, config) -> None:
        super().__init__()
        nx          =   config.n_embd
        self.c_fc   =   Conv1D(n_state, nx)
        self.c_proj =   Conv1D(nx, n_state)
        self.act    =   ACT2FN[config.activation_function]
        self.dropout=   nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h   =   self.act(self.c_fc(x))
        h2  =   self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx: int, config, scale=False) -> None:
        super().__init__()
        hidden_size = config.n_embd
        inner_dim   = config.n_inner if config.n_inner is not None else 4 * hidden_size
        
        self.ln_1   = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn   = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2   = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.cross_attention    =   Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn      =   nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp    = MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past      =   None,
        attention_mask  =   None,
        head_mask       =   None,
        encoder_hidden_states   =   None,
        encoder_attention_mask  =   None,
        use_cache               =   False,
        output_attentions       =   False
    ):
        attn_outputs    =   self.attn(
            self.ln_1(hidden_states),
            layer_past          =   layer_past,
            attention_mask      =   attention_mask,
            head_mask           =   head_mask,
            use_cache           =   use_cache,
            output_attentions   =   output_attentions
        )
        attn_output     =   attn_outputs[0]
        outputs         =   attn_outputs[1:]
        # residual connection
        hidden_states   =   attn_outputs + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "cross_attention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs  =   self.cross_attention(
                self.ln_cross_attn(hidden_states),
                attention_mask          =   attention_mask,
                head_mask               =   head_mask,
                encoder_hidden_states   =   encoder_hidden_states,
                encoder_attention_mask  =   encoder_attention_mask,
                output_attentions       =   output_attentions
            )
            attn_output         =   cross_attn_outputs[0]
            # residual connection
            hidden_states       =   hidden_states + attn_output
            outputs             =   outputs + cross_attn_outputs[2:]    # add cross attentions if we output attention weights

        feed_forward_hidden_states  =   self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states               =   hidden_states + feed_forward_hidden_states

        outputs     =   [hidden_states] + outputs
        return outputs