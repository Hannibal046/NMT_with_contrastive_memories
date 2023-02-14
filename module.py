import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    ModelOutput,
    Seq2SeqModelOutput,
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.marian.modeling_marian import(
    MarianAttention,
    MarianSinusoidalPositionalEmbedding,
    MarianEncoderLayer,
    MarianDecoderLayer
)
from dataclasses import dataclass
import numpy as np
from typing import Optional,Tuple

@dataclass
class DualBaseModelOutput(BaseModelOutput):
    src_enc_hidden_states: torch.Tensor = None
    tm_enc_hidden_states: torch.Tensor = None

    src_enc_last_hidden_state: torch.Tensor = None
    tm_enc_last_hidden_state: torch.Tensor = None

    src_enc_attentions: torch.Tensor = None
    tm_enc_attentions: torch.Tensor = None


@dataclass
class TripleBaseModelOutput(BaseModelOutput):
    src_enc_hidden_states: torch.Tensor = None
    tm_enc_hidden_states: torch.Tensor = None
    label_enc_hidden_states: torch.Tensor = None

    src_enc_last_hidden_state: torch.Tensor = None
    tm_enc_last_hidden_state: torch.Tensor = None
    label_enc_last_hidden_state: torch.Tensor = None

    src_enc_attentions: torch.Tensor = None
    tm_enc_attentions: torch.Tensor = None
    label_enc_attentions: torch.Tensor = None

@dataclass
class MyBaseModelOutput(BaseModelOutput):
    src_enc_hidden_states: torch.Tensor = None
    src_enc_last_hidden_state: torch.Tensor = None
    src_enc_attentions: torch.Tensor = None

@dataclass
class MySeq2SeqLMOutput(Seq2SeqLMOutput):
    scores: torch.Tensor = None
    contrastive_loss: float = None
    cross_entropy_loss: float = None

@dataclass 
class MyBaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    decoder_input_embeds: torch.Tensor = None

@dataclass
class MySeq2SeqModelOutput(Seq2SeqModelOutput):
    decoder_input_embeds: torch.Tensor = None

class GroupSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, 
                group_positions,
                ):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        
        group_positions = torch.tensor(group_positions, dtype=torch.long, device=self.weight.device)
        return super().forward(group_positions)

class CopyModule(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.proj_logits = nn.Linear(d_model,1)
        self.proj_decoder_input_embeds = nn.Linear(d_model,1)
        self.proj_encoder_output = nn.Linear(d_model,1)
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self,
                decoder_attention,
                decoder_last_hidden_state,
                enc_input_ids,
                logits,
                decoder_input_embeds,
                encoder_last_hidden_state):
        # input:
        # decoder_attention: bs,n_heads,tgt_len,src_len
        # decoder_hidden_states: bs,trg_len,hid_dim, already masked pad token in the src
        # enc_input_ids: bs,src_len
        # logits: bs,trg_len,trg_vocab_size
        # encoder_last_hidden_state: bs,seq_len,hid_dim

        # output:
        # scores for each word
        
        g_distribution = torch.softmax(logits,dim=-1) # [bs,trg_len,trg_vocab_size] generate_distribution 
        #c_attention = torch.mean(decoder_attention,dim=1) # [bs,trg_len,src_len] average across all heads
        c_attention = torch.mean(decoder_attention,dim=1)
        trg_len = c_attention.shape[1]

        c_distribution = torch.zeros_like(g_distribution) # [bs,trg_len,trg_vocab_size]
        c_distribution.scatter_add_(dim=2,
                            index = enc_input_ids.unsqueeze(1).expand(-1,trg_len,-1), # bs,trg_len,src_len
                            src = c_attention, #[bs,trg_len,src_len]
                            ) # [bs,trg_len,trg_vocab_size]
                        
        p_copy = torch.sigmoid(
            self.proj_logits(decoder_last_hidden_state)
           +self.proj_decoder_input_embeds(decoder_input_embeds)
           +self.proj_encoder_output(torch.bmm(c_attention,encoder_last_hidden_state))
           +self.bias
            ) # bs,trg_len,1
        # original
        #final_distribution = p_copy * c_distribution + (1-p_copy) * g_distribution
        # for memory efficient
        g_distribution = p_copy * c_distribution + (1-p_copy) * g_distribution
        eps = 1e-12
        return torch.log(g_distribution+eps)

class CombineAttention(MarianAttention):
    def __init__(self,
                embed_dim,
                dropout,
                num_heads,
                is_decoder=False,
                bias=True,):
        super().__init__(
            embed_dim = embed_dim,
            num_heads = num_heads)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        
        self.q_proj = nn.Linear(embed_dim,embed_dim,bias=bias)
        self.k_proj = nn.Linear(embed_dim,embed_dim,bias=bias)
        self.v_proj = nn.Linear(embed_dim,embed_dim,bias=bias)
        self.out_proj = nn.Linear(embed_dim,embed_dim,bias=bias)

        self.combine_conv = nn.Sequential(
            *[nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for _ in range(self.num_heads)]
        )
        self.conv_activation_fn = F.tanh
    def forward(
        self,
        hidden_states,
        attention_mask,
        similarity_score,
    ):
        bsz,tm_len,embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tm_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        ## bsz*num_heads,tm_len,tm_len
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = attn_weights.view(bsz, self.num_heads, tm_len, tm_len)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights,dim=-1) # put attn_weight and similartiy_scores in the same scale
        similarity_score = nn.functional.softmax(similarity_score,dim=-1)

        # combine
        energy_ls = []
        for idx in range(self.num_heads):
            energy = torch.stack((attn_weights[:,idx,:,:],similarity_score),dim=1)
            assert energy.shape[1] == 2
            energy_ls.append(self.combine_conv[idx](energy).squeeze(1))
        attn_weights = self.conv_activation_fn(torch.stack(energy_ls,dim=1))
        
        # mask
        attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tm_len, tm_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tm_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tm_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

class StaticAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        dropout,
        num_heads = 1,
        is_decoder=False,
        bias = True,
    ):
        super().__init_()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        #self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.v_proj = nn.Linear(embed_dim,embed_dim,bias=bias)
        self.out_proj = nn.Linear(embed_dim,embed_dim,bias=bias)


    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        static_attention_mask = None,
        similarity_score = None, # [bs,seq_len,seq_len]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, seq_len, embed_dim = hidden_states.size()

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        value_states = hidden_states.view(*proj_shape)

        inverted_mask = 1.0 - static_attention_mask
        attention_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(value_states.dtype).min)

        if attention_mask is not None:
            attn_weights = similarity_score.view(bsz, self.num_heads, seq_len, seq_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, seq_len, seq_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, seq_len, seq_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, seq_len, seq_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)


        #############################
        #########  modify  ##########
        #############################
        
        # attn_probs: [bs * num_heads,query_len,key_len]
        # attn_probs = attn_probs * static_attention_mask


        # attn_porb: [bs*num_heads,tgt_len,src_len]
        # value_states: [bs*num_heads,src_len,head_dim]
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, seq_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, seq_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

class ExampleEncoderLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
    def forward(
        self,
        hidden_states,
        *args,
        **kwargs,
    ):
        return (hidden_states,)

class BaseTmEncoderLayer(MarianEncoderLayer):

    def __init__(self,config):
        super().__init__(config)
        self.self_attn = MarianAttention(
            embed_dim=self.embed_dim,
            num_heads = config.tm_encoder_attention_heads,
            dropout = config.attention_dropout,
        )
        self.fc1 = nn.Linear(self.embed_dim,config.tm_encoder_ffn_dim)
        self.fc2 = nn.Linear(config.tm_encoder_ffn_dim,self.embed_dim)

class DualSelfAttnEncoderLayer(MarianEncoderLayer):
    def __init__(self,config):
        super().__init__(config)
        self.fc1 = nn.Linear(self.embed_dim,config.tm_encoder_ffn_dim)
        self.fc2 = nn.Linear(config.tm_encoder_ffn_dim,self.embed_dim)
        self.group_attn = MarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.tm_encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.combine_attn = CombineAttention(
            embed_dim = self.embed_dim,
            num_heads=config.tm_encoder_attention_heads,
            dropout=config.attention_dropout
        )

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        layer_head_mask = None,
        output_attentions = False,
        #group_attention_mask = None, # since group attention can be integrate into the original attetion mechanism
        static_attention_mask = None,
        similarity_score = None,
        index_ls = None,
    ):
        # group_attn
        residual = hidden_states
        hidden_states,attn_weights,_ = self.group_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            #group_attention_mask = group_attention_mask
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)


        # static_attn
        all = hidden_states
        hidden_states = torch.stack([hidden_states[idx,index_list] for idx,index_list in enumerate(index_ls)])

        residual = hidden_states
        hidden_states  = self.combine_attn(
            hidden_states=hidden_states,
            attention_mask=static_attention_mask,
            similarity_score = similarity_score,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        for idx in range(all.shape[0]):
            all[idx,index_ls[idx],:] = hidden_states[idx]
        
        hidden_states = all


        # position_ffn
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        # if output_attentions:
        #     outputs += (attn_weights,)

        return outputs

class ExampleDecoderLayer(MarianDecoderLayer):
    
    def __init__(self, config):
        super().__init__(config)

        self.tm_attn = MarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        #encoder_hidden_states: Optional[torch.Tensor] = None,
        #encoder_attention_mask: Optional[torch.Tensor] = None,
        src_enc_last_hidden_state=None,
        src_attention_mask=None,
        tm_enc_last_hidden_state=None,
        tm_attention_mask=None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):

        ## to be compatible with previous setting
        encoder_hidden_states = src_enc_last_hidden_state
        encoder_attention_mask = src_attention_mask

        ## example layer
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        self_attn_hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        self_attn_hidden_states = nn.functional.dropout(self_attn_hidden_states, p=self.dropout, training=self.training)
        
        tm_cross_attn_present_key_value = None
        tm_cross_attn_weights = None
        if tm_enc_last_hidden_state is not None:
            
            tm_cross_attn_past_key_value = past_key_value[4:6] if past_key_value is not None else None
            tm_attn_hidden_states, tm_cross_attn_weights, tm_cross_attn_present_key_value = self.tm_attn(
                hidden_states=hidden_states,
                key_value_states=tm_enc_last_hidden_state,
                attention_mask=tm_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=tm_cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            tm_attn_hidden_states = nn.functional.dropout(tm_attn_hidden_states, p=self.dropout, training=self.training)
        
        ## add and norm
        hidden_states = self.self_attn_layer_norm(self_attn_hidden_states + tm_attn_hidden_states + hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value
            present_key_value = present_key_value + tm_cross_attn_present_key_value


        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, tm_cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class TransformerDecoderLayer(MarianDecoderLayer):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            #encoder_hidden_states: Optional[torch.Tensor] = None,
            #encoder_attention_mask: Optional[torch.Tensor] = None,
            src_enc_last_hidden_state=None,
            src_attention_mask=None,
            tm_enc_last_hidden_state=None,
            tm_attention_mask=None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
        ):
            encoder_hidden_states = src_enc_last_hidden_state
            encoder_attention_mask = src_attention_mask
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )


class DualCrossAttentionDecoderLayer(MarianDecoderLayer):
    def __init__(self,config):
        super().__init__(config)
        
        self.tm_attn = MarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.tm_attn_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        #encoder_hidden_states: Optional[torch.Tensor] = None,
        #encoder_attention_mask: Optional[torch.Tensor] = None,
        src_enc_last_hidden_state=None,
        src_attention_mask=None,
        tm_enc_last_hidden_state=None,
        tm_attention_mask=None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):

        ## to be compatible with previous setting
        encoder_hidden_states = src_enc_last_hidden_state
        encoder_attention_mask = src_attention_mask

        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        #############################
        #########  modify  ##########
        #############################
        tm_cross_attn_present_key_value = None
        tm_cross_attn_weights = None
        if tm_enc_last_hidden_state is not None:
            
            residual = hidden_states
            tm_cross_attn_past_key_value = past_key_value[4:6] if past_key_value is not None else None
            hidden_states, tm_cross_attn_weights, tm_cross_attn_present_key_value = self.tm_attn(
                hidden_states=hidden_states,
                key_value_states=tm_enc_last_hidden_state,
                attention_mask=tm_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=tm_cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.tm_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + tm_cross_attn_present_key_value


        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, tm_cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class ContrastiveLoss(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.contrastive_lambda = config.contrastive_lambda
        self.contrastive_loss_balance = config.contrastive_loss_balance
        self.pooler = Pooler(config)
        self.sim_fn = Similarity(config)
    def forward(
        self,
        encoder_hidden_states = None,
        src_attention_mask = None,
        tm_attention_mask = None,
        label_attention_mask = None,
        tm_input_ids = None
    ):
        src_enc_hidden_states,tm_enc_hidden_states,label_enc_hidden_states = encoder_hidden_states

        ## make tm contrast to each other while approaching label        
        src = self.pooler(src_enc_hidden_states,src_attention_mask) # bs,d_model
        tm = self.pooler(tm_enc_hidden_states,tm_attention_mask,tm_input_ids) # bs*tm_size,d_model
        label = self.pooler(label_enc_hidden_states,label_attention_mask) # bs,d_model
        

        tm_size = self.config.tm_size
        batch_size,d_model = label.shape
        left = tm[:,None,:].expand(-1,tm_size,-1) # [bs*tm_size,tm_size-1+1,d_model]
        
        right = tm.view(batch_size,tm_size,-1)[:,:,None,:].expand(-1,-1,tm_size,-1).permute(0,2,1,3)
        diag_mask = ~torch.diag(torch.ones(tm_size))[None,:,:,None].expand(batch_size,-1,-1,d_model).bool().to(tm.device) # bs,tm_size,tm_size,d_model
        right = right.masked_select(diag_mask).view(batch_size,tm_size,tm_size-1,d_model)
        right = torch.cat((label[:,None,None,:].expand(-1,tm_size,1,-1),right),dim=2).view(left.shape)

        confusion_matrix = self.sim_fn(left,right) # bs*tm_size,tm_size
        loss = -nn.LogSoftmax(1)(confusion_matrix)
        loss = loss[:,0].mean()
        if not self.contrastive_loss_balance:
            balance_factor = 1
        else:
            balance_factor = torch.sum(tm_input_ids != self.config.pad_token_id) / tm_input_ids.shape[0]
         
        return self.contrastive_lambda * balance_factor * loss
        ### make tm a whole approaching label while contrast to other label in batch
        
        # tm = self.pooler(tm_enc_hidden_states,tm_attention_mask) # bs,d_model
        # label = self.pooler(label_enc_hidden_states,label_attention_mask) # bs,d_model
        # bs,d_model = label.shape
        # confusion_matrix = self.sim_fn(
        #     tm[:,None,:].expand(-1,bs,-1),
        #     label[:,None,:].expand(-1,bs,-1).permute(1,0,2)
        # )
        # loss = -nn.LogSoftmax(1)(confusion_matrix).diag().mean()

        # balance_factor = torch.sum(tm_input_ids != self.config.pad_token_id) / tm_input_ids.shape[0]

        # return (self.contrastive_lambda * self.contrastive_loss_balance * balance_factor) * loss

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, config):
        super().__init__()
        self.temp = config.contrastive_temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_mlp': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pooler_type = config.pooler_type
        assert self.pooler_type in ["cls", "cls_mlp", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type
        if self.pooler_type == 'cls_mlp':
            self.mlp = MLPLayer(config)

    def forward(self, hidden_states,attention_mask,tm_input_ids=None):
        if tm_input_ids is None:
            last_hidden = hidden_states[-1]
            
            if self.pooler_type == 'cls':
                return last_hidden[:,0,:]
            if self.pooler_type == 'cls_mlp':
                return self.mlp(last_hidden[:,0,:])
            elif self.pooler_type == "avg":
                return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            elif self.pooler_type == "avg_first_last":
                first_hidden = hidden_states[0]
                pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                return pooled_result
            elif self.pooler_type == "avg_top2":
                second_last_hidden = hidden_states[-2]
                last_hidden = hidden_states[-1]
                pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                return pooled_result
            else:
                raise NotImplementedError
        
        else:
            mask = torch.eq(tm_input_ids,self.config.bos_token_id).unsqueeze(-1)
            bs,tm_len,d_model = hidden_states[-1].shape
            tm_size = self.config.tm_size

            last_hidden_state = hidden_states[-1].masked_select(mask).view(bs*tm_size,d_model)                
            
            if self.pooler_type == 'cls':
                return last_hidden_state
            elif self.pooler_type == 'cls_mlp':
                return self.mlp(last_hidden_state)
            elif self.pooler_type == 'avg_first_last':
                return (hidden_states[0].masked_select(mask).view(bs*tm_size,d_model) + last_hidden_state) / 2.0
            elif self.pooler_type == 'avg_top2':
                return (hidden_states[-2].masked_select(mask).view(bs*tm_size,d_model) + last_hidden_state) / 2.0
            else:
                raise NotImplementedError

def avg_2d_tensor(t,split):
    # t: [seq_len,d_model]
    # split: [2,3,5]
    assert t.shape[0] == sum(split)
    cur = 0
    ret = []
    for s in split:
        ret.append(t[cur:s].mean(dim=1))
        cur += s
    return torch.stack(ret,dim=0) # [len(split),d_model]

def avg_3d_tensor(t,split):
    # t [bs,seq_len,d_model]
    assert t.shape[0] == len(split)
    ret = []
    for idx,s in enumerate(split):
        ret.append(avg_2d_tensor(t[idx],s))
    return torch.stack(ret,dim=0)

