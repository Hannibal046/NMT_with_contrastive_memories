from typing import Union
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from transformers.models.marian.modeling_marian import (
    MarianDecoder,
    MarianEncoder,
    MarianMTModel,
    MarianModel,
    MarianEncoderLayer, 
    MarianPreTrainedModel,
    _expand_mask)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    ModelOutput,
    Seq2SeqModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.generation_beam_search import BeamScorer
from transformers.generation_utils import BeamSearchDecoderOnlyOutput,BeamSearchEncoderDecoderOutput
from transformers.generation_logits_process import (
    LogitsProcessorList,
)
from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.marian.configuration_marian import MarianConfig
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]

import torch.nn as nn
from typing import Optional,Tuple,Dict,Any

from _transformers import MyLabelSmoother

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    ModelOutput,
    Seq2SeqModelOutput,
    BaseModelOutputWithCrossAttentions
)

from module import (
    DualBaseModelOutput,
    TripleBaseModelOutput,
    MarianSinusoidalPositionalEmbedding,
    MyBaseModelOutput,
    MySeq2SeqLMOutput,
    MySeq2SeqModelOutput,
    MyBaseModelOutputWithPastAndCrossAttentions,
    GroupSinusoidalPositionalEmbedding,
    CopyModule,
    CombineAttention,
    StaticAttention,
    ExampleEncoderLayer,
    BaseTmEncoderLayer,
    DualSelfAttnEncoderLayer,
    ExampleDecoderLayer,
    TransformerDecoderLayer,
    DualCrossAttentionDecoderLayer,
    ContrastiveLoss,
    MLPLayer,
    Similarity,
    Pooler,
)

"""
Model Architecture:

MarianMTModel
    MarianModel
        DualEncoder
            SrcEncoder
                SrcEncoderLayer
                SrcEncoderLayer
                SrcEncoderLayer
                SrcEncoderLayer
                SrcEncoderLayer
                SrcEncoderLayer
            TmEncoder
                TmEncoderLayer
                TmEncoderLayer
                TmEncoderLayer
                TmEncoderLayer
                TmEncoderLayer
                TmEncoderLayer
        Decoder
            DecoderLayer
            DecoderLayer
            DecoderLayer
            DecoderLayer
            DecoderLayer
            DecoderLayer
    CopyModule
    LMHead
    ContrastiveLoss
"""

class TransformerEncoder(MarianPreTrainedModel):

    def __init__(self, config: MarianConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.src_vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = MarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([MarianEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.init_weights()

    def reverse_attention_mask(self,attention_mask,dtype):
        
        inverted_mask = 1.0 - attention_mask
        return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        group_attention_mask = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        
        ############
        ## Modify ##
        ############
        if group_attention_mask is not None:
            group_attention_mask = torch.tensor(group_attention_mask,device = inputs_embeds.device).unsqueeze(1)
            attention_mask = attention_mask + self.reverse_attention_mask(group_attention_mask,inputs_embeds.dtype) # -inf in the pos not attended
        
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class TransformerDecoder(MarianDecoder):
    def __init__(self,config,embed_tokens=None):
        super().__init__(config)
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.trg_vocab_size,config.d_model,config.pad_token_id)
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        encoder_hidden_states = kwargs['src_enc_last_hidden_state']
        encoder_attention_mask = kwargs['src_attention_mask']
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, 
        )

class TmEncoder(MarianEncoder):
    def __init__(self,config,embed_tokens=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        if config.tm_encoder_type == 'baseline':
            self.layers = nn.ModuleList([BaseTmEncoderLayer(config) for _ in range(config.tm_encoder_layers)])
        elif config.tm_encoder_type == 'example':
            self.layers = nn.ModuleList([ExampleEncoderLayer(config) for _ in range(1)])
        elif config.tm_encoder_type == 'dual_self_attention':
            self.layers = nn.ModuleList([DualSelfAttnEncoderLayer(config) for _ in range(config.tm_encoder_layers)])
        elif config.tm_encoder_type =='group_attention':
            self.layers = nn.ModuleList([BaseTmEncoderLayer(config) for _ in range(config.tm_encoder_layers)])
            # self.embed_positions = GroupSinusoidalPositionalEmbedding(config.max_position_embeddings,
            #                                                           config.d_model,self.padding_idx,)
        else:
            raise ValueError('no tm encoder type')
        #self.embed_tm_type = nn.Embedding(5,config.d_model)
        self.config = config
        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask = None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        similarity_score = None,
        static_attention_mask = None,
        index_ls = None,
        group_attention_mask = None,
        #group_positions = None,
        #tm_type = None,
    ):
        input_shape=input_ids.size()
        #input_ids = input_ids.review(-1,input_shape[-1]
        # if self.config.tm_encoder_type == 'group_attention':
        #     embed_pos = self.embed_positions(group_positions)
        # else:
        embed_pos = self.embed_positions(input_shape)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        hidden_states = inputs_embeds + embed_pos
        # if tm_type is not None:
        #     tm_type_embeds = self.embed_tm_type(tm_type)
        #     hidden_states = hidden_states + tm_type_embeds
            
        hidden_states = nn.functional.dropout(hidden_states,p=self.dropout,training=self.training)
        

        if attention_mask is not None:
            pad_attention_mask = _expand_mask(attention_mask,inputs_embeds.dtype)
        if group_attention_mask is not None:
            group_attention_mask = torch.tensor(group_attention_mask,device = inputs_embeds.device).unsqueeze(1)
            attention_mask = pad_attention_mask + self.reverse_attention_mask(group_attention_mask,inputs_embeds.dtype) # -inf in the pos not attended
        # elif self.config.tm_encoder_type == 'dual_self_attention':
        #     attention_mask = pad_attention_mask + self.make_group_attention_mask(attention_mask,inputs_embeds.dtype)
        else:
            attention_mask = pad_attention_mask
        # if static_attention_mask is not None:
        #     static_attention_mask = _expand_mask(static_attention_mask,inputs_embeds.dtype)
        
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.config.tm_encoder_type == 'dual_self_attention':
                
                additional_kw_args = {
                    "static_attention_mask":static_attention_mask,
                    'similarity_score':similarity_score,
                    "static_attention_mask":static_attention_mask,
                    "index_ls":index_ls,
                }
            else:
                additional_kw_args = {}
            layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        **additional_kw_args,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return TripleBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
        
    def get_bos_pos(self,input_ids):
        ret = []
        for batch_idx in range(input_ids.shape[0]):
            temp_ls = []
            for pos,token_id in enumerate(input_ids[batch_idx]):
                if int(token_id) == self.config.bos_token_id:temp_ls.append(pos)
            ret.append(temp_ls)
        return ret

    def make_group_attention_mask(self,input_ids,dtype):
        # attention_mask : [bs,seq_len]
        # input_ids: [bs,seq_len]
        # output: [bs,1,seq_len,seq_len]
        bs,seq_len = input_ids.shape
        l = input_ids.tolist()
        expanded_mask = torch.zeros((bs,1,seq_len,seq_len)).to(input_ids.device)
        bos_pos_ls = []
        
        for single_batch in l:
            bos_pos = [idx for idx,x in enumerate(single_batch) if x == self.config.bos_token_id]
            bos_pos.append(len(single_batch))
            bos_pos_ls.append(bos_pos)
            
        for idx,bos_ls in enumerate(bos_pos_ls):
            last_bos_pos = 0
            for jdx,bos in enumerate(bos_ls[1:]):
                expanded_mask[idx][0][last_bos_pos:bos,last_bos_pos:bos] = 1
                last_bos_pos = bos
        
        inverted_mask = 1.0 - expanded_mask
    
        return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

    def reverse_attention_mask(self,attention_mask,dtype):
        
        inverted_mask = 1.0 - attention_mask

        return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

class DualEncoder(MarianPreTrainedModel):

    def __init__(self,config,embed_tokens=None):
        super().__init__(config)
        if config.use_joint_bpe:
            self.src_encoder = TransformerEncoder(config,embed_tokens)
        else:
            self.src_encoder = TransformerEncoder(config,None)
        if self.config.use_shared_encoder: #and self.config.tm_encoder_type == 'group_attention':
            self.tm_encoder = self.src_encoder
        else:
            self.tm_encoder = TmEncoder(config,embed_tokens)
        self.init_weights()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        tm_input_ids = None,
        tm_attention_mask = None,
        # similarity_score=None,
        # static_attention_mask=None,
        # index_ls = None,
        group_attention_mask = None,
        label_attention_mask = None,
        decoder_input_ids = None,
        # group_positions = None,
        # tm_type = None,
        **kwargs,
    ):
        # BaseModelOutput:"last_hidden_state","hidden_states","attentions"

        src_encoder_outputs = self.src_encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states =  output_hidden_states,
            return_dict = return_dict,
        )
        tm_encoder_outputs = self.tm_encoder(
            input_ids = tm_input_ids,
            attention_mask = tm_attention_mask,
            group_attention_mask = group_attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states =  output_hidden_states,
            return_dict = return_dict,
            # group_positions = group_positions,
            # tm_type = tm_type,
        )
        label_enc_hidden_states = None
        label_enc_last_hidden_state = None
        label_enc_attentions = None
        if self.training and self.config.use_contrastive:
            label_encoder_outputs = self.tm_encoder(
                input_ids = decoder_input_ids,
                attention_mask = label_attention_mask,
                head_mask = head_mask,
                inputs_embeds = inputs_embeds,
                output_attentions = output_attentions,
                output_hidden_states =  output_hidden_states,
                return_dict = return_dict,
            )
            label_enc_hidden_states = label_encoder_outputs.hidden_states
            label_enc_last_hidden_state = label_encoder_outputs.last_hidden_state
            label_enc_attentions = label_encoder_outputs.attentions
        
        return TripleBaseModelOutput(
            
            src_enc_hidden_states = src_encoder_outputs.hidden_states,
            tm_enc_hidden_states = tm_encoder_outputs.hidden_states,
            label_enc_hidden_states = label_enc_hidden_states,

            src_enc_last_hidden_state= src_encoder_outputs.last_hidden_state,
            tm_enc_last_hidden_state = tm_encoder_outputs.last_hidden_state,
            label_enc_last_hidden_state = label_enc_last_hidden_state,

            src_enc_attentions = src_encoder_outputs.attentions,
            tm_enc_attentions = tm_encoder_outputs.attentions,
            label_enc_attentions = label_enc_attentions,

        )

class DualDecoder(MarianDecoder):
    
    def __init__(self,config,embed_tokens=None):
        super().__init__(config,embed_tokens)
        if config.decoder_type == 'example':
            self.layers = nn.ModuleList([ExampleDecoderLayer(config) for _ in range(config.decoder_layers)])    
        elif config.decoder_type == 'dual_cross_attention':
            # self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.decoder_layers-1)]
            #                            +[DualCrossAttentionDecoderLayer(config)])    
            self.layers = nn.ModuleList([DualCrossAttentionDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.init_weights()

    # dual cross attn
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        #encoder_hidden_states=None,
        #encoder_attention_mask=None,
        src_enc_last_hidden_state=None,
        src_attention_mask=None,
        tm_enc_last_hidden_state=None,
        tm_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        ## to be compatible with previous code:
        encoder_attention_mask = src_attention_mask
        encoder_hidden_states = src_enc_last_hidden_state
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        #############################
        #########  modify  ##########
        #############################
        if tm_enc_last_hidden_state is not None and tm_attention_mask is not None:
            tm_attention_mask = _expand_mask(tm_attention_mask,inputs_embeds.dtype,tgt_len=input_shape[-1])


        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        inputs_embeds = hidden_states
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        

        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            #if output_hidden_states:
                #all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                #encoder_hidden_states=encoder_hidden_states,
                #encoder_attention_mask=encoder_attention_mask,
                src_enc_last_hidden_state = encoder_hidden_states,
                src_attention_mask = encoder_attention_mask,
                tm_enc_last_hidden_state = tm_enc_last_hidden_state,
                tm_attention_mask = tm_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)


        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        # if not return_dict:
        #     return tuple(
        #         v
        #         for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
        #         if v is not None
        #     )
        return MyBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            decoder_input_embeds = inputs_embeds,
        )

class MarianPreTrainedModel(PreTrainedModel):
    config_class = MarianConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            #nn.init.xavier_uniform_(module.weight.data)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, MarianSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            #nn.init.xavier_uniform_(module.weight.data)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs

class MarianModel(MarianModel):

    def __init__(self,config):
        super().__init__(config)
        if config.use_joint_bpe:
            self.shared = nn.Embedding(config.trg_vocab_size,config.d_model,config.pad_token_id)
        else:
            self.shared = None
        if config.model_arch == 'baseline':
            self.encoder = TransformerEncoder(config,self.shared)
            self.decoder = TransformerDecoder(config,self.shared)
        ## if there are translation memories, use dual encoder anyway
        elif config.model_arch == 'retrieval_augmented':
            if self.shared is None:
                self.shared = nn.Embedding(config.trg_vocab_size,config.d_model,config.pad_token_id)
            self.encoder = DualEncoder(config,self.shared)
            self.decoder = DualDecoder(config,self.shared)
        else:
            raise NotImplementedError
        self.init_weights()
        
    def get_input_embeddings(self):
        return self.decoder.embed_tokens
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        similarity_score=None,
        tm_input_ids=None,
        tm_attention_mask=None,
        static_attention_mask=None,
        index_ls = None,
        group_attention_mask = None,
        label_attention_mask = None,
        # group_positions = None,
        # tm_type = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                similarity_score=similarity_score,
                tm_input_ids=tm_input_ids,
                tm_attention_mask=tm_attention_mask,
                static_attention_mask=static_attention_mask,
                index_ls = index_ls,
                group_attention_mask = group_attention_mask,
                decoder_input_ids = decoder_input_ids,
                label_attention_mask = label_attention_mask,
                # group_positions = group_positions,
                # tm_type = tm_type,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        #############################
        #########  modify  ##########
        #############################
        src_attention_mask = attention_mask
        if isinstance(encoder_outputs,TripleBaseModelOutput):
            src_enc_last_hidden_state = encoder_outputs.src_enc_last_hidden_state
            tm_enc_last_hidden_state = encoder_outputs.tm_enc_last_hidden_state
        else:
            src_enc_last_hidden_state = encoder_outputs.last_hidden_state
            tm_enc_last_hidden_state = None
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,

            #############################
            #########  modify  ##########
            #############################
            src_enc_last_hidden_state = src_enc_last_hidden_state,
            tm_enc_last_hidden_state = tm_enc_last_hidden_state,
            src_attention_mask = src_attention_mask,
            tm_attention_mask = tm_attention_mask,
            
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        
        if isinstance(encoder_outputs,TripleBaseModelOutput):
            encoder_last_hidden_state = (
                encoder_outputs.src_enc_last_hidden_state,
                encoder_outputs.tm_enc_last_hidden_state,
                encoder_outputs.label_enc_last_hidden_state,
            )
            encoder_hidden_states = (
                encoder_outputs.src_enc_hidden_states,
                encoder_outputs.tm_enc_hidden_states,
                encoder_outputs.label_enc_hidden_states,
            )
            encoder_attentions = (
                encoder_outputs.src_enc_attentions,
                encoder_outputs.tm_enc_attentions,
                encoder_outputs.label_enc_attentions,
            )
        else:
            encoder_last_hidden_state = (encoder_outputs.last_hidden_state,)
            encoder_hidden_states = (encoder_outputs.hidden_states,)
            encoder_attentions = (encoder_outputs.attentions,)
        
        if 'decoder_input_embeds' in decoder_outputs:
            decoder_input_embeds = decoder_outputs.decoder_input_embeds
        else:
            decoder_input_embeds = None
        return MySeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attentions=encoder_attentions,
            decoder_input_embeds = decoder_input_embeds,
        )

class MarianMTModel(MarianMTModel):
    _keys_to_ignore_on_save = [] # save sinosidual pos weight for convenience
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.model = MarianModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, config.trg_vocab_size)))
        self.lm_head = nn.Linear(config.d_model, config.trg_vocab_size, bias=False)
        if self.config.use_copy:
            self.copy_module = CopyModule(config.d_model)
        if self.config.use_contrastive:
            self.contrastive_loss_module = ContrastiveLoss(config)
        self.label_smoother = MyLabelSmoother()
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        similarity_score=None,
        static_attention_mask = None,
        tm_input_ids=None,
        tm_attention_mask=None,
        index_ls = None,
        group_attention_mask = None,
        label_attention_mask = None,
        # group_positions = None,
        # tm_type = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            similarity_score=similarity_score,
            tm_input_ids=tm_input_ids,
            tm_attention_mask=tm_attention_mask,
            static_attention_mask=static_attention_mask,
            index_ls = index_ls,
            group_attention_mask = group_attention_mask,
            label_attention_mask = label_attention_mask,
            # group_positions = group_positions,
            # tm_type = tm_type,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        log_prob = None
        if hasattr(self,"copy_module"):
            log_prob = self.copy_module(
                decoder_attention = outputs.cross_attentions[-1],
                decoder_last_hidden_state = outputs.last_hidden_state,
                enc_input_ids = tm_input_ids,
                logits = lm_logits,
                decoder_input_embeds = outputs.decoder_input_embeds,
                encoder_last_hidden_state = outputs.encoder_last_hidden_state[1],
            )
    
        loss = None
        cross_entropy_loss = None
        contrastive_loss = None
        if labels is not None:
            cross_entropy_loss = self.label_smoother(scores = log_prob,labels = labels) if log_prob is not None else self.label_smoother(logits=lm_logits,labels=labels)
            if hasattr(self,"contrastive_loss_module") and self.training:
                contrastive_loss = self.contrastive_loss_module(
                    encoder_hidden_states = outputs.encoder_hidden_states,
                    tm_input_ids = tm_input_ids,
                    src_attention_mask = attention_mask,
                    tm_attention_mask = tm_attention_mask,
                    label_attention_mask = label_attention_mask,
                )
                loss = cross_entropy_loss + contrastive_loss
            else:
                loss = cross_entropy_loss

        return MySeq2SeqLMOutput(
            contrastive_loss = contrastive_loss.item() if contrastive_loss is not None else None,
            cross_entropy_loss = cross_entropy_loss.item() if contrastive_loss is not None else None,
            scores = log_prob,
            loss = loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        tm_input_ids = None,
        tm_attention_mask = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)
        
        if tm_input_ids is not None:
            model_kwargs['tm_input_ids'] = tm_input_ids.index_select(0,expanded_return_idx)

        if tm_attention_mask is not None:
            model_kwargs['tm_attention_mask'] = tm_attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            for name in ["last_hidden_state","src_enc_last_hidden_state","tm_enc_last_hidden_state"]:
                if name in encoder_outputs:
                    encoder_outputs[name] = encoder_outputs[name].index_select(
                        0,expanded_return_idx.to(encoder_outputs[name].device)
                    )
            
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        tm_attention_mask=None,
        tm_input_ids=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        #assert use_cache is True
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "tm_attention_mask":tm_attention_mask,
            "tm_input_ids":tm_input_ids,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    # huggingface 
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            #torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need
                
            #############################
            #########  modify  ##########
            #############################
            if not self.config.use_copy:
                next_token_logits = outputs.logits[:, -1, :]

                # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
                # cannot be generated both before and after the `nn.functional.log_softmax` operation.
                next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
                next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * num_beams, vocab_size)
            else:
                next_token_scores = outputs.scores[:,-1,:]


            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
