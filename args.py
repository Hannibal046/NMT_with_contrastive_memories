import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '1'
from dataclasses import dataclass,field
from transformers import TrainingArguments,MarianConfig
from transformers.configuration_utils import PretrainedConfig
from typing import Optional,List
import CONSTANT
from random import randint

@dataclass
class DataArgs:

    exp_desc = "demo"
    dataset_dir_prefix:str = '../data/'
    dataset_path:str = 'jrc_joint/ende'
    train_file:str = 'train.json'
    dev_file:str = 'dev.json'
    test_file:str = 'test.json'
    tm_size = 5
    tm_path = '../data/retrieval/ende/src_editdis_alpha_0.7.pkl'
    #retrieval_type = 'src_retrieval'
    cache_file = '../.cache/data.pkl'
    use_cache:bool = False
    train_tokenizer = False
    
    # should chang model args counterpart
    max_src_len:int = 250
    max_trg_len:int = 118
    min_trg_len:int = 3
    src_vocab_file:str = None
    trg_vocab_file:str = None
    src:str = ''
    trg:str = ''
    max_train_samples = None
    max_test_samples = None
    max_dev_samples = None
    use_sim_scores = False

    def __post_init__(self):
        self.train_file = os.path.join(self.dataset_dir_prefix,
                                       self.dataset_path,
                                       self.train_file)
        self.dev_file = os.path.join(self.dataset_dir_prefix,
                                       self.dataset_path,
                                       self.dev_file)
        self.test_file = os.path.join(self.dataset_dir_prefix,
                                       self.dataset_path,
                                       self.test_file)
        self.src = self.dataset_path.split('/')[-1][:2]
        self.trg = self.dataset_path.split('/')[-1][2:]
        if not self.train_tokenizer:
            self.src_vocab_file = os.path.join(
                self.dataset_dir_prefix,self.dataset_path,'src.vocab'
            )
            self.trg_vocab_file = os.path.join(
                self.dataset_dir_prefix,self.dataset_path,'tgt.vocab'
            )

class MarianConfig(MarianConfig):
    def __init__(
        self,
        decoder_type = 'dual_cross_attention',
        tm_encoder_type = 'group_attention',
        model_arch = 'retrieval_augmented', 
        use_copy = True,
        
        use_contrastive = True,
        contrastive_temperature =0.15,
        contrastive_lambda = 1,
        pooler_type = 'cls_mlp',
        contrastive_loss_balance = False,

        output_hidden_states = False,
        output_attentions = True,
        max_src_len = 250,
        max_tm_len = 500,
        max_trg_len = 118, # for decoding 
        min_trg_len = 3,
        num_beams = 5,
        src_vocab_size = 0,
        trg_vocab_size = 0,
        max_position_embeddings=1024,
        
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        encoder_layerdrop=0.0,

        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        decoder_layerdrop=0.0,

        tm_encoder_layers = 6,
        tm_encoder_ffn_dim=2048,
        tm_encoder_attention_heads=8,
        tm_encoder_dropout=0.0,

        use_cache=True,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=512,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=CONSTANT.BOS,
        classifier_dropout=0.0,
        scale_embedding=True,
        gradient_checkpointing=False,
        pad_token_id=CONSTANT.PAD,
        eos_token_id=CONSTANT.EOS,
        forced_eos_token_id=CONSTANT.EOS,
        bos_token_id = CONSTANT.BOS,
        tm_size = 0,
        use_shared_encoder = False,
        use_joint_bpe = False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        self.output_hidden_states = output_hidden_states
        self.contrastive_loss_balance = contrastive_loss_balance
        self.pooler_type = pooler_type
        self.use_shared_encoder = use_shared_encoder
        self.use_joint_bpe = use_joint_bpe
        self.tm_size = tm_size
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_lambda = contrastive_lambda
        self.use_contrastive = use_contrastive
        self.decoder_type = decoder_type
        self.tm_encoder_type = tm_encoder_type    
        self.tm_encoder_layers = tm_encoder_layers
        self.tm_encoder_ffn_dim = tm_encoder_ffn_dim
        self.tm_encoder_attention_heads = tm_encoder_attention_heads
        self.tm_encoder_dropout = tm_encoder_dropout
        self.use_copy = use_copy
        self.output_attentions = output_attentions
        self.model_arch = model_arch
        self.bos_token_id = bos_token_id
        self.num_beams = num_beams
        self.max_length = max_trg_len
        self.max_src_len = max_src_len
        self.max_tm_len = max_tm_len
        self.max_trg_len = max_trg_len
        self.min_trg_len = min_trg_len
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        #self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True


@dataclass
class TrainingArgs(TrainingArguments):
    
    output_dir: str = '../results/jrc/ende/dual'
    overwrite_output_dir: bool = False
    multiple_loss:bool = False
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = True
    predict_with_generate = True

    evaluation_strategy:str = 'steps'
    eval_steps: int = 2000
    
    save_strategy:str = "steps"
    save_steps: int = 1000
    
    logging_strategy: str = 'steps'
    logging_first_step: bool = True
    logging_steps: int = 100
    
    num_train_epochs: float = 20
    max_steps:int = 50000
    per_device_train_batch_size: int = 10000 # tokens
    per_device_eval_batch_size: int = 20  # sentences
    #eval_accumulation_steps
    gradient_accumulation_steps: int = 1

    max_grad_norm: float = 1.0
    warmup_steps: int = 8000

    no_cuda: bool = False
    seed: int = 42
    fp16: bool = True
    fp16_opt_level: str = '01'
    fp16_backend: str = 'auto'
    group_by_length = False
    dataloader_num_workers: int = 0
    run_name: Optional[str] = None
    save_total_limit:int =  2
    remove_unused_columns: Optional[bool] = False
    load_best_model_at_end: Optional[bool] = True
    metric_for_best_model: Optional[str] = 'bleu'
    greater_is_better: Optional[bool] = True
    label_smoothing_factor: float = 0
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})

def check_args(data_args,model_args,training_args):
    
    data_args.max_src_len = model_args.max_src_len
    data_args.max_trg_len = model_args.max_trg_len
    data_args.min_trg_len = model_args.min_trg_len
    data_args.train_batch_size = training_args.per_device_train_batch_size
    data_args.eval_batch_size = training_args.per_device_eval_batch_size
    model_args.tm_size = data_args.tm_size
    if model_args.use_copy:
        assert data_args.tm_size > 0
    else:
        model_args.output_attentions = False
    if model_args.tm_encoder_type == 'dual_self_attention':
        data_args.use_sim_scores = True
    if model_args.model_arch == 'baseline':
        data_args.tm_size = 0
        model_args.use_copy = False
        model_args.use_contrastive = False
    if model_args.use_contrastive:
        training_args.multiple_loss = True
        model_args.output_hidden_states = True
    training_args.output_dir = '/'.join(training_args.output_dir.split('/')[:-1] + [str(randint(0,20000))])
    if "joint" in data_args.train_file:
        model_args.use_joint_bpe = True
        model_args.use_shared_encoder = True
    return data_args,model_args,training_args


class BenConfig(PretrainedConfig):

    def __init__(self,
        

        unk_token_id:int = CONSTANT.UNK,
        pad_token_id:int = CONSTANT.PAD,
        eos_token_id:int = CONSTANT.EOS,
        bos_token_id:int = CONSTANT.BOS,

        src_vocab_size:int = 7000,
        trg_vocab_size:int = 6000,
        
        num_beams = 5,
        max_src_len = 250,
        max_tm_len = 500,
        max_trg_len = 118, # for decoding 
        min_trg_len = 3,

        
        tm_enc_n_layers = 1,
        tm_enc_n_heads = 16,
        tm_enc_pf_dim = 2048,
        tm_enc_dropout = 0.1,
        tm_enc_layerdrop = 0.0,

        src_enc_n_layers:int = 6,
        src_enc_pf_dim:int = 2048,
        src_enc_n_heads:int = 8,
        src_enc_layerdrop:float = 0.0,
        src_enc_dropout = 0.1,
        
        dec_n_layers:int = 6,
        dec_pf_dim:int = 2048,
        dec_n_heads:int = 8,
        dec_layerdrop:float = 0.0,
        dec_dropout:int = 0.1,
        cross_attn = False,

        enable_copy:bool = False,
        activation_fn:str = "relu",
        d_model:int = 512,
        dropout:float = 0.1,
        attention_dropout:float = 0.0,
        activation_dropout:float = 0.0,
        init_std:float = 0.02,
        scale_embedding:bool = True,
    ):
        super().__init__(

        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        is_encoder_decoder=True,
        decoder_start_token_id=bos_token_id,
        forced_eos_token_id=eos_token_id,
        )
        self.enable_copy = enable_copy
        self.unk_token_id:int =unk_token_id
        self.pad_token_id:int =pad_token_id
        self.eos_token_id:int =eos_token_id
        self.bos_token_id:int =bos_token_id
        self.src_vocab_size:int =src_vocab_size
        self.trg_vocab_size:int =trg_vocab_size
        self.num_beams:int =num_beams
        self.max_src_len:int =max_src_len
        self.max_tm_len:int =max_tm_len
        self.max_trg_len:int =max_trg_len
        self.min_trg_len:int =min_trg_len
        self.max_length = self.max_trg_len
        self.min_length = self.min_trg_len
        self.tm_enc_n_layers:int =tm_enc_n_layers
        self.tm_enc_n_heads:int =tm_enc_n_heads
        self.tm_enc_pf_dim:int =tm_enc_pf_dim
        self.tm_enc_dropout:float =tm_enc_dropout
        self.tm_enc_layerdrop:float =tm_enc_layerdrop
        self.src_enc_n_layers:int =src_enc_n_layers
        self.src_enc_pf_dim:int =src_enc_pf_dim
        self.src_enc_n_heads:int =src_enc_n_heads
        self.src_enc_layerdrop:float =src_enc_layerdrop
        self.src_enc_dropout = src_enc_dropout
        self.dec_n_layers:int =dec_n_layers
        self.dec_pf_dim:int =dec_pf_dim
        self.dec_n_heads:int =dec_n_heads
        self.dec_layerdrop:float =dec_layerdrop
        self.dec_dropout:int =dec_dropout
        self.cross_attn:bool =cross_attn
        self.activation_fn:str =activation_fn
        self.d_model:int =d_model
        self.dropout:float =dropout
        self.attention_dropout:float =attention_dropout
        self.activation_dropout:float =activation_dropout
        self.init_std:float =init_std
        self.scale_embedding:bool =scale_embedding
