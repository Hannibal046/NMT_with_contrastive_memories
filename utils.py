#from comet_ml import Experiment
import pickle
from CONSTANT import *
import re
from transformers import TrainerCallback,TrainingArguments,TrainerState,TrainerControl

def debpe(bpe):
    return re.sub(r'(@@ )|(@@ ?$)', '', bpe)

def get_similarity_score_between_two_sentences(sent1,sent2):
    """
    sent1,2:str
    """
    import pylev
    sent1 = sent1.split()
    sent2 = sent2.split()
    return 1- (pylev.levenshtein(sent1,sent2)/max(len(sent1),len(sent2)))

def train_tokenizer(data_ls=None,
                    dict_key=None,
                    save_dir=None,
                    vocab_size=20000,
                    normalization = ['Lowercase','NFD','StripAccents'],
                    pre_tokenizers_str=['white_space'],
                    special_tokens = ["[UNK]", "[PAD]","[EOS]","[BOS]"]):
    import json
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers import pre_tokenizers
    from tokenizers.pre_tokenizers import Whitespace,Digits
    from tokenizers.processors import TemplateProcessing
    from tokenizers import normalizers
    from tokenizers.normalizers import Lowercase, NFD, StripAccents

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=vocab_size,
                         special_tokens=special_tokens)
    
    # normalization
    normalization_ls = []
    for norm in normalization:
        if norm == 'Lowercase':normalization_ls.append(Lowercase())
        elif norm == 'NFD':normalization_ls.append(NFD())
        elif norm == 'StripAccents':normalization_ls.append(StripAccents())   
    tokenizer.normalizer = normalizers.Sequence(normalization_ls)
    
    # pre_tokenize
    pre_tokenizer_ls = []
    for _pre_tokenizer in pre_tokenizers_str:
        if _pre_tokenizer == 'white_space':pre_tokenizer_ls.append(Whitespace())
        elif _pre_tokenizer == 'digits':pre_tokenizer_ls.append(Digits(individual_digits=True))
    pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizer_ls)
    tokenizer.pre_tokenizer = pre_tokenizer
    
    # post_process
    tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                            ("[BOS]", 3), 
                            ("[EOS]", 2)
                            ]
            )
    data = []
    for file in data_ls:
        data.extend([json.loads(x) for x in open(file).readlines()])
    data = [x[dict_key] for x in data]
    
    tokenizer.train_from_iterator(data, trainer=trainer)

    
    #tokenizer.save(save_dir)
    return tokenizer

def check_args(data_args,model_args,training_args):
    model_args.max_src_len = data_args.max_src_len
    model_args.max_trg_len = data_args.max_trg_len

def save_src(output_dir):
    pass
def get_json_line(file):
    import json
    return [json.loads(x) for x in open(file).readlines()]

def dump_json_file(file,path):
    import json
    from tqdm import tqdm
    with open(path,'w') as f:
        for instance in tqdm(file):
            f.write(json.dumps(instance)+'\n')
    return 

def split_train_dev_test(file,direction='ende'):
    
    if direction is 'ende' or direction is 'deen':
        train_len = ENDE_TRAIN_LEN
        dev_len = ENDE_DEV_LEN
        test_len = ENDE_TEST_LEN
    else:
        train_len = ENES_TRAIN_LEN
        dev_len = ENES_DEV_LEN
        test_len = ENES_TEST_LEN
    ori_pkl = pickle.load(open(file,'rb'))
    assert isinstance(ori_pkl,list)
    ret = {
        'train':ori_pkl[:train_len],
        'dev':ori_pkl[train_len:dev_len + train_len],
        'test':ori_pkl[-test_len:]
        }
    pickle.dump(ret,open(file,'wb'))


def get_edit_sim(src,tm):
    
    import editdistance
    a = src.split()
    b = tm.split()
    edit_distance = editdistance.eval(a, b)

    edit_sim = 1 - edit_distance / max(len(src), len(tm))

    return edit_sim

def get_cos_sim(src,tm):
    import torch.nn
    sim_fn = torch.nn.CosineSimilarity(dim=-1)
    return sim_fn(src,tm)

def get_json(f):
    import json
    return [json.loads(x) for x in open(f).readlines()]