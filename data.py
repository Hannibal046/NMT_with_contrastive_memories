from copy import copy,deepcopy
from torch.distributed.distributed_c10d import group
from tqdm import tqdm
import datasets
import torch
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding
import CONSTANT
import json
import re
import random
import pickle
from utils import debpe
#from bm25 import get_unedited_words

def get_datasets(data_args,src_tokenizer,trg_tokenizer):
    
    if data_args.use_cache:
        print(f'Reusing Cached TokenDatasets in {data_args.cache_file}')
        return pickle.load(open(data_args.cache_file,'rb'))

    
    
    ret = [
        TokenBatchDataset(data_args.train_file,
                          data_args.src,
                          data_args.trg,
                          src_tokenizer,
                          trg_tokenizer,
                          data_args.train_batch_size,
                          is_training=True,
                          tm_size = data_args.tm_size,
                          #retrieval_type=data_args.retrieval_type,
                          use_sim_scores = data_args.use_sim_scores,
                          tm_path = data_args.tm_path),
        
        SentenceBatchDataset(data_args.dev_file,
                          data_args.src,
                          data_args.trg,
                          src_tokenizer,
                          trg_tokenizer,
                          data_args.eval_batch_size,
                          tm_size = data_args.tm_size,
                          #retrieval_type=data_args.retrieval_type,
                          use_sim_scores = data_args.use_sim_scores,
                          tm_path = data_args.tm_path),
        
        SentenceBatchDataset(data_args.test_file,
                          data_args.src,
                          data_args.trg,
                          src_tokenizer,
                          trg_tokenizer,
                          data_args.eval_batch_size,
                          tm_size = data_args.tm_size,
                          #retrieval_type=data_args.retrieval_type,
                          use_sim_scores = data_args.use_sim_scores,
                          tm_path = data_args.tm_path)
        ]
    #pickle.dump(ret,open(data_args.cache_file,'wb'))
    return ret

def sentence_batch_collate_fn(samples):
        
        # samples: list
        # sample: [src,trg,tm,similarity_scores]

        src = [sample[0][:-1] for sample in samples] # [:-1] delete eos
        trg = [sample[1] for sample in samples]
        max_src_len = max(len(x) for x in src)
        max_trg_len = max(len(x) for x in trg)
        
        # pad src
        src_mask  = [[1]*len(x) + [0]*(max_src_len-len(x)) for x in src]
        src = [x+[CONSTANT.PAD]*(max_src_len-len(x)) for x in src]
        # pad trg
        trg_out = [x[1:]+[-100]*(max_trg_len-1-len(x[1:])) for x in trg] # no bos
        trg_in = [x[:-1]+[CONSTANT.PAD]*(max_trg_len-1-len(x[:-1])) for x in trg] # no eos
        label_mask = [[1]*len(x[:-1]) + [0]*(max_trg_len-1-len(x[:-1])) for x in trg]

        has_tm = len(samples[0]) >= 3
        has_sim = len(samples[0]) == 4
        if not has_tm:
            return {
                "input_ids":torch.tensor(src),
                "attention_mask":torch.tensor(src_mask),
                "labels":torch.tensor(trg_out),
                "decoder_input_ids":torch.tensor(trg_in)
            }
        else:
            tm = [sample[2] for sample in samples]
            max_tm_len = max(len(x) for x in tm)
            tm_mask = [[1]*len(x)+[0]*(max_tm_len-len(x)) for x in tm]
            tm = [x+[CONSTANT.PAD]*(max_tm_len-len(x)) for x in tm]
            group_attention_mask = None
            group_attention_mask = get_group_attention_mask(tm) # bs,tm_len,tm_len

            if not has_sim:
                return {
                    "input_ids":torch.tensor(src),
                    "attention_mask":torch.tensor(src_mask),
                    "labels":torch.tensor(trg_out),
                    "decoder_input_ids":torch.tensor(trg_in),
                    "tm_input_ids":torch.tensor(tm),
                    "tm_attention_mask":torch.tensor(tm_mask),
                    "label_attention_mask":torch.tensor(label_mask),
                    "group_attention_mask":group_attention_mask,

                }
            else:

                similarity_score = [sample[3] for sample in samples]
                max_tm_cnt = max(len(x) for x in similarity_score) # 5 for  default
            
                index_lls = []
                for t in tm:
                    temp_ls = []
                    for pos,token_id in enumerate(t):
                        if int(token_id) == CONSTANT.BOS:
                            temp_ls.append(pos)
                    assert len(temp_ls) > 0
                    index_lls.append(temp_ls)

                static_attention_mask = [[1]*len(index_ls) + [0]*(max_tm_cnt-len(index_ls)) for index_ls in index_lls]

                # pad similarity_score
                similarity_score_matrixs = []
                for idx,sim_score in enumerate(similarity_score):

                    similarity_score_matrix = torch.zeros((max_tm_cnt,max_tm_cnt),dtype = torch.float32)
                    similarity_score_matrix[:len(sim_score),:len(sim_score)] = torch.tensor(sim_score)
                    similarity_score_matrixs.append(similarity_score_matrix)

                return {
                    "input_ids":torch.tensor(src),
                    "attention_mask":torch.tensor(src_mask),
                    "labels":torch.tensor(trg_out),
                    "decoder_input_ids":torch.tensor(trg_in),
                    "tm_input_ids":torch.tensor(tm),
                    "tm_attention_mask":torch.tensor(tm_mask),
                    "similarity_score":torch.stack(similarity_score_matrixs,dim=0),
                    "static_attention_mask":torch.tensor(static_attention_mask),
                    "index_ls":index_lls,
                }

class BatchDataset(torch.utils.data.Dataset):

    def __init__(self,path,src_lang,trg_lang,src_tokenizer,trg_tokenizer,batch_size,
                 is_training = False,
                 tm_size=5,
                 #retrieval_type='src_retrieval',
                 use_sim_scores=False,
                 tm_path = None,):    
        # print("Generating dataset...")
        self.batch_size = batch_size
        self.has_tm = False
        self.split = path.split('/')[-1].split('.')[0]
        dataset = [json.loads(x) for x in open(path).readlines()]#[:100]
        print(path)
        print("Dataset Samples:",len(dataset))
        if tm_size > 0:
            
            tm_lls = pickle.load(open(tm_path,'rb'))[self.split]#[:100]
            print(tm_path,len(tm_lls))
            id2text_path = '/'.join(path.split('/')[:-1] + ['id2text.pkl'])
            id2text = pickle.load(open(id2text_path,'rb'))
            self.has_tm = True

            # filter out abnormal datapoint
            # if is_training: 
            dataset = [x for idx,x in enumerate(dataset) if len(tm_lls[idx])>=tm_size] # for train/dev/test
            tm_lls = [x for x in tm_lls if len(x)>=tm_size]
            print("Dataset Samples After filtering:",len(dataset))
            
            ## get raw tm 
            raw_tm = []
            for tm_ls in tm_lls:
                temp_ls = []
                for iid in tm_ls[:tm_size]:
                    tm = id2text[iid][trg_lang] #if retrieval_type == 'src_retrieval' else id2text[iid][trg_lang]
                    temp_ls.append(tm)
                raw_tm.append(temp_ls)

            similarity_scores = None
            # get similarity matrix
            if use_sim_scores:
                similarity_scores = 'src_similarity_scores' if retrieval_type == 'src_retrieval' else 'trg_similarity_scores' 
                similarity_scores = [x[similarity_scores] for x in dataset]
                for idx,sim_scores in enumerate(similarity_scores):
                    sim_scores = [x[:tm_size] for x in sim_scores]
                    similarity_scores[idx] = sim_scores[:tm_size]
            
            # get src trg tm encoding
            src = src_tokenizer.encode_batch([x[src_lang] for x in tqdm(dataset,desc='src tokenization...')],enable_truncation=True)
            trg = trg_tokenizer.encode_batch([x[trg_lang] for x in tqdm(dataset,desc='trg tokenization...')],enable_truncation=True if is_training else False)
            tm = [trg_tokenizer.encode_batch(x,enable_truncation=True,max_length = len(src[idx])) for idx,x in tqdm(enumerate(raw_tm),desc='tm tokenization...',total=len(raw_tm))]
            
            # concat tms together
            for idx,t in enumerate(tm):
                ret = []
                for sentence in t:
                    ret.extend(sentence[:-1]) # delete eos
                tm[idx] = ret
            # self.tm[idx] = [<bos> this is good <bos> hello]
            
        
            if similarity_scores:
                self.src_trg = [[src[idx],trg[idx],tm[idx],similarity_scores[idx]] for idx in range(len(src))]
            else:
                self.src_trg = [[src[idx],trg[idx],tm[idx]] for idx in range(len(src))]
            self.tm_sizes = np.array([len(x) for x in tm])
        else:
            src = src_tokenizer.encode_batch([x[src_lang] for x in dataset],enable_truncation=True)
            trg = trg_tokenizer.encode_batch([x[trg_lang] for x in dataset],enable_truncation=True if is_training else False)

            self.has_tm = False

            self.src_trg = [[src[idx],trg[idx]] for idx in range(len(src))]
        
        self.src_sizes = np.array([len(x) for x in src])
        self.trg_sizes = np.array([len(x) for x in trg])
        
    
    def __len__(self):
        return len(self.src_trg)
    
    def __getitem__(self,idx):
        return self.src_trg[idx]
    
class SentenceBatchDataset(BatchDataset):
    def __init__(self,path,src_lang,trg_lang,src_tokenizer,trg_tokenizer,batch_size,is_training=False,tm_size=5,use_sim_scores=False,tm_path=None,):
        super().__init__(path,src_lang,trg_lang,src_tokenizer,trg_tokenizer,batch_size,is_training,tm_size,use_sim_scores,tm_path = tm_path)
    def __getitem__(self,idx):
        return self.src_trg[idx]

class TokenBatchDataset(BatchDataset):

    def __init__(self,path,src_lang,trg_lang,src_tokenizer,trg_tokenizer,batch_size,is_training = True,tm_size=5,use_sim_scores=False,tm_path=None):
        super().__init__(path,src_lang,trg_lang,src_tokenizer,trg_tokenizer,batch_size,is_training,tm_size,use_sim_scores,tm_path = tm_path)
    
        
        indices = np.arange(len(self.src_trg))
        indices = indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        
        num_tokens, batch = 0, []
        self.batches = []
        for i in indices:
            if not self.has_tm:
                num_tokens += max(self.src_sizes[i], self.trg_sizes[i])
                if num_tokens > self.batch_size:
                    self.batches.append(batch)
                    num_tokens, batch = max(self.src_sizes[i], self.trg_sizes[i]), [i]
                else:
                    batch.append(i)
            else:
                num_tokens += max(self.src_sizes[i], self.trg_sizes[i],self.tm_sizes[i])
                if num_tokens > self.batch_size:
                    self.batches.append(batch)
                    num_tokens, batch = max(self.src_sizes[i], self.trg_sizes[i],self.tm_sizes[i]), [i]
                else:
                    batch.append(i)
        self.batches.append(batch)
        
        self.batches_idx = copy(self.batches)
        for idx,batch in tqdm(enumerate(self.batches),desc='Generating Training Dataset...',total=len(self.batches)):
            batch_with_content = [self.src_trg[i] for i in batch]
            self.batches[idx] = sentence_batch_collate_fn(batch_with_content)
        self.len_dataset = len(self.src_trg)
        del self.src_trg
        print("Average Senteces in One Batch:",sum(len(x) for x in self.batches_idx)/len(self.batches_idx))
        
    def __getitem__(self,index):
        return self.batches[index]
    def __len__(self):
        return self.len_dataset

class Tokenizer:
    def __init__(self,vocab_file,max_length=147):
        self._id2token = ['UNK','PAD','EOS','BOS']
        self.special_token_ids = [0,1,2,3]
        for line in open(vocab_file).readlines():
            token,cnt = line.rstrip('\n').split('\t')
            cnt = int(cnt)
            self._id2token.append(token)
        self._token2id = dict(zip(self._id2token,range(len(self._id2token))))
        
        self.unk_token = 'UNK'
        self.pad_token = 'PAD'
        self.eos_token = 'EOS'
        self.bos_token = 'BOS'
        
        self.unk_token_id = CONSTANT.UNK
        self.pad_token_id = CONSTANT.PAD
        self.eos_token_id = CONSTANT.EOS
        self.bos_token_id = CONSTANT.BOS
        self.max_length = max_length
        self.vocab_size = len(self._id2token)
    
    def enable_truncation(self,max_length):
        self.max_length = max_length
    
    def id2token(self,x):
        return self._id2token[x]
    
    def token2id(self,token):
        return self._token2id.get(token,self.unk_token_id)
    
    def encode_batch(self,samples,enable_truncation=True,max_length=None):
        ret = []
        for sample in samples:
            if enable_truncation:
                if max_length is not None:max_length = min(max_length,self.max_length)
                else:max_length = self.max_length
                sample = [self.bos_token_id] + [self.token2id(x) for x in sample.split()[:max_length]] + [self.eos_token_id]
            else:
                sample = [self.bos_token_id] + [self.token2id(x) for x in sample.split()] + [self.eos_token_id]
            ret.append(sample)
        return ret

    def decode_batch(self,samples,skip_special_tokens=True,remain_bpe=False):
        # samples: tensor
        if isinstance(samples,torch.Tensor):
            samples = samples.tolist()
        ret = []
        for sample in samples:
            if skip_special_tokens:
                sample = [self.id2token(x) for x in sample if x not in self.special_token_ids]
            else:
                sample = [self.id2token(x) for x in sample ]
            ret.append(sample)
        if not remain_bpe:
            ret = [re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(o)) for o in ret]
        else:
            ret = [' '.join(o) for o in ret]
        return ret

class TokenBatchDataLoader():
    def __init__(self,dataset,batch_size,shuffle=True):
        self.shuffle = shuffle
        self.batches = dataset.batches
        self.dataset = dataset
        self.batch_size = batch_size
    def __iter__(self):
        #batches = deepcopy(self.batches)
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            # yield deepcopy(batch)
            yield batch
    def __len__(self):
        return len(self.batches)

def get_group_attention_mask(tms):

    # tm: list of padded tm
    bs = len(tms)
    tm_len = len(tms[0]) # already padded
    bos_token_id = CONSTANT.BOS
    ret = []
    for idx,tm in enumerate(tms):
        temp = [[0]*tm_len for _ in range(tm_len)]
        bos_pos = [jdx for jdx,x in enumerate(tm) if x == bos_token_id]
        bos_pos.append(len(tm))
        one_hot = [1 if x in bos_pos else 0 for x in range(tm_len)]
        for bos in bos_pos[:-1]:
            temp[bos] = one_hot
        last_bos_pos = 0
        for jdx,bos in enumerate(bos_pos[1:]):
            one_hot  = [1 if last_bos_pos <= x <bos else 0 for x in range(tm_len)]
            #print(one_hot)
            for i in list(range(tm_len))[last_bos_pos:bos]:
                if i not in bos_pos:
                    temp[i] = one_hot
            last_bos_pos = bos
        ret.append(temp)
    return ret

def get_group_positions(index_lls,max_tm_len):
    def merge(lol):
        ret = []
        for l in lol:
            ret.extend(l)
        return ret
    ret = []
    for index_ls in index_lls:
        diffs = [index_ls[i+1]-index_ls[i] for i in range(len(index_ls)-1)]
        diffs.append(max_tm_len-index_ls[-1])
        temp = merge([list(range(diff)) for diff in diffs])
        ret.append(temp)
    return ret

def get_bos_pos_ls(tms):
    ret = []
    for tm in tms:
        ret.append([idx for idx,x in enumerate(tm) if x == CONSTANT.BOS])
    return ret

def get_tm_type(tms):
    ret = []
    for tm in tms:
        temp = []
        tm_type = -1
        for token_id in tm:
            if token_id == CONSTANT.BOS:tm_type += 1
            temp.append(tm_type)
        ret.append(temp)
    return ret


            
