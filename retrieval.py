from utils import debpe
from utils import get_cos_sim,get_edit_sim
import torch
import pickle
import numpy as np
from tqdm import tqdm
import json

def get_json(f):
    return [json.loads(x) for x in open(f).readlines()]

all_ls = get_json('../data/jrc/ende/all.json')
id2text = {x['id']:x['en'] for x in all_ls}
id2tensor = np.load('../data/jrc/ende/simcse_all.npy')
id2tensor = {all_ls[idx]['id']:torch.tensor(id2tensor[idx]) for idx in range(len(all_ls))}
retrieval_ls = pickle.load(open('../data/jrc/ende/src_retrieval_100.pkl','rb'))

for alpha in [x/10 for x in list(range(10))]:
    tm_size = 5
    sorted_retrieval_ls = []
    sim_fn = torch.nn.CosineSimilarity(dim=-1)
    beta = 1-alpha
    for sample,retrieval in tqdm(zip(all_ls,retrieval_ls),total=len(retrieval_ls)):
        if len(retrieval) < tm_size:
            sorted_retrieval_ls.append(retrieval)
            continue

        iid = sample['id']
        src = debpe(sample['en'])
        src_tensor = id2tensor[iid]
        # filter exact match
        retrieval = [x for x in retrieval if debpe(all_dict[x]).strip() != src]

        # ret = [0,]

        # get first by minimun edit distance
        # max_edit_distance = 0
        # for retri_id in retrieval:
        #     retri = all_dict[retri_id]
        #     edit_distance = get_unedited_words(src,retri)
        #     if edit_distance > max_edit_distance:
        #         max_edit_distance = edit_distance
        #         ret[0] = retri_id
        # retrieval.remove(ret[0])
        ret  = []
        # get the rest 
        while len(ret) < tm_size:
            max_sim = 0
            ret_tensor =  [id2tensor[x] for x in ret]
            for retri in retrieval:
                
                sim = alpha * sim_fn(id2tensor[retri],src_tensor)
                sim = beta * sim_fn(
                    torch.stack([*ret_tensor,id2tensor[retri]],dim=0).mean(dim=0),
                    src_tensor
                )
                if sim > max_sim:
                    max_sim = sim
                    max_sim_id = retri
            ret.append(max_sim_id)
            retrieval.remove(max_sim_id)
        sorted_retrieval_ls.append(ret)
    pickle.dump(sorted_retrieval_ls,open('ende_retrieval_pure_embed_alpha'+str(alpha)+'.pkl','wb'))


