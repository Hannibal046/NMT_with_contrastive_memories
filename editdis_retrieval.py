from utils import (
    debpe,
    get_cos_sim,
    get_edit_sim,
    get_json,
)
import torch
import pickle
from tqdm import tqdm


# parameters
translation_direction = 'ende'
src_lang = translation_direction[:2]
trg_lang = translation_direction[-2:]
tm_size = 5
alpha_list = [1,1.1,1.2,1.3,1.4,1.5,1.6]

## load file
all_ls = get_json('../data/jrc/'+translation_direction+'/all.json')
id2text = {x['id']:debpe(x[src_lang]) for x in all_ls}
bm25_dict = pickle.load(open('../data/retrieval/'+translation_direction+'/bm25_src.pkl','rb'))
bm25_ls = [*bm25_dict['train'],*bm25_dict['dev'],*bm25_dict['test']]


def get_tm_overlap(tm1,tm2):
    assert len(tm1) == len(tm2)
    cnt = 0
    for tm_ls_1,tm_ls_2 in zip(tm1,tm2):
        if tm_ls_1 and tm_ls_2:
            overlap = len(set(tm_ls_1[:tm_size]) & set(tm_ls_2[:tm_size]))/len(tm_ls_2)
            cnt += overlap
    return cnt/len(tm1)

def merge_dict_to_list(d):
    ret = []
    for k,v in d.items():
        ret.extend(v)
    return ret
# src_editdis = merge_dict_to_list(pickle.load(open('../data/retrieval/ende/src_editdis.pkl','rb')))



for alpha in alpha_list:
    sorted_retrieval_ls = []
    for idx,retrieval_ls in tqdm(enumerate(bm25_ls),total=len(bm25_ls)):
        src = debpe(all_ls[idx][src_lang])
        candidate_ls = [(x,id2text[x]) for x in retrieval_ls if int(x.split('_')[-1]) != idx]
        candidata_ls = [x for x in candidate_ls if x[1].strip() != src.strip()]
        
        
        if len(candidate_ls) < tm_size:
            sorted_retrieval_ls.append([x[0] for x in candidate_ls])
            continue

        ret = []
        while len(ret) < tm_size:
            max_sim = -1
            for c_id,c_text in candidate_ls:

                edit_sim_src = get_edit_sim(src,c_text)

                edit_sim_tm = 0
                if ret:
                    edit_sim_tm = sum(get_edit_sim(c_text,id2text[x]) for x in ret)/len(ret)

                edit_sim = edit_sim_src - alpha * edit_sim_tm

                if edit_sim > max_sim:
                    max_sim = edit_sim
                    max_sim_id = c_id
                    max_sim_text = c_text

            ret.append(max_sim_id)
            try:
                candidate_ls.remove((max_sim_id,max_sim_text))
            except ValueError:
                print('candidate_ls:',[x[0] for x in candidata_ls])
                print('max_sim_score:',max_sim)
                print(ret)
                print(max_sim_id)
                exit()
        sorted_retrieval_ls.append(ret)
    pickle.dump(sorted_retrieval_ls,open('../data/retrieval/'+translation_direction+'/src_editdis_alpha_'+str(alpha)+'.pkl','wb'))
    from utils import split_train_dev_test
    split_train_dev_test('../data/retrieval/'+translation_direction+'/src_editdis_alpha_'+str(alpha)+'.pkl',direction=translation_direction)


