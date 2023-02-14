import sacrebleu
import argparse
import os
import logging
import json
import tqdm
from elasticsearch import Elasticsearch
import re
import sacrebleu
import editdistance

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--build_index', action='store_true',
        help='whether to train an index from scratch',default=True,)
    parser.add_argument('--search_index', action='store_true',
        help='whether to search from a built index',default=True,)
    parser.add_argument('--index_file', type=str, default='../data/jrc/deen/train.json')
    parser.add_argument('--search_file', type=str, default='../data/jrc/deen/all.json')
    parser.add_argument('--output_file', type=str, default='../data/retrieval/deen/bm25_src.pkl')
    parser.add_argument('--index_name', type=str,default='deen_src')
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--allow_hit', action='store_true',default=True)
    return parser.parse_args()

def debpe(bpe):
    return re.sub(r'(@@ )|(@@ ?$)', '', bpe)

def get_unedited_words(src_sent, src_tm_sent):
    # Here first sentence is the edited sentence and the second sentence is the target sentence
    # If src_sent is the first sentence, then insertion and deletion should be reversed.
    """ Dynamic Programming Version of edit distance and finding unedited words."""
    a = src_sent.split()
    b = src_tm_sent.split()
    edit_distance = editdistance.eval(a, b)

    edit_distance = 1 - edit_distance / max(len(src_sent), len(src_tm_sent))

    return edit_distance

def get_bleu(src,tm):
    src = [src]
    tm = [tm]
    return sacrebleu.corpus_bleu(tm,[src],force=True,lowercase=False,tokenize='none').score

def get_topk_sent_id(src, src_sim, k=6):
    scores = list(map(lambda x: -get_unedited_words(src, x), src_sim))
    topk = sorted(zip(scores, range(len(scores))), key=lambda x: x[0])[:k]
    ans = [it[1] for it in topk]
    return ans

def main(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.WARNING)

    es = Elasticsearch([{u'host': "localhost", u'port': "9200"}])
    if args.build_index:
        ids,queries, responses = [], [],[]
        samples = [json.loads(x) for x in open(args.index_file).readlines()]
        for sample in samples:
            queries.append(sample['de'])
            responses.append(sample['en'])
            ids.append(sample['id'])


        logger.info('build with elasticsearch')
        from elasticsearch.helpers import bulk
        body = {
            "settings": {
                "index": {
                    "analysis": {
                        "analyzer": "standard"
                    },
                    "number_of_shards": "1",
                    "number_of_replicas": "1",
                }
            },
            "mappings": {
                "properties": {
                    "query": {
                        "type": "text",
                        "similarity": "BM25",
                        "analyzer": "standard",
                    },
                    "response": {
                        "type": "text",
                    },
                    "id": {
                        "type": "text",
                    }
                }
            }
        }
        if es.indices.exists(index=args.index_name):
            es.indices.delete(index=args.index_name)
        es.indices.create(index=args.index_name,body=body)

        index = args.index_name

        actions = []

        for idx, (query, response,iid) in tqdm.tqdm(enumerate(zip(queries, responses,ids)), total=len(queries)):
            action = {
                "_index": index,
                "_source": {
                    "query": debpe(query),
                    "response": response,
                    "id":iid,
                }
            }
            actions.append(action)
            if len(actions) >= 1000:
                success, _ = bulk(es, actions, raise_on_error=True)
                actions = []
        if actions:
            success, _ = bulk(es, actions, raise_on_error=True)
        es.indices.refresh(index)
        info = es.indices.stats(index=index)
        print('total document indexed', info["indices"][index]["primaries"]["docs"]["count"]) 

    if args.search_index:
        queries, responses = [],[]
        with open(args.search_file, 'r') as f:

            data = [json.loads(x) for x in f.readlines()]
            for d in data:
                queries.append(d['de'])

        logger.info('search with elasticsearch')

        index = args.index_name
        query_body = {
                "query": {
                "match":{
                     "query": None
                 }
                },
                "size": args.topk
        }
        with open(args.output_file,'w') as fo:
            
            for query in tqdm.tqdm(queries,total=len(queries)):
                query_body["query"]["match"]["query"] = debpe(query)
                es_result = es.search(index=index, body=query_body)
                ret_q = [ item["_source"]["query"] for item in es_result["hits"]["hits"]]
                ret_i = [ item["_source"]["id"] for item in es_result["hits"]["hits"]]

                fo.write(json.dumps(ret_i))
                fo.write('\n')
if __name__ == "__main__":
    args = parse_args()
    main(args)
