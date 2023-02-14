## Neural Machine Translation with Contrastive Translation Memories
This repository contains the code and data for this EMNLP2022 paper [Neural Machine Translation with Contrastive Translation Memories](https://aclanthology.org/2022.emnlp-main.235/). 

The main idea of this paper is to exploit contrastive translation memories in retrieval-augmented NMT system.

<div align=center>
<img src="model.png" width="650" height="600">
</div>
## Environment

The required packages are listed in `requirement.txt` and we highly recommend to use `conda` to create an isolated environment as follows:

```bash
conda create -n nmt python=3.7
conda activate nmt
pip install -r requirement.txt
```

## Data

We conduct all our experiments on **JRC-Acquis** dataset. It is proposed in this paper: [The JRC-Acquis: A Multilingual Aligned Parallel Corpus with 20+ Languages](http://www.lrec-conf.org/proceedings/lrec2006/pdf/340_pdf.pdf). We use the same data version as in [Search Engine Guided Neural Machine Translation](https://ojs.aaai.org/index.php/AAAI/article/view/12013) and [Neural Machine Translation with Monolingual Translation Memory](https://aclanthology.org/2021.acl-long.567.pdf). 

For original data, we refer to this [LINK](https://drive.google.com/file/d/1iuBH_YsnL28cTYjjpSq5BgukG7QhBLs_/view) to download the data and this [script](https://github.com/jcyk/copyisallyouneed/blob/master/scripts/prepare.sh) for data pre-processing.

For ready-to-go data, we provide it [here](https://drive.google.com/file/d/1ghmdaTFUGVj_rIM0YJpSr6P_Zj3cNcyZ/view?usp=share_link).

## Memory Retrieval

We use `ElasticSearch` to conduct first-stage memory retrieval based on BM25 score as in `bm25.py`. A useful guide about launching `ElasticSearch` cound be found [here](https://cuiqingcai.com/6214.html).

As for contrastive retrieval, we refer to `editdis_retrieval.py`

For ready-to-go memory, we provide it [here](https://drive.google.com/file/d/15LibHuRtvOGsrCVs8BsVnHM1wUnbcE_e/view?usp=share_link).

## Training

The whoel model is based on the awesome [HuggingFace/Transformers](https://github.com/huggingface/transformers). The definition of a **CMM** model is in `module.py` and `model.py`

After everything is ready, just run the following command would get a **CMM** model in En->De direction. All configs could be modified in `args.py`.

```python
python main.py
```

## Citation

If you find this project helpful, please consider cite our paper. 

Please feel free to open an issue or email me ([chengxin1998@stu.pku.edu.cn](mailto:chengxin1998@stu.pku.edu.cn)) for questions and suggestions.

```
@inproceedings{cheng-etal-2022-neural,
    title = "Neural Machine Translation with Contrastive Translation Memories",
    author = "Cheng, Xin  and
      Gao, Shen  and
      Liu, Lemao  and
      Zhao, Dongyan  and
      Yan, Rui",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.235",
    pages = "3591--3601",
    abstract = "Retrieval-augmented Neural Machine Translation models have been successful in many translation scenarios. Different from previous works that make use of mutually similar but redundant translation memories (TMs), we propose a new retrieval-augmented NMT to model contrastively retrieved translation memories that are holistically similar to the source sentence while individually contrastive to each other providing maximal information gain in three phases. First, in TM retrieval phase, we adopt contrastive retrieval algorithm to avoid redundancy and uninformativeness of similar translation pieces. Second, in memory encoding stage, given a set of TMs we propose a novel Hierarchical Group Attention module to gather both local context of each TM and global context of the whole TM set. Finally, in training phase, a Multi-TM contrastive learning objective is introduced to learn salient feature of each TM with respect to target sentence. Experimental results show that our framework obtains substantial improvements over strong baselines in the benchmark dataset.",
}
```



