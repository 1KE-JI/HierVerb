
# Hierarchical Verbalizer for Few-Shot Hierarchical Text Classification

This is the official repository for the ACL 2023 paper
[Hierarchical Verbalizer for Few-Shot Hierarchical Text Classification](https://arxiv.org/pdf/2305.16885.pdf)
![DataConstruction](./image/overview.png)
## Requirements

* Python >= 3.6
* torch == 1.10.1
* openprompt == 0.1.2
* transformers == 4.18.0
* datasets == 2.4.0

## Preprocess

Please download the original dataset and then use these scripts.

### WebOfScience

The original dataset can be acquired in [the repository of HDLTex](https://github.com/kk7nc/HDLTex). Preprocess code could refer to [the repository of HiAGM](https://github.com/Alibaba-NLP/HiAGM) and we provide a copy of preprocess code here. For convenience, here is the WOS dataset [Google Drive](https://drive.google.com/file/d/1UuVDd3uEVVFcuy6i-LdZUo6SHJSMQ1-b/view?usp=share_link) after preprocessing.
```shell
cd ./data/WebOfScience
python preprocess_wos.py
```

### DBPedia

The original dataset wiki_data.csv can be acquired [Google Drive](https://drive.google.com/file/d/1UuVDd3uEVVFcuy6i-LdZUo6SHJSMQ1-b/view?usp=share_link).
```shell
cd ./data/DBPedia
```

### RCV1-V2

The preprocess code could refer to the [repository of reuters_loader](https://github.com/ductri/reuters_loader) and we provide a copy here. The original dataset can be acquired [here](https://trec.nist.gov/data/reuters/reuters.html) by signing an agreement.

```shell
cd ./data/rcv1
python preprocess_rcv1.py
python data_rcv1.py
```

## Train

```
usage: train.py [-h] [--lr LR] [--dataset DATA] [--batch BATCH] [--device DEVICE] --name NAME [--shot SHOT]
                [--seed SEED]....

optional arguments:
  --lr                      LR, learning rate for language model.                   
  --lr2                     LR, learning rate for verbalizer.
  --dataset                 {wos,dbp,rcv1} Dataset.
  --batch BATCH             Batch size
  --shot SHOT               fewshot seeting
  --device DEVICE           cuda or cpu. Default: cuda
  --seed SEED               Random seed.
  --constraint_loss         Hierarchy-aware constraint chain
  --contrastive_loss        flat Hierarchical contrastive loss
  --contrastive_level       \alpha
  --constraint_alpha        \lambda_1 the weight of HCC(default -1 )
  --contrastive_alpha       \lambda_2 the weight of FHC(default 0.99)
```

- Results are in `./result/few_shot_train.txt`.
- Checkpoints are in `./ckpts/`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively.
- For example (`wos-seed550-lr5e-05-coarse_alpha-1-shot-1-ratio-1.0-length30070-macro.ckpt`, 
`wos-seed171-lr5e-05-coarse_alpha-1-shot-1-ratio-1.0-length30070-micro.ckpt`).

## Run the scripts
```shell
## Train and test on WOS dataset
python train.py --device=0 --batch=5 --dataset=wos --shot=1 --seed=550 --constraint_loss=1 --contrastive_loss=1 --contrastive_alpha=0.99 --contrastive_level=1 --use_dropout_sim=1 --contrastive_logits=1
```

## Reproducibility

We experiment on one Tesla V100-SXM2-32GB with CUDA version $10.2$. We use a batch size of $5$ to fully tap one GPU.

## Citation
If you found this repository is helpful, please cite our paper:
```
@inproceedings{ji-etal-2023-hierarchical,
    title = "Hierarchical Verbalizer for Few-Shot Hierarchical Text Classification",
    author = "Ji, Ke  and
      Lian, Yixin  and
      Gao, Jingsheng  and
      Wang, Baoyuan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.164",
    pages = "2918--2933",
    abstract = "Due to the complex label hierarchy and intensive labeling cost in practice, the hierarchical text classification (HTC) suffers a poor performance especially when low-resource or few-shot settings are considered. Recently, there is a growing trend of applying prompts on pre-trained language models (PLMs), which has exhibited effectiveness in the few-shot flat text classification tasks. However, limited work has studied the paradigm of prompt-based learning in the HTC problem when the training data is extremely scarce. In this work, we define a path-based few-shot setting and establish a strict path-based evaluation metric to further explore few-shot HTC tasks. To address the issue, we propose the hierarchical verbalizer ({``}HierVerb{''}), a multi-verbalizer framework treating HTC as a single- or multi-label classification problem at multiple layers and learning vectors as verbalizers constrained by hierarchical structure and hierarchical contrastive learning. In this manner, HierVerb fuses label hierarchy knowledge into verbalizers and remarkably outperforms those who inject hierarchy through graph encoders, maximizing the benefits of PLMs. Extensive experiments on three popular HTC datasets under the few-shot settings demonstrate that prompt with HierVerb significantly boosts the HTC performance, meanwhile indicating an elegant way to bridge the gap between the large pre-trained model and downstream hierarchical classification tasks.",
}
```