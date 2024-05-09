# LIMFA

This is the official implementation of our paper **LIMFA:Label-irrelevant multi-domain feature alignment based fake news detection for unseen domain**, which has been published in Neural Computing and Applications. [Paper](https://link.springer.com/article/10.1007/s00521-023-09340-z)

## Abstract

Fake news in social networks causes disastrous effects on the real world yet effectively detecting newly emerged fake news remains difficult. This problem is particularly pronounced when the testing samples (target domain) are derived from different topics, events, platforms or time periods from the training dataset (source domains). Though efforts have focused on learning domain-invariant features (DIF) across multiple source domains to transfer universal knowledge from the source to the target domain, they ignore the complexity that arises when the number of source domains increases, resulting in unreliable DIF. In this paper, we first point out two challenges faced by learning DIF for fake news detection, (1) high intra-domain correlations, caused by the similarity of news samples within the same domain but different categories can be higher than that in different domains but the same categories, and (2) complex inter-domain correlations, stemming from that news samples in different domains are semantically related. To tackle these challenges, we propose two modules, center-aware feature alignment and likelihood gain-based feature disentanglement, to enhance the multiple domains alignment while enforcing two categories separated and disentangle the domain-specific features in an adversarial supervision manner. By combining these modules, we conduct a label-irrelevant multi-domain feature alignment (LIMFA) framework. Our experiments show that LIMFA can be deployed with various base models and it outperforms the state-of-the-art baselines in 4 cross-domain scenarios. Our source codes will be available upon the acceptance of this manuscript.

## Dataset
The datasets we used in our paper are Pheme and Weibo. We provide the link to the original dataset and data processing code. The Pheme dataset can be download from https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619. the weibo dataset is avalable at https://github.com/ICTMCG/Characterizing-Weibo-Multi-Domain-False-News.

## Requirements

- Python 3.6
- PyTorch > 1.0

  
## Run
You can run this code through:

```powershell
python main.py 
```

## Reference

```
Wu, D., Tan, Z., Zhao, H. et al. LIMFA: label-irrelevant multi-domain feature alignment-based fake news detection for unseen domain. Neural Comput & Applic 36, 5197–5215 (2024). https://doi.org/10.1007/s00521-023-09340-z
```
