
# SGAP-Net: Semantic-Guided Attentive Prototypes Network for Few-Shot Human-Object Interaction Recognition, AAAI2020.

## Few-Shot Human-Object Interaction Recognition with Semantic-Guided Attentive Prototypes Network

We resubmit it in TIP with 4 extensions as follows:
+ an alternative prototypes calculation approach called Hallucinatory Graph Prototypes (HGP), which consists of a hallucinator and an HOI Graph Convolution Network (GCN); 
+ a new dataset split strategy, and the corresponding experiments; 
+ cross-domain experiments between different datasets; 
+ additional introduction to related work and ablation studies.

## Dependencies

This code requires the following:

    python 3.6+*
    Pytorch 1.0+

## Dataset
[FS-HOI] <https://pan.baidu.com/s/19KpUojAfL75EgIOxZVJxAA>
code：283w 

## Abstract

Extreme instance imbalance among categories and combinatorial explosion make the recognition of Human-Object Interaction (HOI) a challenging task. Few studies have addressed both challenges directly. Motivated by the success of few-shot learning that learns a robust model from a few instances, we formulate HOI as a few-shot task in a meta-learning framework to alleviate the above challenges. Due to the fact that the intrinsic characteristic of HOI is diverse and interactive, we propose a Semantic-Guided Attentive Prototypes Network (SGAP-Net) to learn a semantic-guided metric space where HOI recognition can be performed by computing distances to attentive prototypes of each class. Specifically, the model generates attentive prototypes guided by the category names of actions and objects, which highlight the commonalities of images from the same class in HOI. In addition, we design a novel decision method to alleviate the biases produced by different patterns of the same action in HOI. Finally, in order to realize the task of few-shot HOI, we reorganize two HOI benchmark datasets, i.e., HICO-FS and TUHOI-FS, to realize the task of few-shot HOI. Extensive experimental results on both datasets have demonstrated the effectiveness of our proposed SGAP-Net approach.
