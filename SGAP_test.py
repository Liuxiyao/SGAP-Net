#-------------------------------------
# Project:
# Date:
# Author:
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_test as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
import scipy.io as sio

parser = argparse.ArgumentParser(description="HOI Recognition")
parser.add_argument("-dataset","--dataset",default = 'HICO')
parser.add_argument("-way","--class_num",type = int, default = 5)
parser.add_argument("-shot","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-query_num","--query_num_per_class",type = int, default =1)
parser.add_argument("-episode","--episode",type = int, default= 10)
parser.add_argument("-test_episode","--test_episode", type = int, default = 600)
parser.add_argument("-learning_rate","--learning_rate", type = float, default = 0.000001)
parser.add_argument("-gpu","--gpu",type=int, default=0)
parser.add_argument("-t","--tau",type=int, default=1.5)
parser.add_argument("-a","--alpha",type=int, default=0.5)
args = parser.parse_args()

Feature_D = 512
# Hyper Parameters
Dataset = args.dataset
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
Tau = args.tau
Alpha = args.alpha


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h
class TNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(TNetwork, self).__init__()
        self.word_feature1 = nn.Linear(400,Feature_D)
        self.word_feature2 = nn.Linear(Feature_D, Feature_D)
        self.features=nn.Linear(Feature_D,Feature_D)
        self.map=nn.Linear(Feature_D,Feature_D)
        self.norm = nn.BatchNorm1d(Feature_D, momentum=1, affine=True)

    def forward(self,x,y):
        out1 = F.relu(self.word_feature1(x))
        out1 = F.sigmoid(self.word_feature2(out1))
        out2 = F.relu(self.features(y))
        out = F.relu((out1 + 1) * out2)
        out = F.relu(self.map(out))
        return out

class Generate_word(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(Generate_word, self).__init__()
        self.word_feature1 = nn.Linear(Feature_D,400)
        #self.word_feature2 = nn.Linear(1000,400)

    def forward(self,x):
        out1 = F.relu(self.word_feature1(x))
        #out2 = F.relu(self.word_feature2(y))
        return out1

def euclidean_dist2(x,y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d !=y.size(1):
        raise Exception
    x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(n,m,d)
    return torch.pow(x-y,2).sum(2)

def euclidean_dist(x,y):
    x1 = x.size(0)
    x2 = x.size(1)
    x3 = x.size(2)
    y1 = y.size(0)
    y2 = y.size(1)
    if x3 !=y2:
        raise Exception
    #x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(x1,x2,x3)
    return torch.pow(x-y,2).sum(2)

class Task_norm(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(Task_norm, self).__init__()
        self.norm = nn.BatchNorm1d(Feature_D, momentum=1, affine=True)
        #self.word_feature2 = nn.Linear(1000,400)

    def forward(self,x):
        if len(x.size())==3:
            x = x.view(-1,Feature_D)
            out = self.norm(x)
            out = out.view(5,-1,Feature_D)
        else:
            out = self.norm(x)

        #out2 = F.relu(self.word_feature2(y))
        return out

def step_seq(a,x):
    x[x<a]=0
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.dataset_folders(Dataset)

    # Step 2: init neural networks
    print("init neural networks")
    tnetwork = TNetwork()
    tvnetwork = TNetwork()
    generate_word_noun = Generate_word()
    generate_word_verb = Generate_word()
    task_norm = Task_norm()

    tnetwork.apply(weights_init)
    tvnetwork.apply(weights_init)
    generate_word_noun.apply(weights_init)
    generate_word_verb.apply(weights_init)
    task_norm.apply(weights_init)

    tnetwork.cuda(GPU)
    tvnetwork.cuda(GPU)
    generate_word_noun.cuda(GPU)
    generate_word_verb.cuda(GPU)
    task_norm.cuda(GPU)


    print("Testing on test * 10...")
    if os.path.exists('./models/' + Dataset + '_tvnetwork_5way' + str(SAMPLE_NUM_PER_CLASS) + 'shot_max.pkl'):
        tvnetwork.load_state_dict(torch.load('./models/' + Dataset + '_tvnetwork_5way' + str(SAMPLE_NUM_PER_CLASS) + 'shot_max.pkl'))
        tnetwork.load_state_dict(torch.load('./models/' + Dataset + '_tnetwork_5way' + str(SAMPLE_NUM_PER_CLASS) + 'shot_max.pkl'))
        print("load noun-verb part success")
    if os.path.exists('./models/' + Dataset + '_generate_word_verb_5way' + str(SAMPLE_NUM_PER_CLASS) + 'shot_max.pkl'):
        generate_word_noun.load_state_dict(torch.load('./models/' + Dataset + '_generate_word_noun_5way' + str(SAMPLE_NUM_PER_CLASS) + 'shot_max.pkl'))
        generate_word_verb.load_state_dict(torch.load('./models/' + Dataset + '_generate_word_verb_5way' + str(SAMPLE_NUM_PER_CLASS) + 'shot_max.pkl'))
        task_norm.load_state_dict(torch.load('./models/' + Dataset + '_task_norm_5way' + str(SAMPLE_NUM_PER_CLASS) + 'shot_max.pkl'))
        print("load generate_wordNet success")

    total_accuracy = 0.0
    H_all = 0.0
    for episode in range(EPISODE):
            # test
            print("Testing...")

            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                task = tg.MiniImagenetTask(metatest_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, 1)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                                     split="train", shuffle=False)
                num_per_class = 1
                test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=num_per_class, split="test",
                                                                   shuffle=False)
                sample_images, sample_labels = sample_dataloader.__iter__().next()
                for test_images, test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = sample_images.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, Feature_D).cuda(GPU)
                    test_features = test_images.cuda(GPU)  # 20x64

                    #sample_features = task_norm(sample_features)
                    #test_features = task_norm(test_features)

                    n_vector1 = generate_word_noun(sample_features)
                    sample_noun = tnetwork(n_vector1, sample_features)
                    sample_features1 = sample_features - step_seq(Tau,sample_noun)
                    v_vector1 = generate_word_verb(sample_features1)
                    sample_verb = tvnetwork(v_vector1, sample_features1)
                    sample_features2 = sample_features1 + sample_noun + sample_verb

                    fake_n_vector = generate_word_noun(test_features)
                    test_noun = tnetwork(fake_n_vector, test_features)
                    test_features1 = test_features - step_seq(Tau,test_noun)
                    fake_v_vector = generate_word_verb(test_features1)
                    test_verb = tvnetwork(fake_v_vector, test_features1)
                    test_features2 = test_features1 + test_noun + test_verb

                    prototypes = torch.mean(sample_features2, 1)
                    dists = euclidean_dist2(test_features2.view(-1, Feature_D), prototypes)
                    log_p_y = F.log_softmax(-dists, dim=1)

                    _, predict_labels = torch.max(log_p_y.data, 1)

                    rewards = [1 if predict_labels[j] == test_labels.cuda()[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)


                accuracy = total_rewards/1.0/CLASS_NUM/1
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)

            total_accuracy += test_accuracy
            H_all += h

    print("aver_accuracy:", total_accuracy / EPISODE, "aver_h:", H_all / EPISODE)






if __name__ == '__main__':
    main()