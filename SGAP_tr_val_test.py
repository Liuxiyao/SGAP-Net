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
import task_generator as tg
import task_generator_test as tg_test
import os
import math
import argparse
import scipy as sp
import scipy.stats
import torchvision
import scipy.io as sio

parser = argparse.ArgumentParser(description="HOI Recognition")
parser.add_argument("-dataset","--dataset",default = 'HICO')
parser.add_argument("-way","--class_num",type = int, default = 5)
parser.add_argument("-shot","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-query_num","--query_num_per_class",type = int, default =5)
parser.add_argument("-episode","--episode",type = int, default= 500000)
parser.add_argument("-test_episode","--test_episode", type = int, default = 600)
parser.add_argument("-test_round","--test_round",type = int, default= 10)
parser.add_argument("-learning_rate","--learning_rate", type = float, default = 0.000001)
parser.add_argument("-gpu","--gpu",type=int, default=0)
parser.add_argument("-t","--tau",type=int, default=1.5)
parser.add_argument("-a","--alpha",type=int, default=0.5)
args = parser.parse_args()


# Hyper Parameters
Dataset = args.dataset
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
TEST_ROUNd = args.test_round
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
        self.word_feature1 = nn.Linear(400,1000)
        self.word_feature2 = nn.Linear(1000, 1000)
        self.features=nn.Linear(1000,1000)
        self.map=nn.Linear(1000,1000)
        self.norm = nn.BatchNorm1d(1000,momentum=1,affine=True)

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
        self.word_feature1 = nn.Linear(1000,400)
        #self.word_feature2 = nn.Linear(1000,400)

    def forward(self,x):
        out1 = F.relu(self.word_feature1(x))
        #out2 = F.relu(self.word_feature2(y))
        return out1

class Task_norm(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(Task_norm, self).__init__()
        self.norm = nn.BatchNorm1d(1000, momentum=1, affine=True)
        #self.word_feature2 = nn.Linear(1000,400)

    def forward(self,x):
        if len(x.size())==3:
            x = x.view(-1,1000)
            out = self.norm(x)
            out = out.view(5,-1,1000)
        else:
            out = self.norm(x)

        #out2 = F.relu(self.word_feature2(y))
        return out

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

    reader = sio.loadmat('./'+Dataset+'_Vector.mat') #HOCI word_features
    word_vector = reader.get('word_fea')
    class_name_raw = open('./'+Dataset+'_class.txt', 'r')
    class_name_repeat = []
    for name in class_name_raw:  # contain '+' in combination word
        ind2 = name.find('\r')
        class_name_repeat.append(name[0: ind2].replace('-', '_'))
    class_name = sorted(set(class_name_repeat))

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

    tnetwork_optim = torch.optim.Adam(tnetwork.parameters(), lr=LEARNING_RATE,weight_decay=0.01)
    tnetwork_scheduler = StepLR(tnetwork_optim, step_size=100000, gamma=0.1)
    tvnetwork_optim = torch.optim.Adam(tvnetwork.parameters(), lr=LEARNING_RATE,weight_decay=0.01)
    tvnetwork_scheduler = StepLR(tvnetwork_optim, step_size=100000, gamma=0.1)
    generate_word_noun_optim = torch.optim.Adam(generate_word_noun.parameters(), lr=LEARNING_RATE,weight_decay=0.01)
    generate_word_noun_scheduler = StepLR(generate_word_noun_optim, step_size=10000, gamma=0.5)
    generate_word_verb_optim = torch.optim.Adam(generate_word_verb.parameters(), lr=LEARNING_RATE,weight_decay=0.01)
    generate_word_verb_scheduler = StepLR(generate_word_verb_optim, step_size=10000, gamma=0.5)
    task_norm_optim = torch.optim.Adam(task_norm.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    task_norm_scheduler = StepLR(task_norm_optim, step_size=10000, gamma=0.5)

    # if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
    #     feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
    #     print("load feature encoder success")
    # if os.path.exists(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
    #     relation_network.load_state_dict(torch.load(str("./models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
    #     print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        tnetwork_scheduler.step(episode)
        tvnetwork_scheduler.step(episode)
        generate_word_noun_scheduler.step(episode)
        generate_word_verb_scheduler.step(episode)
        task_norm_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=False)

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()
        # Lookup semantic
        noun_idx = [class_name.index(noun) for noun in task.n_folders]
        verb_idx = [class_name.index(verb) for verb in task.v_folders]
        n_vector = [word_vector[noun_idx[i], :] for i in range(CLASS_NUM)]  # list w*400
        v_vector = [word_vector[verb_idx[i], :] for i in range(CLASS_NUM)]  # list w*400

        # calculate features
        sample_features = samples.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 1000).cuda(GPU)  # w*s*1000
        batch_features = batches.cuda(GPU)  # 5*1000


        n_vector = F.relu(torch.FloatTensor(n_vector))
        v_vector = F.relu(torch.FloatTensor(v_vector))
        truth_n_vector = torch.FloatTensor(n_vector).unsqueeze(0).repeat(BATCH_NUM_PER_CLASS, 1, 1).view(-1, 400)
        truth_v_vector = torch.FloatTensor(v_vector).unsqueeze(0).repeat(BATCH_NUM_PER_CLASS, 1, 1).view(-1, 400)
        n_vector = torch.FloatTensor(n_vector).unsqueeze(1).repeat(1, SAMPLE_NUM_PER_CLASS, 1)
        v_vector = torch.FloatTensor(v_vector).unsqueeze(1).repeat(1, SAMPLE_NUM_PER_CLASS, 1)

        sample_features = task_norm(sample_features)
        #batch_features = task_norm(batch_features)

        sample_noun = tnetwork(n_vector.cuda(), sample_features)
        sample_features1 = sample_features - step_seq(Tau,sample_noun)
        sample_verb = tvnetwork(v_vector.cuda(), sample_features1)
        sample_features2 = sample_noun + sample_verb + sample_features1

        fake_n_vector = generate_word_noun(batch_features)
        batch_noun = tnetwork(fake_n_vector, batch_features)
        batch_features1 = batch_features - step_seq(Tau,batch_noun)
        fake_v_vector = generate_word_verb(batch_features1)
        batch_verb = tvnetwork(fake_v_vector, batch_features1)
        batch_features2 = batch_noun + batch_verb + batch_features1

        vector_distance = nn.PairwiseDistance().cuda(GPU)
        loss_vector = torch.mean(vector_distance(fake_n_vector, truth_n_vector.cuda()))+torch.mean(vector_distance(fake_v_vector, truth_v_vector.cuda()))
        loss_vector = 0.1*loss_vector


        prototypes_origins = torch.mean(sample_features2, 1)


        batch_features2 = batch_features2.unsqueeze(1)  # 10*1*1000
        batch_features2 = batch_features2.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1)  # 5*10*1*1000
        batch_features_a = torch.transpose(batch_features2, 0, 1).squeeze()  # 10*5*1*1000
        batch_features2 = torch.transpose(batch_features2, 0, 1)  # 10*5*1*1000
        prototypes_origins1 = prototypes_origins.unsqueeze(0).repeat(CLASS_NUM * BATCH_NUM_PER_CLASS, 1, 1)
        prototypes_all = prototypes_origins1 * (1 - Alpha) + Alpha * batch_features_a


        #triplet loss, prototypes shift
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1, 1), 1).cuda(GPU))
        positive_tensor = torch.mm(one_hot_labels, prototypes_origins)
        navigative_labels = np.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM)
        labels = torch.unique(batch_labels)
        for i in range(BATCH_NUM_PER_CLASS * CLASS_NUM):
            pick = np.random.choice(np.setdiff1d(labels, batch_labels.numpy()[i]), replace=True)
            navigative_labels[i] = pick
        navigative_labels = torch.LongTensor(navigative_labels)
        one_hot_labels_false = Variable(torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, navigative_labels.view(-1, 1), 1).cuda(GPU))
        navigative_tensor = torch.mm(one_hot_labels_false, prototypes_origins)
        loss_triplet = torch.nn.functional.triplet_margin_loss(batch_features2, positive_tensor, navigative_tensor)

        #protypes loss
        dists = euclidean_dist(prototypes_all,prototypes_origins)  # 50*5
        log_p_y = F.log_softmax(-dists, dim=1)
        batch_labels_ex = batch_labels.view(CLASS_NUM * BATCH_NUM_PER_CLASS, 1)
        loss_proto = -log_p_y.gather(1, batch_labels_ex.cuda())
        loss_proto = loss_proto.squeeze().view(-1).mean()



        loss = loss_proto + loss_vector + loss_triplet




        # training

        tnetwork.zero_grad()
        tvnetwork.zero_grad()
        generate_word_noun.zero_grad()
        generate_word_verb.zero_grad()
        task_norm.zero_grad()



        loss.backward()


        # torch.nn.utils.clip_grad_norm(mapping_network.parameters(),0.5)
        # torch.nn.utils.clip_grad_norm(tnetwork.parameters(), 0.5)
        # torch.nn.utils.clip_grad_norm(tvnetwork.parameters(), 0.5)
        #torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)

        tnetwork_optim.step()
        tvnetwork_optim.step()
        generate_word_noun_optim.step()
        generate_word_verb_optim.step()
        task_norm_optim.step()

        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.item(),"loss_proto",loss_proto.item(),"loss_vector",loss_vector.item(),"loss_triplet",loss_triplet.item())

        if (episode+1)%1000 == 0:

            # test
            print("Testing on val...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                task = tg_test.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,1)
                sample_dataloader = tg_test.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                num_per_class = 1
                test_dataloader = tg_test.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)
                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = sample_images.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 1000).cuda(GPU)
                    test_features = test_images.cuda(GPU)  # 20x64

                    sample_features = task_norm(sample_features)
                    #test_features = task_norm(test_features)

                    n_vector1 = generate_word_noun(sample_features)
                    sample_noun = tnetwork(n_vector1, sample_features)
                    sample_features1 = sample_features - step_seq(Tau,sample_noun)
                    v_vector1 = generate_word_verb(sample_features1)
                    sample_verb = tvnetwork(v_vector1, sample_features1)
                    sample_features2 = sample_noun + sample_verb + sample_features1

                    fake_n_vector = generate_word_noun(test_features)
                    test_noun = tnetwork(fake_n_vector, test_features)
                    test_features1 = test_features - step_seq(Tau,test_noun)
                    fake_v_vector = generate_word_verb(test_features1)
                    test_verb = tvnetwork(fake_v_vector, test_features1)
                    test_features2 = test_noun + test_verb + test_features1

                    prototypes = torch.mean(sample_features2, 1)
                    dists = euclidean_dist2(test_features2.view(-1, 1000), prototypes)
                    log_p_y = F.log_softmax(-dists, dim=1)

                    _,predict_labels = torch.min(log_p_y.data,1)

                    rewards = [1 if predict_labels[j]==test_labels.cuda()[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)


                accuracy = total_rewards/1.0/CLASS_NUM/1
                accuracies.append(accuracy)


            test_accuracy,h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(generate_word_noun.state_dict(),str("./models/"+Dataset+"_generate_word_noun_"+str(CLASS_NUM)+"way"+str(SAMPLE_NUM_PER_CLASS)+"shot" + "_max.pkl"))
                torch.save(generate_word_verb.state_dict(),str("./models/"+Dataset+"_generate_word_verb_"+str(CLASS_NUM)+"way"+str(SAMPLE_NUM_PER_CLASS)+"shot" + "_max.pkl"))
                torch.save(tnetwork.state_dict(),str("./models/"+Dataset+"_tnetwork_"+str(CLASS_NUM)+"way"+str(SAMPLE_NUM_PER_CLASS)+"shot" + "_max.pkl"))
                torch.save(tvnetwork.state_dict(),str("./models/"+Dataset+"_tvnetwork_"+str(CLASS_NUM)+"way"+str(SAMPLE_NUM_PER_CLASS)+"shot" + "_max.pkl"))
                torch.save(task_norm.state_dict(), str("./models/"+Dataset+"_task_norm_"+str(CLASS_NUM)+"way"+str(SAMPLE_NUM_PER_CLASS)+"shot" + "_max.pkl"))

                print("save networks for episode:",episode)
                best_episode = episode
                last_accuracy = test_accuracy
            print('best accuracy:',last_accuracy,"best networks from episode:",best_episode)



    print("********************************")
    print("Testing on test * 10...")
    if os.path.exists('./models/'+Dataset+'_tvnetwork_5way'+str(SAMPLE_NUM_PER_CLASS)+'shot_max.pkl'):
        tvnetwork.load_state_dict(torch.load('./models/'+Dataset+'_tvnetwork_5way'+str(SAMPLE_NUM_PER_CLASS)+'shot_max.pkl'))
        tnetwork.load_state_dict(torch.load('./models/'+Dataset+'_tnetwork_5way'+str(SAMPLE_NUM_PER_CLASS)+'shot_max.pkl'))
        print("load noun-verb part success")
    if os.path.exists('./models/'+Dataset+'_generate_word_verb_5way'+str(SAMPLE_NUM_PER_CLASS)+'shot_max.pkl'):
        generate_word_noun.load_state_dict(torch.load('./models/'+Dataset+'_generate_word_noun_5way'+str(SAMPLE_NUM_PER_CLASS)+'shot_max.pkl'))
        generate_word_verb.load_state_dict(torch.load('./models/'+Dataset+'_generate_word_verb_5way'+str(SAMPLE_NUM_PER_CLASS)+'shot_max.pkl'))
        task_norm.load_state_dict(torch.load('./models/'+Dataset+'_task_norm_5way'+str(SAMPLE_NUM_PER_CLASS)+'shot_max.pkl'))
        print("load generate_wordNet success")

    total_accuracy = 0.0
    H_all = 0.0
    metatrain_folders, metatest_folders = tg_test.dataset_folders(Dataset)
    for episode in range(TEST_ROUNd):
            # test


            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                task = tg_test.MiniImagenetTask(metatest_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, 1)
                sample_dataloader = tg_test.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                                     split="train", shuffle=False)
                num_per_class = 1
                test_dataloader = tg_test.get_mini_imagenet_data_loader(task, num_per_class=num_per_class, split="test",
                                                                   shuffle=False)
                sample_images, sample_labels = sample_dataloader.__iter__().next()
                for test_images, test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = sample_images.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 1000).cuda(GPU)
                    test_features = test_images.cuda(GPU)  # 20x64

                    sample_features = task_norm(sample_features)
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
                    dists = euclidean_dist2(test_features2.view(-1, 1000), prototypes)
                    log_p_y = F.log_softmax(-dists, dim=1)

                    _, predict_labels = torch.min(log_p_y.data, 1)

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
