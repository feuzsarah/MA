import random
import time
import sys
import pathlib
from collections import OrderedDict, defaultdict
import numpy
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from vast.tools import set_device_gpu, set_device_cpu, device
import vast
from loguru import logger
from .metrics import confidence, auc_score_binary, auc_score_multiclass
from .dataset import ImagenetDataset
from .model import ResNet50, load_checkpoint, save_checkpoint, set_seeds,ResNet50Layers
from .losses import AverageMeter, EarlyStopping, EntropicOpensetLoss
import tqdm
import csv

import os
import pandas as pd


def mix_algo_real(beta_distribution, feature_map_unknown, labels_unknown,feature_map_neg, labels_neg):
    """
    Input: 
    beta_distribution: defined beta distribution
    feature_map_unknown: input from the known samples to mix with negative samples
    labels_unknown: labels from the known samples to mix with negative samples

    feature_map_neg:input from the negative samples to mix with known samples
    labels_neg:labels from the negative samples to mix with known samples

    Output:
    mixed_features: the mixed features
    mixed_labels: the mixed labels (all equal to -1)

    """
    number_of_feature_maps = len(feature_map_unknown)
    beta = beta_distribution.sample([]).item()

    feature_h1 = feature_map_unknown
    feature_h2 = feature_map_neg

    labels_h1 = labels_unknown
    labels_h2 = labels_neg
    labels = torch.cat((labels_h1,labels_h2), dim = 0)

    #mix the neg and known samples together to form new mixed samples
    for i in range(number_of_feature_maps): 
            if i == 0: 
                mix1 = beta * feature_h1[i] + (1-beta) * feature_h2[i]
                mix1 = mix1[None,:]
                mix2 = (1-beta) * feature_h1[i] + (beta) * feature_h2[i]
                mix2 = mix2[None,:]
                mixed_features = torch.cat((mix1, mix2), dim=0)
            else: 
                mix1 = beta * feature_h1[i] + (1-beta) * feature_h2[i]
                mix1 = mix1[None,:]
                mix2 = (1-beta) * feature_h1[i] + (beta) * feature_h2[i]
                mix2 = mix2[None,:]
                mixed_features = torch.cat((mixed_features, mix1), dim = 0)
                mixed_features = torch.cat((mixed_features, mix2), dim = 0)
    
    #set the labels to -1 
    for i in range(number_of_feature_maps*2):
            labels[i] = -1 

    mixed_labels = labels
    return mixed_features, mixed_labels
           

def mix_algo(beta_distribution, feature_map, labels,k):
    """
    Input: 
    beta_distribution: defined beta distribution
    feature_map: input from the known samples to mix
    labels: labels from the known samples to mix with negative samples
    k: counter

    Output:
    mixed_features: the mixed features
    mixed_labels: the mixed labels (all equal to -1)
    k: counter
    """

    beta = beta_distribution.sample([]).item()
    half = int(len(feature_map)/2)
    feature_h1 = feature_map[:half]
    feature_h2 = feature_map[half:]

    labels_h1 = labels[:half]
    labels_h2 = labels[half:]

    index_1 = torch.arange(0,half)
    index_2 = torch.arange(0,half)

    wrong_idx = [x for x in range(half) if labels_h1[index_1[x]] == labels_h2[index_2[x]]]
    loop = 0

    #when labels match for the mixing-pairs shuffle them by rearranging the index array for the matching pairs
    #shuffle down
    while(len(wrong_idx) > 0 and loop < (half)): 
        for i in wrong_idx:
            if(i >0): 
                l = index_1[i]
                index_1[i] = index_1[i-1]
                index_1[i-1] = l
            if(i == 0):
                l = index_1[i]
                index_1[i] = index_1[-1]
                index_1[-1] = l
        wrong_idx = [x for x in range(half) if labels_h1[index_1[x]] == labels_h2[index_2[x]]]

        loop = loop + 1

    wrong_idx = [x for x in range(half) if labels_h1[index_1[x]] == labels_h2[index_2[x]]]

    loop = 0

    #shuffle up
    while(len(wrong_idx) > 0 and loop < (half)): 
        for i in wrong_idx:
            if(i < half-1): 
                l = index_1[i]
                index_1[i] = index_1[i+1]
                index_1[i+1] = l
            if(i == half-1):
                l = index_1[i]
                index_1[i] = index_1[0]
                index_1[0] = l
        wrong_idx = [x for x in range(half) if labels_h1[index_1[x]] == labels_h2[index_2[x]]]
        
        loop = loop +1

    #if we do not have any matching labels in the mixing-pairs, mix them up and set labels to -1
    #else return the original feature map with the corresponding labels (no mixup)
    if(len(wrong_idx) == 0): 
        
        for i in range(half): 
            if i == 0: 
                mix1 = beta * feature_h1[index_1[i]] + (1-beta) * feature_h2[index_2[i]]
                mix1 = mix1[None,:]
                mix2 = (1-beta) * feature_h1[index_1[i]] + (beta) * feature_h2[index_2[i]]
                mix2 = mix2[None,:]
                mixed_features = torch.cat((mix1, mix2), dim=0)
            else: 
                mix1 = beta * feature_h1[index_1[i]] + (1-beta) * feature_h2[index_2[i]]
                mix1 = mix1[None,:]
                mix2 = (1-beta) * feature_h1[index_1[i]] + (beta) * feature_h2[index_2[i]]
                mix2 = mix2[None,:]
                mixed_features = torch.cat((mixed_features, mix1), dim = 0)
                mixed_features = torch.cat((mixed_features, mix2), dim = 0)

        for i in range(len(feature_map)):
             labels[i] = -1 

        k = k + 1
        mixed_labels = labels
    else:
        mixed_features = feature_map
        mixed_labels = labels

    return mixed_features, mixed_labels, k

def train(model, data_loader, optimizer, loss_fn, trackers, cfg, cfg_ns, waiting, current_epoch,nr_neg):
    """ Main training loop.
    Args:
        model (torch.model): Model
        data_loader (torch.DataLoader): DataLoader
        optimizer (torch optimizer): optimizer
        loss_fn: Loss function
        trackers: Dictionary of trackers
        cfg: General configuration structure
        cfg_ns: configuration of negative samples
        waiting: waiting epochs (True/False)
        current_epoch: current epoch
        nr_neg: counter
    """

    # Reset dictionary of training metrics
    for metric in trackers.values():
        metric.reset()

    #define parameters for negative sample methods
    submethod = cfg_ns.submethod
    filter_method = cfg_ns.filter_method
    j = None
    k = 0
    method = cfg_ns.method
    style = cfg_ns.style
    alpha = cfg_ns.alpha
    beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
    mixlayer = cfg_ns.mixlayer

    #include adaptive noise method
    if(cfg_ns.adaptive_noise == True): 
        part_epoch = min((current_epoch-cfg_ns.waiting)/(cfg.epochs-cfg_ns.waiting),0.5)
        pert = cfg_ns.noise*(1.5- part_epoch)
    else:
        pert = cfg_ns.noise

    # training loop
    if not cfg.parallel:
        data_loader = tqdm.tqdm(data_loader)
    for images, labels in data_loader:
        model.train()  # To collect batch-norm statistics
        batch_len = labels.shape[0]  # Samples in current batch
        optimizer.zero_grad()

        #use real negative samples from corresponding protocol as negative samples. 
        if(method == 'real'):
     
            images = device(images)
            labels = device(labels)

            # Forward pass
            logits, features = model(images)

            # Calculate loss
            j = loss_fn(logits, labels)
            trackers["j"].update(j.item(), batch_len)
            # Backward pass
            j.backward()
            optimizer.step()

        #negative sample generation with gaussian noise
        elif(method == 'noise'):

            #submethod that includes negative samples from the protocol 
            if(submethod == 'real'): 
                filter_condition = labels == -1
                count_real_neg = numpy.count_nonzero(filter_condition)
            
                #we look at the known samples and leave out the real negative samples
                remaining_size = batch_len - count_real_neg
                unknown_count = batch_len - count_real_neg

                #if the number of known samples exceeds the max number of negative samples that are generated through noise injection, we set it to the defined maximum number
                if(unknown_count > cfg_ns.number_neg_real):
                        unknown_count = cfg_ns.number_neg_real

                indexes = numpy.where(labels != -1)[0]
                indexes_neg = numpy.where(labels == -1)[0]

                labels_neg = labels[indexes_neg]
                images_neg = images[indexes_neg]

                #counter
                nr_neg = nr_neg + len(labels_neg)

                labels_kn = labels[indexes]
                images_kn = images[indexes]

                #we noise the defined samples, set the labels to -1 and patch it all together
                if(unknown_count > 0):

                    #split the labels and images into two parts; candidate for neg. samples and known samples
                    labels_unknown = labels_kn[:unknown_count]
                    labels_known = labels_kn[unknown_count:]

                    images_unknown = images_kn[:unknown_count]
                    images_known = images_kn[unknown_count:]

                    for el in range(len(images_unknown)):
                        noise = torch.randn(images_unknown[el].shape)                
                        images_unknown[el] = torch.clip(images_unknown[el] +(noise*pert),min = 0, max = 1)
                        labels_unknown[el] = -1 
                    
                    nr_neg = nr_neg + len(labels_unknown)

                    images = torch.cat((images_unknown, images_known, images_neg), dim=0)
                    labels = torch.cat((labels_unknown, labels_known, labels_neg))

                #forward through the network
                images = device(images)
                labels = device(labels)

                # Forward pass
                logits, features = model(images)

                # Calculate loss
                j = loss_fn(logits, labels)
                trackers["j"].update(j.item(), batch_len)
                # Backward pass
                j.backward()

                optimizer.step()
            else: 
                #We use the filter method threshold
                if(filter_method == 'threshold'):
                    model.eval()
                    
                    #we forward the images through the network, calculate the scores and generate a mask based on threshold and correctly classified. 
                    with torch.no_grad():
                        images = device(images)
                        labels = device(labels)

                        # Forward pass
                        logits, features = model(images)
                        scores = torch.nn.functional.softmax(logits, dim=1)
                        value, index = torch.max(scores, dim =1)
                        condition = index == labels
                        condition_2 = value >= cfg_ns.filterthreshold_value
                        mask = torch.logical_and(condition, condition_2)

                        images.cpu()
                        labels.cpu()
                    
                    model.train()
                    #counter
                    neg_count = torch.count_nonzero(mask == True).item()
                    nr_neg = nr_neg + neg_count

                    #apply the above created mask to filter the samples that are used to generate negative samples

                    if(neg_count >0 ):

                        #split the samples into candidates for negative samples and the known samples
                        labels = device(labels)
                        labels_unknown = labels[mask]

                        mask_known = torch.logical_not(mask)
                        labels_known = labels[mask_known]

                        images_unknown = images[mask]
                        images_known = images[mask_known]

                        images_unknown = device(images_unknown)
                        images_known = device(images_known)
                        labels_unknown = device(labels_unknown)
                        labels_known = device(labels_known)

                        #make negative samples with noise injection method
                        for el in range(len(images_unknown)):
                            noise = device(torch.randn(images_unknown[el].shape))                
                            images_unknown[el] = torch.clip(images_unknown[el] +(noise*pert),min = 0, max = 1)
                            labels_unknown[el] = -1 

                        images = torch.cat((images_unknown, images_known), dim=0)
                        labels = torch.cat((labels_unknown, labels_known))


                    images = device(images)
                    labels = device(labels)

                    # Forward pass
                    logits, features = model(images)

                    # Calculate loss                       
                    j = loss_fn(logits, labels)
                    trackers["j"].update(j.item(), batch_len)
                        # Backward pass
                    j.backward()

                    optimizer.step()

                else: 
                    #we use half the batch size for the negative samples 
                    unknown_count = int(batch_len//2)
                    nr_neg = nr_neg +unknown_count

                    #randomly marking known and negative samples
                    labels_unknown = labels[:unknown_count]
                    labels_known = labels[unknown_count:]

                    images_unknown = images[:unknown_count]
                    images_known = images[unknown_count:]

                    #generate negative samples and corresponding labels
                    for el in range(len(images_unknown)):
                        noise = torch.randn(images_unknown[el].shape)                
                        images_unknown[el] = torch.clip(images_unknown[el] +(noise*pert),min = 0, max = 1)
                        labels_unknown[el] = -1 

                    images = torch.cat((images_unknown, images_known), dim=0)
                    labels = torch.cat((labels_unknown, labels_known))

                    images = device(images)
                    labels = device(labels)

                    # Forward pass
                    logits, features = model(images)

                    # Calculate loss                      
                    j = loss_fn(logits, labels)
                    trackers["j"].update(j.item(), batch_len)
                    
                    # Backward pass
                    j.backward()

                    optimizer.step()

        elif(method == 'mixup'):

            if(submethod == 'real'): 
                
                #use the style to mix up real negative samples and known samples
                if(style == 'mixup_real'):
                    filter_condition = labels == -1

                    count_real_neg = numpy.count_nonzero(filter_condition)

                    remaining_size = batch_len - count_real_neg
                    unknown_count = batch_len - count_real_neg

                    #we limit the additional generated negative samples to the maximum number defined in the config file
                    if(unknown_count > cfg_ns.number_neg_real):
                        unknown_count = cfg_ns.number_neg_real

                    indexes = numpy.where(labels != -1)
                    indexes_neg = numpy.where(labels == -1)
                    indexes = indexes[0]
                    indexes_neg = indexes_neg[0]

                    #check if we have enough known and real negative samples, otherwise we lower the mixing number
                    if(len(indexes_neg) > unknown_count): 
                        indexes_neg_mixup = indexes_neg[:unknown_count]
                        indexes_neg = indexes_neg[unknown_count:]
                    else:                      
                        unknown_count = len(indexes_neg)
                        indexes_neg_mixup = indexes_neg
                        indexes_neg = []

                    #counter
                    nr_neg = nr_neg + len(indexes_neg)

                    #if we have enough samples to do a mixup from the known and real negative data, we split them into four groups; real negative data, known data, candidates for the mixing from the known data, candidates for the mixing from the real negative data.
                    if(len(indexes_neg_mixup) >0): 
                        labels_neg_mixup = labels[indexes_neg_mixup]
                        images_neg_mixup = device(images[indexes_neg_mixup])

                        labels_neg = labels[indexes_neg]
                        images_neg = device(images[indexes_neg])

                        labels_kn = labels[indexes]
                        images_kn = device(images[indexes])

                        labels_unknown = labels_kn[:unknown_count]
                        labels_known = labels_kn[unknown_count:]

                        images_unknown = device(images_kn[:unknown_count])
                        images_known = device(images_kn[unknown_count:])

                        #start forward it through the network
                        unknown = model.first_blocks(images_unknown)
                        known = model.first_blocks(images_known)
                        neg = model.first_blocks(images_neg)
                        neg_mixup = model.first_blocks(images_neg_mixup)

                        unknown = model.maxpool_layer(unknown)
                        known = model.maxpool_layer(known)
                        neg = model.maxpool_layer(neg)
                        neg_mixup = model.maxpool_layer(neg_mixup)

                        unknown = model.first_layer(unknown)
                        known = model.first_layer(known)
                        neg = model.first_layer(neg)
                        neg_mixup = model.first_layer(neg_mixup)

                        #depending on the defined mixing layer in the config file, we will mix them up at 1,2,3,4 or 5.
                        #after mixing them we will patch all the groups together and forward it through the network
                        if(mixlayer == 1):     
                            
                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown,labels_unknown,neg_mixup, labels_neg_mixup)
                            feature_map = torch.cat((mixed_features, known,neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known,labels_neg))

                            nr_neg = nr_neg + len(mixed_labels)

                            x = model.second_layer(feature_map)
                            x = model.third_layer(x)
                            x = model.fourth_layer(x)
                            x = model.avgpool_layer(x) 
                        
                        if(mixlayer == 2): 
                            unknown = model.second_layer(unknown)
                            known = model.second_layer(known)
                            neg = model.second_layer(neg)
                            neg_mixup = model.second_layer(neg_mixup)

                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown,labels_unknown,neg_mixup, labels_neg_mixup)
                            feature_map = torch.cat((mixed_features, known,neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known,labels_neg))

                            nr_neg = nr_neg + len(mixed_labels)
                            
                            x = model.third_layer(feature_map)
                            x = model.fourth_layer(x)
                            x = model.avgpool_layer(x)

                        if(mixlayer == 4):
                            unknown = model.second_layer(unknown)
                            known = model.second_layer(known)
                            neg = model.second_layer(neg)
                            neg_mixup = model.second_layer(neg_mixup)

                            unknown = model.third_layer(unknown)
                            known = model.third_layer(known)
                            neg = model.third_layer(neg)
                            neg_mixup = model.third_layer(neg_mixup)

                            unknown = model.fourth_layer(unknown)
                            known = model.fourth_layer(known)
                            neg = model.fourth_layer(neg)
                            neg_mixup = model.fourth_layer(neg_mixup)

                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown,labels_unknown,neg_mixup, labels_neg_mixup)
                            feature_map = torch.cat((mixed_features, known,neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known,labels_neg))

                            nr_neg = nr_neg + len(mixed_labels)

                            x = model.avgpool_layer(feature_map)
                                    
                        if(mixlayer == 5):       
                            unknown = model.second_layer(unknown)
                            known = model.second_layer(known)
                            neg = model.second_layer(neg)
                            neg_mixup = model.second_layer(neg_mixup)

                            unknown = model.third_layer(unknown)
                            known = model.third_layer(known)
                            neg = model.third_layer(neg)
                            neg_mixup = model.third_layer(neg_mixup)

                            unknown = model.fourth_layer(unknown)
                            known = model.fourth_layer(known)
                            neg = model.fourth_layer(neg)
                            neg_mixup = model.fourth_layer(neg_mixup)

                            unknown = model.avgpool_layer(unknown)
                            known = model.avgpool_layer(known)
                            neg = model.avgpool_layer(neg)
                            neg_mixup = model.avgpool_layer(neg_mixup)

                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown,labels_unknown,neg_mixup, labels_neg_mixup)
                            feature_map = torch.cat((mixed_features, known,neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known,labels_neg))
                            nr_neg = nr_neg + len(mixed_labels)

                            
                        if(mixlayer == 3): # third layer default (PROSER Style)
                            
                            unknown = model.second_layer(unknown)
                            known = model.second_layer(known)
                            neg = model.second_layer(neg)
                            neg_mixup = model.second_layer(neg_mixup)

                            unknown = model.third_layer(unknown)
                            known = model.third_layer(known)
                            neg = model.third_layer(neg)
                            neg_mixup = model.third_layer(neg_mixup)

                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown,labels_unknown,neg_mixup, labels_neg_mixup)
                            feature_map = torch.cat((mixed_features, known,neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known,labels_neg))
                            nr_neg = nr_neg + len(mixed_labels)

                            x = model.fourth_layer(feature_map)
                            x = model.avgpool_layer(x)

                        logits, features = model.last_blocks(x)

                        # Calculate loss
                        j = loss_fn(logits, labels)
                        trackers["j"].update(j.item(), batch_len)
                        # Backward pass
                        j.backward()
                        optimizer.step()

                    #otherwise we do no mixup and forward it as usual through the network
                    else: 

                        print("no mixup")   
                        images = device(images)
                        labels = device(labels)

                        # Forward pass
                        logits, features = model(images)

                        print(nr_neg)

                        # Calculate loss
                        j = loss_fn(logits, labels)
                        trackers["j"].update(j.item(), batch_len)
                        # Backward pass
                        j.backward()
                        optimizer.step()


                else: 

                    filter_condition = labels == -1

                    count_real_neg = numpy.count_nonzero(filter_condition)
                
                    #counter
                    nr_neg = nr_neg + count_real_neg

                    #look at the known samples in a batch
                    remaining_size = batch_len - count_real_neg

                    unknown_count = batch_len - count_real_neg


                    #restricting the unknown count to the max defined number in the config file
                    if(unknown_count > cfg_ns.number_neg_real):
                        unknown_count = cfg_ns.number_neg_real

                    #if the unknown count is larger or equal 2, then we do a mixup (since we need two samples for that)
                    if(unknown_count >=2):
                        
                        #we adjust the number to fit the mixup algorithm (only even numbers)
                        while((unknown_count)%2 >0 and unknown_count != 2): 
                            unknown_count = unknown_count-1
                    
                        nr_neg = nr_neg + unknown_count

                        indexes = numpy.where(labels != -1)
                        indexes_neg = numpy.where(labels == -1)
                        indexes = indexes[0]
                        indexes_neg = indexes_neg[0]
                    
                        labels_neg = labels[indexes_neg]
                        images_neg = device(images[indexes_neg])

                        labels_kn = labels[indexes]
                        images_kn = device(images[indexes])

                        labels_unknown = labels_kn[:unknown_count]
                        labels_known = labels_kn[unknown_count:]

                        images_unknown = device(images_kn[:unknown_count])
                        images_known = device(images_kn[unknown_count:])

                        #start with the forward pass of the unknown (potential negative samples), the known and the real neg data through the network
                        unknown = model.first_blocks(images_unknown)
                        known = model.first_blocks(images_known)
                        neg = model.first_blocks(images_neg)

                        unknown = model.maxpool_layer(unknown)
                        known = model.maxpool_layer(known)
                        neg = model.maxpool_layer(neg)

                        unknown = model.first_layer(unknown)
                        known = model.first_layer(known)
                        neg = model.first_layer(neg)

                        #do the mixup on the respective layer
                        if(mixlayer == 1):
                            
                            mixed_features, mixed_labels,k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            feature_map = torch.cat((mixed_features, known,neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known,labels_neg))

                            x = model.second_layer(feature_map)
                            x = model.third_layer(x)
                            x = model.fourth_layer(x)
                            x = model.avgpool_layer(x) 
                        
                        if(mixlayer == 2): 
                            unknown = model.second_layer(unknown)
                            known = model.second_layer(known)
                            neg = model.second_layer(neg)

                            mixed_features, mixed_labels, k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            feature_map = torch.cat((mixed_features, known,neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known,labels_neg))

                            x = model.third_layer(feature_map)
                            x = model.fourth_layer(x)
                            x = model.avgpool_layer(x)

                        if(mixlayer == 4):
                            unknown = model.second_layer(unknown)
                            known = model.second_layer(known)
                            neg = model.second_layer(neg)

                            unknown = model.third_layer(unknown)
                            known = model.third_layer(known)
                            neg = model.third_layer(neg)


                            unknown = model.fourth_layer(unknown)
                            known = model.fourth_layer(known)
                            neg = model.fourth_layer(neg)

                            mixed_features, mixed_labels,k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            feature_map = torch.cat((mixed_features, known, neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known, labels_neg))

                            x = model.avgpool_layer(feature_map)
                                    
                        if(mixlayer == 5):       
                            unknown = model.second_layer(unknown)
                            known = model.second_layer(known)
                            neg = model.second_layer(neg)

                            unknown = model.third_layer(unknown)
                            known = model.third_layer(known)
                            neg = model.third_layer(neg)

                            unknown = model.fourth_layer(unknown)
                            known = model.fourth_layer(known)
                            neg = model.fourth_layer(neg)

                            unknown = model.avgpool_layer(unknown)
                            known = model.avgpool_layer(known)
                            neg = model.avgpool_layer(neg)

                            mixed_features, mixed_labels,k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            x = torch.cat((mixed_features, known, neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known, labels_neg))
                            
                        if(mixlayer == 3): # third layer default (PROSER Style)
                            
                            unknown = model.second_layer(unknown)
                            known = model.second_layer(known)
                            neg = model.second_layer(neg)

                            unknown = model.third_layer(unknown)
                            known = model.third_layer(known)
                            neg = model.third_layer(neg)

                            mixed_features, mixed_labels,k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            feature_map = torch.cat((mixed_features, known,neg), dim=0)
                            labels = torch.cat((mixed_labels, labels_known, labels_neg))

                            x = model.fourth_layer(feature_map)
                            x = model.avgpool_layer(x)


                        logits, features = model.last_blocks(x)

                        # Calculate loss

                        j = loss_fn(logits, labels)
                        trackers["j"].update(j.item(), batch_len)
                        # Backward pass
                        j.backward()
                        optimizer.step()

                    else: 
                        #if we do not have enough samples, we do no mixup and forward pass it as usual
                        print("no mixup")   
                        images = device(images)
                        labels = device(labels)

                        # Forward pass
                        logits, features = model(images)

                        # Calculate loss
                        j = loss_fn(logits, labels)
                        trackers["j"].update(j.item(), batch_len)
                        # Backward pass
                        j.backward()
                        optimizer.step()

            
            
            
            else:
                #we filter the known samples based on the logit values
                if(filter_method == 'logits'):

                    #get the top 2 labels based on the logits values for each sample
                    model.eval()
                    with torch.no_grad(): 
                        images = device(images)
                        labels= device(labels)
                        # Forward pass
                        logits, features = model(images)

                        top_pairs = torch.unique(torch.topk(logits, 2, dim = 1).indices, dim =0)
                                        
                    idx_1 =[]
                    idx_2 = []
                    total_idx = []
                    #check if there exists already a combination with this label and if we have both labels in the current batch
                    for pairs in range(len(top_pairs)): 
                        first_val = top_pairs[pairs][0].item()
                        second_val = top_pairs[pairs][1].item()

                        if(first_val in labels and second_val in labels and (first_val not in total_idx) and (second_val not in total_idx)): 
                            total_idx.append(first_val)
                            total_idx.append(second_val)
                            idx_1.append(first_val)
                            idx_2.append(second_val)

                    model.train()
                    
                    #if we have enough mixing candidate pair, we select them and mix them 
                    if(len(idx_1) >0):
                        unknown_count = len(idx_1)+len(idx_2)
                        #images = device(images)
                        #labels = device(labels)

                        idx_1 = torch.tensor(idx_1, device = 'cuda:0')
                        idx_2 = torch.tensor(idx_2, device = 'cuda:0')
                        total_idx = torch.tensor(total_idx, device = 'cuda:0')

                        #filter only unique values
                        indexes_1 = torch.unique(torch.where(torch.isin(labels, idx_1))[0]).tolist()
                        indexes_2 = torch.where(torch.isin(labels,idx_2))[0].tolist()
                        total_indexes = torch.where(torch.isin(labels, total_idx))[0].tolist()

                        if(len(indexes_1) != len(idx_1)): 
                            indexes_1 = []
                    
                            for element in idx_1: 
                                index = torch.where(torch.isin(labels,element))[0][0].item()
                                indexes_1.append(index)
                        
                        if(len(indexes_2) != len(idx_2)): 
                            indexes_2 = []
                    
                            for element in idx_2: 
                                index = torch.where(torch.isin(labels,element))[0][0].item()
                                indexes_2.append(index)
                                

                        labels_unknown_1 = labels[indexes_1]
                        labels_unknown_2 = labels[indexes_2]

                        labels_known_temp = labels.detach().clone()

                        images_unknown_1 = images[indexes_1]
                        images_unknown_2 = images[indexes_2]

                        labels_unknown_total_mask = torch.ones(len(labels),dtype = torch.bool)
                        labels_unknown_total_mask[total_indexes] = False

                        labels_known = labels[labels_unknown_total_mask]
                        images_known = images[labels_unknown_total_mask]
                        
                        #forward three groups through the network, known samples, candidate 1 for mixing and candidate 2 for mixing.
                        unknown_1 = model.first_blocks(device(images_unknown_1))
                        unknown_2 = model.first_blocks(device(images_unknown_2))
                        known = model.first_blocks(device(images_known))

                        unknown_1 = model.maxpool_layer(unknown_1)
                        unknown_2 = model.maxpool_layer(unknown_2)
                        known = model.maxpool_layer(known)

                        unknown_1 = model.first_layer(unknown_1)
                        unknown_2 = model.first_layer(unknown_2)
                        known = model.first_layer(known)

                        if(mixlayer == 1):
                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown_1,labels_unknown_1,unknown_2, labels_unknown_2)
                            feature_map = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))

                            nr_neg = nr_neg + len(mixed_labels)

                            x = model.second_layer(feature_map)
                            x = model.third_layer(x)
                            x = model.fourth_layer(x)
                            x = model.avgpool_layer(x) 
                        
                        if(mixlayer == 2): 
                            unknown_1 = model.second_layer(unknown_1)
                            unknown_2 = model.second_layer(unknown_2)
                            known = model.second_layer(known)

                            
                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown_1,labels_unknown_1,unknown_2, labels_unknown_2)
                            feature_map = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))

                            nr_neg = nr_neg + len(mixed_labels)
                            x = model.third_layer(feature_map)
                            x = model.fourth_layer(x)
                            x = model.avgpool_layer(x)

                        if(mixlayer == 4):
                            unknown_1 = model.second_layer(unknown_1)
                            unknown_2 = model.second_layer(unknown_2)
                            known = model.second_layer(known)

                            unknown_1 = model.third_layer(unknown_1)
                            unknown_2 = model.third_layer(unknown_2)
                            known = model.third_layer(known)

                            unknown_1 = model.fourth_layer(unknown_1)
                            unknown_2 = model.fourth_layer(unknown_2)
                            known = model.fourth_layer(known)

                            
                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown_1,labels_unknown_1,unknown_2, labels_unknown_2)
                            feature_map = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))

                            nr_neg = nr_neg + len(mixed_labels)

                            x = model.avgpool_layer(feature_map)
                                    
                        if(mixlayer == 5):       
                            unknown_1 = model.second_layer(unknown_1)
                            unknown_2 = model.second_layer(unknown_2)
                            known = model.second_layer(known)

                            unknown_1 = model.third_layer(unknown_1)
                            unknown_2 = model.third_layer(unknown_2)
                            known = model.third_layer(known)

                            unknown_1 = model.fourth_layer(unknown_1)
                            unknown_2 = model.fourth_layer(unknown_2)
                            known = model.fourth_layer(known)

                            unknown_1 = model.avgpool_layer(unknown_1)
                            unknown_2 = model.avgpool_layer(unknown_2)
                            known = model.avgpool_layer(known)

                            
                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown_1,labels_unknown_1,unknown_2, labels_unknown_2)
                            x = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))

                            nr_neg = nr_neg + len(mixed_labels)
                            
                        if(mixlayer == 3): # third layer default (PROSER Style)
                            
                            unknown_1 = model.second_layer(unknown_1)
                            unknown_2 = model.second_layer(unknown_2)
                            known = model.second_layer(known)

                            unknown_1 = model.third_layer(unknown_1)
                            unknown_2 = model.third_layer(unknown_2)
                            known = model.third_layer(known)

                            mixed_features, mixed_labels = mix_algo_real(beta_distribution,unknown_1,labels_unknown_1,unknown_2, labels_unknown_2)
                            feature_map = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))

                            nr_neg = nr_neg + len(mixed_labels)

                            x = model.fourth_layer(feature_map)
                            x = model.avgpool_layer(x)

                        logits, features = model.last_blocks(x)
                        # print(len(logits))
                        # print(len(labels))

                        # Calculate loss

                        j = loss_fn(logits, labels)
                        trackers["j"].update(j.item(), batch_len)
                        # Backward pass
                        j.backward()
                        optimizer.step()

                    else:
                        #if we have no mixing pairs
                        print("no neg samples included")
                        images = device(images)
                        labels = device(labels)

                        # Forward pass
                        logits, features = model(images)

                        # Calculate loss
                        j = loss_fn(logits, labels)
                        trackers["j"].update(j.item(), batch_len)
                        # Backward pass
                        j.backward()
                        optimizer.step()

                else: 

                    unknown_count = int(batch_len//2)
                    nr_neg = nr_neg + unknown_count

                    images = device(images)
                    labels = device(labels)
                    labels_unknown = labels[:unknown_count]
                    labels_known = labels[unknown_count:]

                    unknown = model.first_blocks(images[:unknown_count])
                    known = model.first_blocks(images[unknown_count:])

                    unknown = model.maxpool_layer(unknown)
                    known = model.maxpool_layer(known)

                    unknown = model.first_layer(unknown)
                    known = model.first_layer(known)
                    if(mixlayer == 1):

                        if(style == 'mixup'):
                            mixed_features, mixed_labels,k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            feature_map = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))

                        x = model.second_layer(feature_map)
                        x = model.third_layer(x)
                        x = model.fourth_layer(x)
                        x = model.avgpool_layer(x) 
                    
                    if(mixlayer == 2): 
                        unknown = model.second_layer(unknown)
                        known = model.second_layer(known)

                        if(style == 'mixup'):
                            mixed_features, mixed_labels, k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            feature_map = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))

                        x = model.third_layer(feature_map)
                        x = model.fourth_layer(x)
                        x = model.avgpool_layer(x)

                    if(mixlayer == 4):
                        unknown = model.second_layer(unknown)
                        known = model.second_layer(known)

                        unknown = model.third_layer(unknown)
                        known = model.third_layer(known)

                        unknown = model.fourth_layer(unknown)
                        known = model.fourth_layer(known)

                        if(style == 'mixup'):
                            mixed_features, mixed_labels,k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            feature_map = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))

                        x = model.avgpool_layer(feature_map)
                                
                    if(mixlayer == 5):       
                        unknown = model.second_layer(unknown)
                        known = model.second_layer(known)

                        unknown = model.third_layer(unknown)
                        known = model.third_layer(known)

                        unknown = model.fourth_layer(unknown)
                        known = model.fourth_layer(known)

                        unknown = model.avgpool_layer(unknown)
                        known = model.avgpool_layer(known)

                        if(style == 'mixup'):
                            mixed_features, mixed_labels,k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            x = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))
                        
                    if(mixlayer == 3): # third layer default (PROSER Style)
                        
                        unknown = model.second_layer(unknown)
                        known = model.second_layer(known)

                        unknown = model.third_layer(unknown)
                        known = model.third_layer(known)

                        if(style == 'mixup'):
                            mixed_features, mixed_labels,k = mix_algo(beta_distribution,unknown,labels_unknown,k)
                            feature_map = torch.cat((mixed_features, known), dim=0)
                            labels = torch.cat((mixed_labels, labels_known))

                        x = model.fourth_layer(feature_map)
                        x = model.avgpool_layer(x)


                    logits, features = model.last_blocks(x)
                    #print(len(logits))
                    #print(len(labels))

                    # Calculate loss

                    j = loss_fn(logits, labels)
                    trackers["j"].update(j.item(), batch_len)
                    # Backward pass
                    j.backward()
                    optimizer.step()

        #no negative samples       
        else:
            print("no neg samples included")
            images = device(images)
            labels = device(labels)

            # Forward pass
            logits, features = model(images)

            # Calculate loss
            j = loss_fn(logits, labels)
            trackers["j"].update(j.item(), batch_len)
            # Backward pass
            j.backward()
            optimizer.step()

    return nr_neg



def validate(model, data_loader, loss_fn, n_classes, trackers, cfg, cfg_ns, waiting):
    """ Validation loop.
    Args:
        model (torch.model): Model
        data_loader (torch dataloader): DataLoader
        loss_fn: Loss function
        n_classes(int): Total number of classes
        trackers(dict): Dictionary of trackers
        cfg: General configuration structure
    """
    # Reset all validation metrics
    for metric in trackers.values():
        metric.reset()

    if cfg.loss.type == "garbage":
        min_unk_score = 0.
        unknown_class = n_classes - 1
        last_valid_class = -1
    else:
        min_unk_score = 1. / n_classes
        unknown_class = -1
        last_valid_class = None

    model.eval()
    with torch.no_grad():
        data_len = int(len(data_loader.dataset)/cfg.batch_size)*cfg.batch_size  # size of dataset
        all_targets = device(torch.empty((data_len,), dtype=torch.int64, requires_grad=False))
        all_scores = device(torch.empty((data_len, n_classes), requires_grad=False))

        for i, (images, labels) in enumerate(data_loader):
            batch_len = labels.shape[0]  # current batch size, last batch has different value
            images = device(images)
            labels = device(labels)
            logits, features = model(images)
            scores = torch.nn.functional.softmax(logits, dim=1)

            j = loss_fn(logits, labels)
            trackers["j"].update(j.item(), batch_len)

            # accumulate partial results in empty tensors
            start_ix = i * cfg.batch_size
            all_targets[start_ix: start_ix + batch_len] = labels
            all_scores[start_ix: start_ix + batch_len] = scores

        kn_conf, kn_count, neg_conf, neg_count = confidence(
            scores=all_scores,
            target_labels=all_targets,
            offset=min_unk_score,
            unknown_class = unknown_class,
            last_valid_class = last_valid_class)
        if kn_count:
            trackers["conf_kn"].update(kn_conf, kn_count)
        if neg_count:
            trackers["conf_unk"].update(neg_conf, neg_count)


def get_arrays(model, loader, garbage, pretty=False):
    """ Extract deep features, logits and targets for all dataset. Returns numpy arrays

    Args:
        model (torch model): Model.
        loader (torch dataloader): Data loader.
        garbage (bool): Whether to remove final logit value
    """
    model.eval()
    with torch.no_grad():
        data_len = len(loader.dataset)         # dataset length
        logits_dim = model.logits.out_features  # logits output classes
        if garbage:
            logits_dim -= 1
        features_dim = model.logits.in_features  # features dimensionality
        all_targets = torch.empty(data_len, device="cpu")  # store all targets
        all_logits = torch.empty((data_len, logits_dim), device="cpu")   # store all logits
        all_feat = torch.empty((data_len, features_dim), device="cpu")   # store all features
        all_scores = torch.empty((data_len, logits_dim), device="cpu")

        index = 0
        if pretty:
            loader = tqdm.tqdm(loader)
        for images, labels in loader:
            curr_b_size = labels.shape[0]  # current batch size, very last batch has different value
            images = device(images)
            labels = device(labels)
            logits, feature = model(images)
            # compute softmax scores
            scores = torch.nn.functional.softmax(logits, dim=1)
            # shall we remove the logits of the unknown class?
            # We do this AFTER computing softmax, of course.
            if garbage:
                logits = logits[:,:-1]
                scores = scores[:,:-1]
            # accumulate results in all_tensor
            all_targets[index:index + curr_b_size] = labels.detach().cpu()
            all_logits[index:index + curr_b_size] = logits.detach().cpu()
            all_feat[index:index + curr_b_size] = feature.detach().cpu()
            all_scores[index:index + curr_b_size] = scores.detach().cpu()
            index += curr_b_size
        return(
            all_targets.numpy(),
            all_logits.numpy(),
            all_feat.numpy(),
            all_scores.numpy())


def worker(cfg, cfg_ns, protocol):
    """ Main worker creates all required instances, trains and validates the model.
    Args:
        cfg (NameSpace): Configuration of the experiment
    """
    # referencing best score and setting seeds
    set_seeds(cfg.seed)
    method = cfg_ns.method

    BEST_SCORE = 0.0    # Best validation score
    START_EPOCH = 0     # Initial training epoch

    # Configure logger. Log only on first process. Validate only on first process.
#    msg_format = "{time:DD_MM_HH:mm} {message}"
    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
    logger.add(
        sink= pathlib.Path(cfg.output_directory) / cfg.log_name,
        format=msg_format,
        level="INFO",
        mode='w')

    # Set image transformations
    train_tr = transforms.Compose(
        [transforms.Resize(256),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(0.5),
         transforms.ToTensor()])

    val_tr = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    # create datasets
    train_file = pathlib.Path(cfg.data.train_file.format(cfg.protocol))
    train_file_kn = pathlib.Path(cfg.data.train_file_kn.format(cfg.protocol))
    val_file = pathlib.Path(cfg.data.val_file.format(cfg.protocol))

    if train_file.exists() and val_file.exists():
        train_ds = ImagenetDataset(
            csv_file=train_file,
            imagenet_path=cfg.data.imagenet_path,
            transform=train_tr
        )

        train_ds_kn = ImagenetDataset(
            csv_file=train_file_kn,
            imagenet_path=cfg.data.imagenet_path,
            transform=train_tr
        )


        val_ds = ImagenetDataset(
            csv_file=val_file,
            imagenet_path=cfg.data.imagenet_path,
            transform=val_tr
        )

        # If using garbage class, replaces label -1 to maximum label + 1
        if cfg.loss.type == "garbage":
            # Only change the unknown label of the training dataset
            train_ds.replace_negative_label()
            val_ds.replace_negative_label()
        elif cfg.loss.type == "softmax":
            # remove the negative label from softmax training set, not from val set!
            train_ds.remove_negative_label()
    else:
        raise FileNotFoundError("train/validation file does not exist")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last = True,
        num_workers=cfg.workers,
        pin_memory=True)

    train_loader_kn = DataLoader(
        train_ds_kn,
        drop_last = True,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last = True,
        num_workers=cfg.workers,
        pin_memory=True,)

    # setup device
    if cfg.gpu is not None:
        set_device_gpu(index=cfg.gpu)
    else:
        logger.warning("No GPU device selected, training will be extremely slow")
        set_device_cpu()

    #print(len(train_ds))
    #print(len(train_ds_kn))
    #print(len(val_ds))

 

    # Callbacks
    early_stopping = None
    if cfg.patience > 0:
        early_stopping = EarlyStopping(patience=cfg.patience)

    # Set dictionaries to keep track of the losses
    t_metrics = defaultdict(AverageMeter)
    v_metrics = defaultdict(AverageMeter)

    # set loss
    loss = None
    if cfg.loss.type == "entropic":
        # number of classes - 1 since we have no label for unknown
        n_classes = train_ds.label_count-1
        #print(train_ds.unique_classes)
    else:
        # number of classes when training with extra garbage class for unknowns, or when unknowns are removed
        n_classes = train_ds.label_count

    if cfg.loss.type == "entropic":
        # We select entropic loss using the unknown class weights from the config file
        loss = EntropicOpensetLoss(n_classes, cfg.loss.w)
    elif cfg.loss.type == "softmax":
        # We need to ignore the index only for validation loss computation
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif cfg.loss.type == "garbage":
        # We use balanced class weights
        if(method != 'real'):
            class_weights = device(train_ds_kn.calculate_class_weights())
            loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        else: 
            class_weights = device(train_ds.calculate_class_weights())
            loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Create the model
    model = ResNet50Layers(fc_layer_dim=n_classes,
                     out_features=n_classes,
                     logit_bias=False)

    device(model)

    # Create optimizer
    if cfg.opt.type == "sgd":
        opt = torch.optim.SGD(params=model.parameters(), lr=cfg.opt.lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(params=model.parameters(), lr=cfg.opt.lr)

    # Learning rate scheduler
    if cfg.opt.decay > 0:
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=cfg.opt.decay,
            gamma=cfg.opt.gamma,
            verbose=True)
    else:
        scheduler = None


        # Resume a training from a checkpoint
    if cfg.checkpoint is not None:
        # Get the relative path of the checkpoint wrt train.py
        START_EPOCH, BEST_SCORE = load_checkpoint(
            model=model,
            checkpoint=cfg.checkpoint,
            opt=opt,
            scheduler=scheduler)
        logger.info(f"Best score of loaded model: {BEST_SCORE:.3f}. 0 is for fine tuning")
        logger.info(f"Loaded {cfg.checkpoint} at epoch {START_EPOCH}")


    # Print info to console and setup summary writer

    # Info on console
    logger.info("============ Data ============")
    logger.info(f"train_len incl. neg samples:{len(train_ds)}, labels:{train_ds.label_count}")
    logger.info(f"val_len:{len(val_ds)}, labels:{val_ds.label_count}")
    logger.info("========== Training ==========")
    logger.info(f"Initial epoch: {START_EPOCH}")
    logger.info(f"Last epoch: {cfg.epochs}")
    logger.info(f"Waiting period: {cfg_ns.waiting}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"workers: {cfg.workers}")
    logger.info(f"Loss: {cfg.loss.type}")
    logger.info(f"optimizer: {cfg.opt.type}")
    logger.info(f"Learning rate: {cfg.opt.lr}")
    logger.info(f"Device: {cfg.gpu}")
    logger.info(f"Negative samples method: {cfg_ns.method}")
    logger.info(f"Negative samples submethod: {cfg_ns.submethod}")
    logger.info(f"Negative samples style: {cfg_ns.style}")
    logger.info(f"Noise: {cfg_ns.noise}")  
    logger.info(f"Real methods: Nr. of generated neg. samples incl.: {cfg_ns.number_neg_real}")   
    logger.info(f"Adaptive Noise: {cfg_ns.adaptive_noise}")
    logger.info(f"Mixup: alpha: {cfg_ns.alpha}")
    logger.info(f"Mixup: mixing layer: {cfg_ns.mixlayer}")
    logger.info(f"Experiment number: {cfg.experiment_number}")
    logger.info("Training...")

    writer = SummaryWriter(log_dir=cfg.output_directory, filename_suffix="-"+cfg.log_name)

    waiting_epoch = cfg_ns.waiting
    waiting = True

    submethod = cfg_ns.submethod

    #write a csv file with all relevant information of the logger file + number of negative samples (only works for threshold method)
    absolute_path = os.path.abspath("experiments/Protocol_"+str(protocol)+"/"+cfg_ns.method+"/"+cfg.loss.type+"_threshold_"+str(cfg.experiment_number)+".csv")
    csv_conf = open(absolute_path, 'w')
    writer_csv = csv.writer(csv_conf)
    writer_csv.writerow(['epoch', 'conf_kn', 'conf_unk', 'score','training_loss', 'val_loss','nr_neg'])
    for epoch in range(START_EPOCH, cfg.epochs):
        epoch_time = time.time()

        waiting = epoch+1 <= waiting_epoch
        
        # training loop
        if(method == 'real'):
            nr_neg = train(
                model=model,
                data_loader=train_loader,
                optimizer=opt,
                loss_fn=loss,
                trackers=t_metrics,
                cfg=cfg,
                cfg_ns = cfg_ns,
                waiting = waiting,
                current_epoch = epoch,
                nr_neg = 0)
            
            train_time = time.time() - epoch_time


            validate(
                model=model,
                data_loader=val_loader,
                loss_fn=loss,
                n_classes=n_classes,
                trackers=v_metrics,
                cfg=cfg,
                cfg_ns = cfg_ns,
                waiting = waiting)
        else:
            if(submethod == 'real'):
                nr_neg =  train(
                        model=model,
                        data_loader=train_loader,
                        optimizer=opt,
                        loss_fn=loss,
                        trackers=t_metrics,
                        cfg=cfg,
                        cfg_ns = cfg_ns,
                        waiting = waiting,
                        current_epoch = epoch,
                        nr_neg = 0)
            else: 
                nr_neg =  train(
                        model=model,
                        data_loader=train_loader_kn,
                        optimizer=opt,
                        loss_fn=loss,
                        trackers=t_metrics,
                        cfg=cfg,
                        cfg_ns = cfg_ns,
                        waiting = waiting,
                        current_epoch = epoch,
                        nr_neg = 0)

            train_time = time.time() - epoch_time

            
            validate(
                model=model,
                data_loader=val_loader,
                loss_fn=loss,
                n_classes=n_classes,
                trackers=v_metrics,
                cfg=cfg,
                cfg_ns = cfg_ns,
                waiting = waiting)
        
        

        curr_score = v_metrics["conf_kn"].avg + v_metrics["conf_unk"].avg

        # learning rate scheduler step
        if cfg.opt.decay > 0:
            scheduler.step()

        # Logging metrics to tensorboard object
        writer.add_scalar("train/loss", t_metrics["j"].avg, epoch)
        writer.add_scalar("val/loss", v_metrics["j"].avg, epoch)
        # Validation metrics
        writer.add_scalar("val/conf_kn", v_metrics["conf_kn"].avg, epoch)
        writer.add_scalar("val/conf_unk", v_metrics["conf_unk"].avg, epoch)

        #  training information on console
        # validation+metrics writer+save model time
        val_time = time.time() - train_time - epoch_time
        def pretty_print(d):
            #return ",".join(f'{k}: {float(v):1.3f}' for k,v in dict(d).items())
            return dict(d)

        logger.info(
            f"loss:{cfg.loss.type} "
            f"protocol:{cfg.protocol} "
            f"ep:{epoch} "
            f"train:{pretty_print(t_metrics)} "
            f"val:{pretty_print(v_metrics)} "
            f"t:{train_time:.1f}s "
            f"v:{val_time:.1f}s")

        # save best model and current model
        ckpt_name = cfg.model_path.format(cfg.output_directory+"/Protocol_"+str(protocol)+"/"+cfg_ns.method, cfg.loss.type, "threshold", "curr", str(cfg.experiment_number))
        save_checkpoint(ckpt_name, model, epoch, opt, curr_score, scheduler=scheduler)
        list_row = [epoch ,v_metrics["conf_kn"].avg, v_metrics["conf_unk"].avg, curr_score, t_metrics["j"].avg, v_metrics["j"].avg,nr_neg]
        writer_csv.writerow(list_row)

        if curr_score > BEST_SCORE:
            BEST_SCORE = curr_score
            ckpt_name = cfg.model_path.format(cfg.output_directory+"/Protocol_"+str(protocol)+"/"+cfg_ns.method, cfg.loss.type, "threshold", "best", str(cfg.experiment_number))
            # ckpt_name = f"{cfg.name}_best.pth"  # best model
            logger.info(f"Saving best model {ckpt_name} at epoch: {epoch}")
            save_checkpoint(ckpt_name, model, epoch, opt, BEST_SCORE, scheduler=scheduler)

        # Early stopping
        if cfg.patience > 0:
            early_stopping(metrics=curr_score, loss=False)
            if early_stopping.early_stop:
                logger.info("early stop")
                break

    # clean everything
    del model
    csv_conf.close()
    logger.info("Training finished")
