# Indexing
import faiss
import time
import random
import os
import numpy as np

import faiss
import torch
from torch.utils.data import DataLoader, SequentialSampler
from src.extractor import MyResnet50, MyVGG16, RGBHistogram, LBP
from src.dataloader import MyDataLoader

def get_faiss_indexer(shape):
    indexer = faiss.IndexFlatL2(shape) # features.shape[1]
    return indexer


def random_drop(n, drop_rate):
    keep = int(n * (1-drop_rate))
    keep_idx = random.sample(range(n), keep)
    return keep_idx

def create_droppers(n, drop_rate, num):
    droppers = []
    for i in range(num):
        dropper = random_drop(n, drop_rate)
        droppers.append(dropper)
    return droppers

def indexing(opt=None):
    image_root = opt.data_dir
    feature_root = opt.feature_dir
    args_extractor = opt.network
    split = opt.spliting
    split_ft_drop = opt.split_drop
    divide_n = opt.divide_n

    print('Start indexing .......')

    args_feature_extractor = args_extractor
    args_device = 'cuda:0'
    args_test_image_path = image_root
    args_batch_size = 64

    device = torch.device(args_device)
    batch_size = args_batch_size

    # Load module feature extraction
    if (args_feature_extractor == 'Resnet50'):
        extractor = MyResnet50(device)
    elif (args_feature_extractor == 'VGG16'):
        extractor = MyVGG16(device)
    elif (args_feature_extractor == 'RGBHistogram'):
        extractor = RGBHistogram(device)
    elif (args_feature_extractor == 'LBP'):
        extractor = LBP(device)
    else:
        print("No matching model found")
        return

    dataset = MyDataLoader(image_root, split_pic=split, divide_n=divide_n)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,batch_size=batch_size,sampler=sampler)


    indexer = get_faiss_indexer(extractor.shape)
    if split == True:
        droppers = create_droppers(extractor.shape, split_ft_drop, divide_n*divide_n)
        dropper_len = len(droppers[0])
        split_indexers = []
        for i in range(divide_n*divide_n):
            split_indexer = get_faiss_indexer(dropper_len)
            split_indexers.append(split_indexer)

    if split == False:
        for images, image_paths in dataloader:
            images = images.to(device)
            features = extractor.extract_features(images)
            indexer.add(features)
    else:
        for images, image_paths, splits in dataloader:
            images = images.to(device)
            images_ft = extractor.extract_features(images)

            for i, sp in enumerate(splits):
                sp_d = sp.to(device)
                sp_ft = extractor.extract_features(sp_d)
                sp_ft = np.ascontiguousarray(sp_ft[:, droppers[i]])
                split_indexers[i].add(sp_ft)

            indexer.add(images_ft)

    # Save features
    using_split_txt = '-split' if split else ''
    faiss.write_index(indexer, feature_root + '/' + args_feature_extractor + '.index.bin')
    if split:
        for i, sp_indexer in enumerate(split_indexers):
            faiss.write_index(sp_indexer, feature_root + '/' + args_feature_extractor + using_split_txt + str(i) + '.index.bin')
        np.save(os.path.join(feature_root, args_feature_extractor + '-droppers'), droppers)

if __name__ == '__main__':
    indexing()