
import time
import pathlib
from PIL import Image
from argparse import ArgumentParser
import os
import numpy as np

import torch
import faiss

from src.extractor import MyResnet50, MyVGG16, RGBHistogram, LBP
from src.dataloader import get_transformation
from src.imagesplit import image_split
from src.dataloader import MyDataLoader

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']


def get_image_list(image_root):
    dataloader = MyDataLoader(image_root, split_pic=False, naming=True)
    image_list = []
    for x in dataloader:
        image_list.append(x[2][0])
    return image_list

def retrieve(opt, img_test, k, ret_ret=True, no_split=False):

    args_feature_extractor = opt.network
    args_device = 'cuda:0'
    args_top_k = k + 1
    args_test_image_path = opt.data_dir
    image_root = opt.data_dir
    feature_root = opt.feature_dir
    split = opt.spliting and (not no_split)
    divide_n = opt.divide_n

    if split:
        droppers = np.load(os.path.join(feature_root, args_feature_extractor + '-droppers.npy'))

    if ret_ret == True:
        print('Start retrieving .......')

    device = torch.device(args_device)

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

    img_list = get_image_list(image_root)

    transform = get_transformation()

    # Preprocessing
    test_image_path = pathlib.Path(args_test_image_path)
    pil_image = Image.open(os.path.join(image_root, img_test))
    pil_image = pil_image.convert('RGB')
    image_tensor = transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(feature_root + '/' + args_feature_extractor + '.index.bin')
    _, indices = indexer.search(feat, k=args_top_k)
    retrieved_list = []

    if split:
        sp_feat = []
        ori_wgt = 1
        blocks = divide_n*divide_n
        sp_wgt = 1 / (blocks*blocks - blocks)
        retrieved_rank = {}
        for i, sp in enumerate(image_split(pil_image, divide_n)):
            sp_tensor = transform(sp)
            sp_tensor = sp_tensor.unsqueeze(0).to(device)
            sp_ft = extractor.extract_features(sp_tensor)
            sp_feat.append(sp_ft[:, droppers[i]])
        using_split_txt = '-split'
        sp_indexer = []
        for i in range(divide_n*divide_n):
            sp_indexer.append(faiss.read_index(feature_root + '/' + args_feature_extractor + using_split_txt + str(i) + '.index.bin'))

        for index in indices[0]:
            img_name = str(img_list[index]).replace(image_root, '')
            if img_name == img_test:
                continue
            if img_name in retrieved_rank:
                retrieved_rank[img_name] += ori_wgt
            else:
                retrieved_rank[img_name] = ori_wgt

        for i in range(divide_n*divide_n):
            for j in range(divide_n*divide_n):
                if i == j:
                    continue
                _, sp_indices = sp_indexer[i].search(sp_feat[j], k=args_top_k)
                for index in sp_indices[0]:
                    img_name = str(img_list[index]).replace(image_root, '')
                    if img_name == img_test:
                        continue
                    if img_name in retrieved_rank:
                        retrieved_rank[img_name] += sp_wgt
                    else:
                        retrieved_rank[img_name] = sp_wgt

        rank = sorted(retrieved_rank.items(), key=lambda x:x[1], reverse = True)
        for keyval in rank:
            img = keyval[0]
            if img == img_test:
                continue
            retrieved_list.append(os.path.join(image_root, img))
            if len(retrieved_list) == k:
                break
    else:
        for index in indices[0]:
            retrieved_list.append(img_list[index])

        
        

    return retrieved_list