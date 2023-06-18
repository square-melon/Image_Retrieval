# Mutual Acknowledgement
import shutil
import time
import numpy as np
import os
from src.test import retrieve
from src.indexer import indexing
from src.parser import Parser


def main(opt, im=None, stop_indexing=False):
    start = time.time()

    img_dir = opt.data_dir
    result_dir = opt.result_dir
    extractor = opt.network
    img_test = opt.img_test
    if im != None:
        img_test = im
    feature_root = opt.feature_dir
    using_split = opt.spliting
    not_indexing = opt.not_indexing
    split_ft_drop = opt.split_drop
    mutual = opt.mutual
    nocopy = opt.nocopy

    k = opt.top_k

    k1 = int(k*1.5)
    k2 = k//4
    if k2 == 0:
        k2 = 1
    ori_weight = 1
    ret_weight = 0.25

    retrieved_rank = {}

    print('Using extractor:', extractor)
    if not_indexing == False and stop_indexing == False:
        indexing(opt)

    if mutual:
        print('Start retrieving .......')
        retrieved_list = retrieve(opt, img_test, k1, False)
        print('Retrieving from retrieved image:')
        for img in retrieved_list:
            img_name = str(img).replace(img_dir, '')
            if img_name == img_test:
                continue
            if img_name in retrieved_rank:
                retrieved_rank[img_name] += ori_weight
            else:
                retrieved_rank[img_name] = ori_weight
            ret_from_ret = retrieve(opt, img_name, k2, False, no_split=True)
            print(img_name, '-> ', end = '')
            for im in ret_from_ret:
                img_name2 = str(im).replace(img_dir, '')
                if img_name2 == img_name:
                    continue
                print(img_name2, '', end = '')
                if img_name2 in retrieved_rank:
                    retrieved_rank[img_name2] += ret_weight
                else:
                    retrieved_rank[img_name2] = ret_weight
            print()
    else:
        retrieved_list = retrieve(opt, img_test, k, True)
        for img in retrieved_list:
            img_name = str(img).replace(img_dir, '')
            if img_name == img_test:
                continue
            if img_name in retrieved_rank:
                retrieved_rank[img_name] += ori_weight
            else:
                retrieved_rank[img_name] = ori_weight
        print()

    rank = sorted(retrieved_rank.items(), key=lambda x:x[1], reverse = True)
    result_img = []

    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    print()
    print('Result (Saved in {}):'.format(result_dir))
    os.mkdir(result_dir)
    for keyval in rank:
        img = keyval[0]
        if img == img_test:
            continue
        img_path = os.path.join(img_dir, img)
        result_path = os.path.join(result_dir, img)
        result_img.append(img)
        if not nocopy:
            shutil.copyfile(img_path, result_path)
        print(img)
        if len(result_img) == k:
            break

    end = time.time()
    print('Finish in {:.2f} seconds'.format(end - start))

if __name__ == '__main__':
    parser = Parser()
    opt = parser.parse()
    if opt.img_test == 'test_all':
        img_list = ['1.jpg', '101.jpg', '201.jpg', '301.jpg', '401.jpg', '501.jpg', '601.jpg', '701.jpg', '801.jpg', '901.jpg']
        first = True
        for im in img_list:

            print('img:', im)
            if first:
                main(opt, im)
            else:
                main(opt, im, stop_indexing=True)
            first = False
            print()
    else:
        main(opt)