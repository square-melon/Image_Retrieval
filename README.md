This is code for image retrieval by using learning-based method. Also, we add mutual acknowledgement and spliting picture to increase the accuracy.

Requirements:

    faiss-gpu=1.7.2
    pytorch
    skimage
    PIL(Pillow)

For scoring:
    To test images seperately:
    '''
    python main.py -m -sp -im image_to_test
    '''
    Directly test all images:
    '''
    python main.py -m -sp
    '''

Options:

    '''
    python main.py [-n NETWORK] [-d DATA_DIR] [-f FEATURE_DIR] [-r --RESULT_DIR] [-k TOP_K] [-im IMG_TEST] [-dn CROP_INTO_NxN] [-nin] [-sp] [-m] [-np] [-dr SPLIT_DROP_RATE]
    '''

Help:
    -n: network for retrieving
    -d: image data folder
    -f: feature folder
    -r: result folder
    -k: top k retrieved images
    -im: image to test
    -dn: divide images to n*n blocks
    -nin: do not do indexing
    -sp: use split method
    -m: use mutually acknowledgement method
    -np: do not copy retrieved images to result folder
    -dr: define the drop rate of splited feature

Reference:
    https://github.com/KhaLee2307/image-retrieval