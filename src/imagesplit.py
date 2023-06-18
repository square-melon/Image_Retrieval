import numpy as np
from PIL import Image

def image_split(img, n):
    img_h = np.array(img).shape[0]
    img_w = np.array(img).shape[1]
    imgs = []
    for i in range(n):
        for j in range(n):
            xbeg = img_h//n*i
            xend = min(img_h//n*(i+1), img_h)
            ybeg = img_w//n*j
            yend = min(img_w//n*(j+1), img_w)
            _img = Image.fromarray(np.array(img)[xbeg:xend, ybeg:yend])
            imgs.append(_img)
    return imgs