# Data Loader
from PIL import Image
import pathlib
import numpy as np
from src.imagesplit import image_split

from torchvision import transforms
from torch.utils.data import Dataset

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']


def get_transformation():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    return transform

class MyDataLoader(Dataset):

    def __init__(self, image_root, divide_n=2, split_pic=False, naming=False):
        self.image_root = pathlib.Path(image_root)
        self.image_list = list()
        for image_path in self.image_root.iterdir():
            if image_path.exists() and image_path.suffix.lower() in ACCEPTED_IMAGE_EXTS:
                self.image_list.append(image_path)
        self.image_list = sorted(self.image_list, key = lambda x: x.name)
        self.transform = get_transformation()
        self.split = split_pic
        self.naming = naming
        self.divide_n = divide_n

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        _img = self.image_list[index]
        _imgname = _img
        _img = Image.open(_img)

        _img = _img.convert("RGB")

        if self.split:
            sp_trans = []
            for sp in image_split(_img, self.divide_n):
                trans = self.transform(sp)
                sp_trans.append(trans)
            return self.transform(_img), str(self.image_list[index]), sp_trans
        if self.naming == True:
            return self.transform(_img), str(self.image_list[index]), np.array([_imgname])
        return self.transform(_img), str(self.image_list[index])