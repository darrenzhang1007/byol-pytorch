import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np

class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output

#创建一个类并继承Dataset
class allOrgansData(Dataset):
    def __init__(self, root_dir, transform):
        super(allOrgansData, self).__init__()

        #创建一个初始化方法为该类提供全局变量
        self.root_dir = root_dir
        #列表存储地址文件夹中所有图片的名字
        self.img_path = os.listdir(self.root_dir)
        self.transform = transform


    def __getitem__(self, idx):
        # 创建一个获取每一张图片地址和标签的方法
        # 获取idx索引中图片的名字
        img_name = self.img_path[idx]
        # 拼接该图片的完整路径
        img_item_path = os.path.join(self.root_dir, img_name)
        # 读取图片
        img = Image.open(img_item_path)
        img = img.convert('RGB')

        if self.transform is not None:
            batch_view = self.transform(img)

        return batch_view

    #计算该列表的长度
    def __len__(self):
        return len(self.img_path)


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img



def get_simclr_data_transforms(input_shape, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
                                          transforms.ToTensor()])
    return data_transforms

if __name__ == '__main__':
    #图片文件件的地址
    root_dir = r"F:\byol\data\unlabel"

    data_transform = get_simclr_data_transforms(input_shape='(256,256,3)')

    #创建一个对象存储蜜蜂对应文件夹的信息
    ants_dataset = allOrgansData(root_dir, MultiViewDataInjector([data_transform, data_transform]))

    #获取列表中第一张图片的信息
    img = ants_dataset

    #查看这张图片
    print(img.shape)
