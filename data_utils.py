import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as tf
from PIL import Image


class ImgDataset(Dataset):
    def __init__(self, path,enhance=False):
        super(Dataset, self).__init__()
        self.dataset=pd.read_csv(path)
        self.imgs = self.dataset['image_name'].values
        self.labels = self.dataset['label'].values
        if enhance:
            self.mytf = tf.Compose([tf.Resize((224, 224)),
                                    tf.ToTensor(),
                                    tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    tf.RandomHorizontalFlip(0.5),
                                    tf.RandomVerticalFlip(0.5),
                                    tf.RandomGrayscale(0.4)])
        else:
            self.mytf = tf.Compose([tf.Resize((224, 224)),
                                    tf.ToTensor(),
                                    tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = Image.open('scene_classification/imgs/'+self.imgs[index])
        img = self.mytf(img)
        label = self.labels[index]
        return img,label



