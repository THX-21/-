import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as tf
from PIL import Image
from collections import Counter


class ImgDataset(Dataset):
    def __init__(self, path,enhance=False, reduce_classes=None, reduction_factor=None):
        super(Dataset, self).__init__()
        self.dataset=pd.read_csv(path)
        self.imgs = self.dataset['image_name'].values
        self.labels = self.dataset['label'].values
        if reduce_classes is not None:
            if isinstance(reduction_factor, list):
                self.reduce_class_samples(reduce_classes, reduction_factor)
            elif isinstance(reduction_factor, (int, float)):
                self.reduce_class_samples(reduce_classes, [reduction_factor] * len(reduce_classes))

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
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open('scene_classification/imgs/'+self.imgs[index])
        img = self.mytf(img)
        label = self.labels[index]
        return img,label

    def count_label(self):
        label_counts = Counter(self.labels)
        return label_counts

    def reduce_class_samples(self, reduce_classes, reduction_factors):
        indices_to_keep = []
        for label in np.unique(self.labels):
            indices = np.where(self.labels == label)[0]
            if label in reduce_classes:
                reduction_factor = reduction_factors[reduce_classes.index(label)]
                retain_count = int(len(indices) * (1-reduction_factor))
                indices = np.random.choice(indices, retain_count, replace=False)
            indices_to_keep.extend(indices)

        self.imgs = self.imgs[indices_to_keep]
        self.labels = self.labels[indices_to_keep]


