# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger
from PIL import Image
from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import json
import pandas as pd
import os

logger = getLogger()


def check_data_exist(data_frame, dataroot, mode):
    drop_indices = []
    for indx, row in data_frame.iterrows():
        image_id = row['image_id']
        if mode == "training_sup":
            image_path = os.path.join(dataroot, "sup_images/{}".format(image_id))
        elif mode == "training_unsup":
            image_path = os.path.join(dataroot, "unsup_images/{}".format(image_id))
        elif mode == "val":
            image_path = os.path.join(dataroot, "sup_images/{}".format(image_id))

        elif mode == "test":
            image_path = os.path.join(dataroot, "sup_images/{}".format(image_id))

        if not os.path.isfile(image_path):
            drop_indices.append(indx)
    return data_frame.drop(drop_indices)


class ClueDataset_Kate():
    def __init__(
            self,
            im_preprocessor,
            text_tokenizer,
            dataroot,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            batch_size=512,
            num_workers=25,
            size_dataset=-1,
            return_index=False,
            mode='training_sup',
            labels=['True', 'Meta', 'Action', 'Subjective', 'Story', 'Irrelevant', 'Other']
    ):
        # super(ClueDataset_Kate, self).__init__()
        if mode == 'training_unsup':
            data = pd.read_csv(os.path.join(dataroot, 'data_files/CC_Unsup_Extension.tsv'), sep='\t')
            data = check_data_exist(data, dataroot, mode)
            self.true_targets = None
        elif mode == 'training_sup':
            data = pd.read_csv(os.path.join(dataroot, 'data_files/train_small_400.csv'))
            data = check_data_exist(data, dataroot, mode)
            self.true_targets = np.array(data[labels].values)
        elif mode == 'val':
            data = pd.read_csv(os.path.join(dataroot, 'data_files/val.csv'))
            data = check_data_exist(data, dataroot, mode)
            self.true_targets = np.array(data[labels].values)
        elif mode == 'test':
            data = pd.read_csv(os.path.join(dataroot, 'data_files/test.csv'))
            data = check_data_exist(data, dataroot, mode)
            self.true_targets = np.array(data[labels].values)

        self.captions = list(data['caption'].array)
        self.image_names = list(data['image_id'].array)
        print("number of images :  " + str(len(self.image_names)))

        self.dataroot = dataroot
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.im_preprocessor = im_preprocessor
        self.text_tokenizer = text_tokenizer

        self.labels = labels
        self.mode = mode

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
                         ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):

        image_id = self.image_names[index]
        caption = self.captions[index]
        if self.mode == "training_sup":
            image_path = os.path.join(self.dataroot, "sup_images/{}".format(image_id))
            augmet = True
        elif self.mode == "training_unsup":
            image_path = os.path.join(self.dataroot, "unsup_images/{}".format(image_id))
            augmet = True
        elif self.mode == "val":
            image_path = os.path.join(self.dataroot, "sup_images/{}".format(image_id))
            augmet = False
        elif self.mode == "test":
            image_path = os.path.join(self.dataroot, "sup_images/{}".format(image_id))
            augmet = False

        image = Image.open(image_path).convert("RGB")

        if self.mode != 'training_unsup':
            target = self.true_targets[index]
        else:
            target = None
        caption = self.text_tokenizer(caption).reshape(-1)

        if augmet:
            multi_crops = list(map(lambda trans: trans(image), self.trans))
            assert len(multi_crops) == 2, "For now please specify only two multi-crops"
            assert multi_crops[0].shape[1] == 224, "1st image should be high resolution, i.e., 224 * 224"
            assert multi_crops[1].shape[1] == 224, "2n image should be low resolution, i.e., 96 * 96"

            high_res_image = multi_crops[0]
            low_res_image = multi_crops[1]
            # high_res_image = self.im_preprocessor(high_res_image)
            # low_res_image = self.im_preprocessor(low_res_image)

            if self.return_index:
                if target is not None:
                    return index, high_res_image, low_res_image, caption, target
                else:
                    index, high_res_image, low_res_image, caption,
            if target is not None:
                return high_res_image, low_res_image, caption, target
            else:
                return high_res_image, low_res_image, caption

        else:
            image = self.im_preprocessor(image)
            return image, caption, image_id, target

    def __len__(self):
        assert len(self.captions) == len(self.image_names), "number of captions is not equal to images"
        if self.mode != "training_unsup":
            assert len(self.captions) == len(self.true_targets), "number of captions is not equal to targets"
        return len(self.captions)


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
