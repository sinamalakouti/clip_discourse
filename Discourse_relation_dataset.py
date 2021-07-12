# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import json

import os

logger = getLogger()


class DiscourseRelationDataset(datasets.ImageFolder):
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
        is_training = True
    ):
        super(DiscourseRelationDataset, self).__init__(dataroot)
        caption_path = os.path.join(dataroot, "captions_all_json.json")
        self.captions = json.load(open(caption_path, "r"))
        self.image_name = list(self.captions.keys())
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.im_preprocessor = im_preprocessor
        self.text_tokenizer = text_tokenizer
        self.is_training = is_training
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
        if self.is_training:
            image_id = self.image_name[index]
            caption = self.captions[image_id]
            image_path = os.path.join(self.dataroot, "images/{}.jpg".format(image_id))
            image = self.loader(image_path)

            multi_crops = list(map(lambda trans: trans(image), self.trans))

            assert len(multi_crops) == 2, "For now please specify only two multi-crops"
            assert multi_crops[0].shape[1] == 224, "1st image should be high resolution, i.e., 224 * 224"
            assert multi_crops[1].shape[1] == 224,  "2n image should be low resolution, i.e., 96 * 96"

            high_res_image = multi_crops[0]
            low_res_image = multi_crops[1]
            # high_res_image = self.im_preprocessor(high_res_image)
            # low_res_image = self.im_preprocessor(low_res_image)
            caption = self.text_tokenizer(caption).reshape(-1)

            if self.return_index:
                return (index, high_res_image, low_res_image, caption)
            return (high_res_image, low_res_image, caption)
        else:
            image_id = self.image_name[index]
            caption = self.captions[image_id]
            image_path = os.path.join(self.dataroot, "images/{}.jpg".format(image_id))
            image = self.loader(image_path)
            image = self.im_preprocessor(image)
            caption = self.text_tokenizer(caption).reshape(-1)
            return (image, caption, image_id)
    def len(self):
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
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort