import pandas as pd
import numpy  as np
import os
import torch
import clip
from PIL import Image
import sklearn


def get_all_data(dataroot):
    data = pd.read_csv(os.path.join(dataroot, 'data_files/train.csv'))
    return data


def check_data_exist(data_frame, dataroot):
    drop_indices = []
    for indx, row in data_frame.iterrows():
        image_id = row['image_id']
        image_path = os.path.join(dataroot, "sup_images/{}".format(image_id))

        if not os.path.isfile(image_path):
            drop_indices.append(indx)
    return data_frame.drop(drop_indices)


def compute_similarity(data, dataroot):
    cos_sim = []
    dot_prod = []
    model, preprocess = clip.load("ViT-B/32", device='cpu')

    for indx, row in data.iterrows():
        caption = row['caption']
        image_id = row['image_id']
        image_path = os.path.join(dataroot, "sup_images/{}".format(image_id))
        image = Image.open(image_path).convert("RGB")
        # image = preprocess(image)
        image = preprocess(image).unsqueeze(0).to('cpu')
        text = clip.tokenize(caption).to('cpu')

        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        # print("shapes")
        # print(image_features.shape)
        # print(text_features.shape)
        dot_product = torch.dot(image_features.reshape(-1), text_features.reshape(-1))
        cosine_sim = torch.cosine_similarity(image_features, text_features)
        cos_sim.append(cosine_sim)
        dot_prod.append(dot_product)

    data['cos_sim'] = cos_sim
    data['dot_product'] = dot_prod
    data.to_csv('~/Desktiop/data_similarity.csv')


def sim(data, dataroot):
    cos_sim = []
    dot_prod = []
    model, preprocess = clip.load("ViT-B/32", device='cpu')

    for indx, row in data.iterrows():
        caption = row['caption']
        image_id = row['image_id']
        image_path = os.path.join(dataroot, "sup_images/{}".format(image_id))
        image = Image.open(image_path).convert("RGB")
        # image = preprocess(image)
        image = preprocess(image).unsqueeze(0).to('cpu')
        text = clip.tokenize(caption).to('cpu')

        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        # print("shapes")
        # print(image_features.shape)
        # print(text_features.shape)
        dot_product = torch.dot(image_features.reshape(-1), text_features.reshape(-1))
        cosine_sim = torch.cosine_similarity(image_features, text_features)
        cos_sim.append(cosine_sim)
        dot_prod.append(dot_product)

    return cos_sim, dot_prod
