# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from Discourse_relation_dataset import DiscourseRelationDataset
import clip
from Clip import DiscourseModel
import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import numpy as np
import json
from sklearn.metrics import precision_score, f1_score
import argparse
import yaml
from easydict import EasyDict as edict


labels = ["Visible", 'Subjective', 'Action', 'Story', 'Meta', 'Irrelevant', 'Other']
parser = argparse.ArgumentParser()


def evaluate(testdata_path, device, model, cfg):
    model.eval()
    batch_size = cfg['batch_size']
    target_path = os.path.join(testdata_path, "all_targets_json.json")

    all_targets = json.load(open(target_path, "r"))
    avg_avg = 0
    avg_sample = 0
    avg_micro = 0
    counter = 0

    test_set = DiscourseRelationDataset(
        clip_preprocess,
        clip.tokenize,
        testdata_path,
        [224, 224],
        [1, 1],
        min_scale_crops=[0.5, 0.14],
        max_scale_crops=[1., 0.5],
        batch_size=batch_size,
        num_workers=25,
        size_dataset=-1,
        return_index=False,
        is_training=False
    )

    test_sampler = RandomSampler(test_set)

    test_loader = DataLoader(
        test_set,
        sampler=test_sampler,
        batch_size=batch_size,
    )

    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(t.to(device=device, non_blocking=True) if type(t) == torch.Tensor else t for t in batch)
            images, captions, image_ids = batch
            true_targets = []
            for img_id in image_ids:
                true_targets.append(np.fromiter(all_targets[img_id].values(), dtype=np.float   ))
            true_targets = torch.from_numpy(np.array(true_targets))
            true_targets = true_targets.to(device)
            # model.double()
            model = model.to(device)
            logits, _ = model(
                high_res_images=images,
                low_res_images=None,
                texts=captions,
                is_supervised=True
            )

            discourse_prediction = torch.sigmoid(logits)
            discourse_prediction = discourse_prediction.to('cpu')
            true_targets = true_targets.to('cpu')
            res = compute_score(discourse_prediction, true_targets.type(torch.float), 0.5)
            avg_avg += res['weighted/f1']
            avg_micro += res['micro/f1']
            avg_sample += res['samples/f1']
            # pred[counter * batch_size: (counter + 1) * batch_size, :] = discourse_prediction
            counter += 1
        print("micro/f1 : {},   weighted/f1 : {},    samples/f1 : {}".format(avg_micro / counter, avg_avg / counter,
                                                                                 avg_sample / counter))
    model.train()


def compute_score(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)

    return {
        'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
        'weighted/f1': f1_score(y_true=target, y_pred=pred, average='weighted'),
        'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
    }


def train_val(cfg, arg):

    traindata_path = cfg['train_path']
    testdata_path = cfg['test_path']
    batch_size = cfg['batch_size']

    train_set = DiscourseRelationDataset(
        clip_preprocess,
        clip.tokenize,
        traindata_path,
        [224, 224],
        [1, 1],
        min_scale_crops=[0.5, 0.14],
        max_scale_crops=[1., 0.5],
        batch_size=batch_size,
        num_workers=25,
        size_dataset=-1,
        return_index=False,
    )

    print("******* mmain dataset sie**********")
    print(len(train_set))
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    train_sampler = RandomSampler(train_set)

    loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=batch_size,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    model = DiscourseModel(1024, 512, len(labels))
    #model.float()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg['lr'],
        momentum=0.9,
        weight_decay=1e-4,
    )
    n_epochs = cfg['n_epochs']
    model = model.to(device)
    for epoch in range(0, n_epochs):
        print("*********** EPOCH ITERATION :  {} *************".format(epoch))

        for batch in loader:
            model.train()
            is_sup = False
            batch = tuple(t.to(device=device, non_blocking=True) if type(t) == torch.Tensor else t for t in batch)
            (high_res, low_res, text) = batch
            out = model(high_res, low_res, text, is_sup)
            loss = model.compute_loss(out[0], out[1], loss_fn, False)
            print("********* TRAINING LOSSS: {} *************".format(loss.item()))
            # ============ backward and optim step ... ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("EVALUATION")
        evaluate(testdata_path, device, model, cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        default="0",
        type=str,
        help="cuda indices 0,1,2,3"
    )

    parser.add_argument(
        "--config",
        type=str,
        default='config.yaml'
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))
    return cfg, args
if __name__ == '__main__':
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    cfg, args = parse_args()
    train_val(cfg, args)

