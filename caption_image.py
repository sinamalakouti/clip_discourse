import argparse
import json
import logging
import os
import random
from io import open
import math
import sys
from PIL import Image
#import pandas as pd
#import requests

#f#rom time import gmtime, strftime
#from timeit import default_timer as timer

import numpy as np
#from tqdm import tqdm, trange

#import torch
#from torch.utils.data import DataLoader, Dataset, RandomSampler
#from torch.utils.data.distributed import DistributedSampler
#from tensorboardX import SummaryWriter
##
#from pytorch_transformers.tokenization_bert import BertTokenizer
#from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

#import utils as utils
#import torch.distributed as dist


dataroot = '/projects/sina/vilbert/discourse_project/vilbert-multi-task/data/discoursedata/test/'


caption_path = os.path.join(dataroot, "captions_all_json.json")
with open(caption_path) as json_file:
    captions= json.load(json_file)

h = captions.copy()

for k in captions.keys():
    img_path = os.path.join(dataroot, "images/{}.jpg".format(str(k)))
    try:
        img = Image.open(img_path)
    except:
        print(img_path)
        h.pop(k)

print(len(h))
with open(caption_path, 'w') as outfile:
    json.dump(h, outfile)
