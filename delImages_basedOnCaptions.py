from os import listdir
from os.path import isfile, join
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
import os
import json

from os import walk

caption_path = '/projects/sina/vilbert/discourse_project/vilbert-multi-task/data/discoursedata/train/captions_all_json.json'
images_root = '/projects/sina/vilbert/discourse_project/vilbert-multi-task/data/discoursedata/train/images'
onlyfiles = [f for f in listdir(images_root) if isfile(join(images_root, f))]
with open(caption_path) as json_file:
    captions= json.load(json_file)

print(captions)
h = 0
t = 0
for root, directories, files in os.walk(images_root, topdown=False):
	for f in files:

            name = f.split('.')[0]
#            print(name)

            if name not in captions:
                print(name)
                t +=1
            else:
                h+=1



print(h)
print(t)
print(len(captions))
