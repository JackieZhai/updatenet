import vot
from vot import Rectangle
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
import os
import pdb
from tqdm import tqdm
import argparse

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from utils import overlap_ratio

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, default='experiments/siammaske_r50_l3/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='experiments/siammaske_r50_l3/model.pth', help='model name')
args = parser.parse_args()
cfg.merge_from_file(args.config)
cfg.CUDA = torch.cuda.is_available() and cfg.CUDA

# create model
model = ModelBuilder()
# load model
model.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu()))
model.eval().cuda()

temp_path = 'updatenet_dataset'
if not os.path.isdir(temp_path):
    os.makedirs(temp_path)

videosets = open('../DAVIS/ImageSets/480p/trainval.txt','r')
videos = [line.strip() for line in videosets]
videos.sort()

template_acc = None; template_cur = None; template_gt = None
init0 = []; init = []; pre = []; gt = []  #init0 is reset init

init_rect = None; tracker = build_tracker(model); num_reset = None
tracker0 = build_tracker(model)
for v in tqdm(range(len(videos))):
    num = int(videos[v].split('/')[-1].split('.')[0])
    try:
        num_next = int(videos[v+1].split('/')[-1].split('.')[0])
    except:
        num_next = None
    img_path = videos[v].split(' ')[0]
    anno_path = videos[v].split(' ')[1]
    img = cv2.imread('../DAVIS/' + img_path)
    anno = cv2.imread('../DAVIS/' + anno_path, cv2.IMREAD_GRAYSCALE)
    anno = anno > 0
    anno_x = np.sum(anno, axis=1)
    anno_y = np.sum(anno, axis=0)
    for i in range(len(anno_x)):
        if anno_x[i] > 0:
            x = i
            break
    for i in range(len(anno_x)-1, -1, -1):
        if anno_x[i] > 0:
            w = i
            break
    for j in range(len(anno_y)):
        if anno_y[j] > 0:
            y = j
            break
    for j in range(len(anno_y)-1, -1, -1):
        if anno_y[j] > 0:
            h = j
            break
    w = w - x
    h = h - y
    img_rect = [x, y, w, h]
    tracker0.init(img, tuple(img_rect))
    if not template_gt:
            template_gt = tracker0.model.zf.cpu().data.numpy()
        else:
            np.concatenate(template_gt.append(tracker0.model.zf.cpu().data.numpy()))

    if num == 0:
        num_reset = 0
        init_rect = img_rect
        # build tracker
        tracker.init(img, tuple(init_rect))
        # ----------------
        if not template_acc:
            template_acc = tracker.model.zf.cpu().data.numpy()
        else:
            np.concatenate(template_acc.append(tracker.model.zf.cpu().data.numpy()))
        if not template_cur:
            template_cur = tracker.model.zf.cpu().data.numpy()
        else:
            np.concatenate(template_cur.append(tracker.model.zf.cpu().data.numpy()))
        init.append(num)
        init0.append(num_reset)
        pre.append(0)
        gt.append(1)
        # ----------------
    elif overlap_ratio(np.array(init_rect), np.array(img_rect)) > 0.05:
        num_reset += 1
        # execute tracker
        outputs = tracker.track(img)
        # ----------------
        if not template_acc:
            template_acc = tracker.model.zf.cpu().data.numpy()
        else:
            np.concatenate(template_acc.append(tracker.model.zf.cpu().data.numpy()))
        if not template_cur:
            template_cur = tracker.model.zf.cpu().data.numpy()
        else:
            np.concatenate(template_cur.append(outputs['xf'].cpu().data.numpy()))
        init.append(num)
        init0.append(num_reset)
        pre.append(1)
        if num_next and (num_next != 0):
            gt.append(1)
        else:
            gt.append(0)
        # ----------------
    else:
        num_reset = 0
        init_rect = img_rect
        tracker.init(img, tuple(init_rect))
        # ----------------
        if not template_acc:
            template_acc = tracker.model.zf.cpu().data.numpy()
        else:
            np.concatenate(template_acc.append(tracker.model.zf.cpu().data.numpy()))
        if not template_cur:
            template_cur = tracker.model.zf.cpu().data.numpy()
        else:
            np.concatenate(template_cur.append(tracker.model.zf.cpu().data.numpy()))
        init.append(num)
        init0.append(num_reset)
        pre.append(0)
        if num_next and (num_next != 0):
            gt.append(1)
        else:
            gt.append(0)
        # ----------------

np.save(temp_path+'/template.npy',template_acc); np.save(temp_path+'/templatei.npy',template_cur); np.save(temp_path+'/template0.npy',template_gt)
np.save(temp_path+'/init0.npy',init0); np.save(temp_path+'/init.npy',init);np.save(temp_path+'/pre.npy',pre);np.save(temp_path+'/gt.npy',gt)