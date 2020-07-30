import vot
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

from utils import get_axis_aligned_bbox, cxy_wh_2_rect, overlap_ratio

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

reset = 1
temp_path = 'updatenet_lasot_dataset'
if not os.path.isdir(temp_path):
    os.makedirs(temp_path)

video_path = '../LaSOT/'
category = os.listdir(video_path)
category.sort()

template_acc = None; template_cur = None; template_gt = None
init0 = []; init = []; pre = []; gt = []  #init0 is reset init

init_rect = None; tracker = build_tracker(model); num_reset = None
tracker0 = build_tracker(model)
for tmp_cat in category:
    videos = os.listdir(join(video_path, tmp_cat)); videos.sort()    
    for video in tqdm(videos):
        print(video)        
        gt_path = join(video_path,tmp_cat,video, 'groundtruth.txt')
        ground_truth = np.loadtxt(gt_path, delimiter=',')
        num_frames = len(ground_truth)
        img_path = join(video_path,tmp_cat,video, 'img')
        imgFiles = [join(img_path,'%08d.jpg') % i for i in range(1,num_frames+1)]
        frame = 0
        while frame < num_frames:
            Polygon = ground_truth[frame]
            if Polygon[2] * Polygon[3] != 0:
                image_file = imgFiles[frame]
                im = cv2.imread(image_file)  # HxWxC
                tracker0.init(im, tuple(Polygon))
                if template_gt is None:
                    template_gt = tracker0.model.zf.cpu().data.numpy()
                else:
                    template_gt = np.concatenate((template_gt, tracker0.model.zf.cpu().data.numpy()))
                tracker.init(im, tuple(Polygon))
                if template_acc is None:
                    template_acc = tracker.model.zf.cpu().data.numpy()
                else:
                    template_acc = np.concatenate((template_acc, tracker.model.zf.cpu().data.numpy()))
                if template_cur is None:
                    template_cur = tracker.model.zf.cpu().data.numpy()
                else:
                    template_cur = np.concatenate((template_cur, tracker.model.zf.cpu().data.numpy()))
                init0.append(0); init.append(frame); frame_reset=0; pre.append(0); gt.append(1)
                while frame < (num_frames-1):
                    frame = frame + 1; frame_reset=frame_reset+1
                    image_file = imgFiles[frame]
                    if not image_file:
                        break
                    im = cv2.imread(image_file)  # HxWxC
                    outputs = tracker.track(im)
                    if template_acc is None:
                        template_acc = tracker.model.zf.cpu().data.numpy()
                    else:
                        template_acc = np.concatenate((template_acc, tracker.model.zf.cpu().data.numpy()))
                    if template_cur is None:
                        template_cur = tracker.model.zf.cpu().data.numpy()
                    else:
                        template_cur = np.concatenate((template_cur, outputs['zf_cur'].cpu().data.numpy()))
                    init0.append(frame_reset); init.append(frame); pre.append(1)
                    if frame==(num_frames-1): #last frame
                        gt.append(0)
                    else:
                        gt.append(1)
                    if reset:                    
                        gt_Polygen = ground_truth[frame]
                        tracker0.init(im, tuple(gt_Polygen))
                        if template_gt is None:
                            template_gt = tracker0.model.zf.cpu().data.numpy()
                        else:
                            template_gt = np.concatenate((template_gt, tracker0.model.zf.cpu().data.numpy()))
                        iou = overlap_ratio(np.array(gt_Polygen), np.array(outputs['bbox']))
                        if iou <= 0:
                            break    
            else:
                template_acc = np.concatenate((template_acc, torch.zeros([1, 512, 6, 6], dtype=torch.float32)))
                template_cur = np.concatenate((template_cur, torch.zeros([1, 512, 6, 6], dtype=torch.float32)))
                init0.append(0); init.append(frame); pre.append(1)
                if frame==(num_frames-1): #last frame
                    gt.append(0)
                else:
                    gt.append(1)           
            frame = frame + 1 #skip

np.save(temp_path+'/template.npy',template_acc); np.save(temp_path+'/templatei.npy',template_cur); np.save(temp_path+'/template0.npy',template_gt)
np.save(temp_path+'/init0.npy',init0); np.save(temp_path+'/init.npy',init);np.save(temp_path+'/pre.npy',pre);np.save(temp_path+'/gt.npy',gt)