import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.chdir(os.path.dirname(__file__))
print("CWD:", os.getcwd())
import argparse
import json
import random
import numpy as np
from GlobalFlowNets.GlobalPWCNets import getGlobalPWCModel_mine
#from Utils.VideoUtility import VideoReader, VideoWriter
import cv2
import torch
from GlobalFlowNets.raft_module_latest import get_raft_module
import flow_vis
import natsort

def read_img_as_tensor(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (1280, 704))
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)/255.0
    img = torch.tensor(img.astype(np.float32)).cuda()
    return img

def getConfig(filePath):
    with open(filePath, 'r') as openfile:
        config = json.load(openfile)
    return config

def flow_to_img(flo):
    flow_uv = flo.cpu().detach().numpy()
    flow_uv = np.transpose(flow_uv[0], (1, 2, 0))
    flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=True)
    return flow_color

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpVideoPath', dest='inpVideoPath', default='inputs/sample.avi')
    parser.add_argument('--outVideoPath', dest='outVideoPath', default='outputs/sample.avi')
    parser.add_argument('--maxAffineCrop', dest='maxAffineCrop', default=0.8, type=float)

    args = parser.parse_args()

    #inpVideo = VideoReader(args.inpVideoPath, maxFrames=30)
    #outVideo = VideoWriter(args.outVideoPath, fps=inpVideo.getFPS())

    Stabilizers = {}
    Stabilizers['GNetAffine'] = ['GNetAffine']
    Stabilizers['MSPhotometric'] = ['MSPhotometric']
    Stabilizers['GNetMSPhotometric'] = ['GNetAffine', 'MSPhotometric']

    inpTag = 'Original'
    outTag = 'GNetMSPhotometric'
    modelTag = 'GLNoWarp4YTBB'
    maxAffineCrop = .8

    #config = getConfig('GlobalFlowNets/trainedModels/config.json')['GlobalNetModelParameters']
    OptNet = getGlobalPWCModel_mine('./GFlowNet.pth')
    OptNet = OptNet.eval().cuda()
    raft = get_raft_module().eval().cuda()
#    im1 = read_img_as_tensor("./00800.png").cuda()
#    im2 = read_img_as_tensor("./00814.png").cuda()
#    im1 = torch.nn.functional.interpolate(im1, (320, 640))
#    im2 = torch.nn.functional.interpolate(im2, (320, 640))
#    flo_fourth = OptNet(im1, im2)
#    flo_full = OptNet.estimateFlowFull(im1, im2)
#    flo_raft = raft(im1, im2)[-1]

    root = "/home/ali/stab_new_repo/synth_data/"
    seqs = ["stable", "unstable"]
    for seq in seqs:
        fldr = root + seq + "/"
        lof = natsort.natsorted(os.listdir(fldr))
        for i in range(len(lof) - 1):
            print("Doing: {}/{}".format(i + 1, len(lof)))
            im1 = read_img_as_tensor(fldr + lof[i])
            im2 = read_img_as_tensor(fldr + lof[i + 1])
            save_fldr_raft = root + seq + "_of_raft" + "/"
            save_fldr_gpwc = root + seq + "_of_gpwc" + "/"
            os.makedirs(save_fldr_raft, exist_ok= True)
            os.makedirs(save_fldr_gpwc, exist_ok= True)
            flo_gpwc = OptNet.estimateFlowFull(im1, im2)
            flo_raft = raft(im1, im2)[-1]
            flo_gpwc = flow_to_img(flo_gpwc)
            flo_raft = flow_to_img(flo_raft)

            cv2.imwrite(save_fldr_gpwc + str(i) + ".png", flo_gpwc)
            cv2.imwrite(save_fldr_raft + str(i) + ".png", flo_raft)






    #flo_fourth = flow_to_img(flo_fourth)
    #flo_full = flow_to_img(flo_full)
    #flo_raft = flow_to_img(flo_raft)
    #cv2.imwrite("flo_fourth.png", flo_fourth)
    #cv2.imwrite("flo_full.png", flo_full)
    #cv2.imwrite("flo_raft.png", flo_raft)