import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

os.chdir(os.path.dirname(__file__))
print("CWD:", os.getcwd())

import argparse
import numpy as np 
import cv2
from tqdm import tqdm
import natsort
import glob
import re
from collections import OrderedDict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import higher
import kornia as K
from tensorboardX import SummaryWriter
from datasets import MultiFramesDataset, create_data_loader
from losses_test_time_adapt import inner_loop_loss_perc_affine as inner_loop_loss
import utils
from networks.GPWC_module.get_global_pwc_module import get_global_pwc as GPWC
import copy

SKIP = False
INNER_ITER_COUNTER = 0
OUTER_ITER_COUNTER = 0
LBLS_GIVEN = True
USE_PREV_CROP_LOCS = False
PREV_CROP_H = 0
PREV_CROP_W = 0


def get_model(opts):
    if opts.model == "DMBVS":
        from models.model_enet import Generator as Model
        model = Model(in_channels=15, out_channels=3, residual_blocks=64)
        state_dict = torch.load(os.path.join("pretrained_models", opts.ckp_name))['model']
        ##### create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        print(f'Loaded ckpt from {os.path.join("pretrained_models", opts.ckp_name)}')
        return model

    else:
        raise Exception("Model not implemented: (%s)" %opts.model)



def get_test_dataset(opts):
    test_dataset = MultiFramesDataset(mode= "val", opts= opts)
    return test_dataset



def ip_gen_DMBVS_recurrent(frames, frames_op, start_id, device= "cuda", return_orig= False):
    if len(frames_op) < 2:
        if len(frames_op) == 0:
            frames_op.append(frames[start_id].clone().detach())
            frames_op.append(frames[start_id + 1].clone().detach())
        if len(frames_op) == 1:
            frames_op.append(frames[start_id + 1])
    if device == "cuda":
        ip = torch.cat([frames_op[-2].cpu().detach(),
            frames_op[-1].cpu().detach(),
            frames[start_id + 2],
            frames[start_id + 3],
            frames[start_id + 4]], 1).cuda()
        if return_orig:
            temp = frames[start_id + 2].clone().cuda()
    else:
        ip = torch.cat([frames_op[-2].cpu().detach(),
            frames_op[-1].cpu().detach(),
            frames[start_id + 2],
            frames[start_id + 3],
            frames[start_id + 4]], 1)
        if return_orig:
            temp = frames[start_id + 2].clone()
    if return_orig:
        return ip, temp
    else:
        return ip


def ip_gen_DMBVS_recurrent_test(frames, frames_op, start_id, device= "cuda", patch_size= 330): 
    b, c, h, w = frames[0].shape
    global USE_PREV_CROP_LOCS
    global PREV_CROP_H
    global PREV_CROP_W
    
    if not USE_PREV_CROP_LOCS:
        if h == patch_size:
            h_ = 0
        else:
            h_ = np.random.randint(0, h - patch_size)
        w_ = np.random.randint(0, w - patch_size)
        PREV_CROP_H = h_
        PREV_CROP_W = w_
        USE_PREV_CROP_LOCS = True
    else:
        h_ = PREV_CROP_H
        w_ = PREV_CROP_H
    
    if len(frames_op) < 2:
        if len(frames_op) == 0:
            frames_op.append(utils.tensor_crop(frames[start_id], h_, w_, patch_size).clone().detach())
            frames_op.append(utils.tensor_crop(frames[start_id + 1], h_, w_, patch_size).clone().detach())
        if len(frames_op) == 1:
            frames_op.append(utils.tensor_crop(frames[start_id + 1], h_, w_, patch_size))

    if device == "cuda":
        ip = torch.cat([frames_op[-2].detach().cpu(),
            frames_op[-1].detach().cpu(),
            utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 3], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 4], h_, w_, patch_size)], 1).cuda()
        spt = utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size).cuda()
    else:
        ip = torch.cat([frames_op[-2].detach().cpu(),
            frames_op[-1].detach().cpu(),
            utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 3], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 4], h_, w_, patch_size)], 1)
        spt = utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size)

    return ip, spt


def test_no_adaptation(db, net, device, opts, video_ptr):
    net.eval()
    
    final_op_list = []
    final_ip_list = []
    frame_pointer = 0
    
    test_batch = db[video_ptr]
    frames = test_batch['X']
    video_name = test_batch['meta_data']["video_name"].split("/")[-2]
    save_path = "./evaluation_study_DMBVSRECURRENT/output_" + opts.model_name + "_" + str(opts.epoch_to_test) + "_comparative/" + video_name
    no_adapt_path = save_path + "/no_adapt/"
    unstab_path = save_path + "/unstable/"
    
    
    if not os.path.exists(no_adapt_path):
        os.makedirs(no_adapt_path)
    if not os.path.exists(unstab_path):
        os.makedirs(unstab_path)
    
    
    pbar = tqdm(total= len(frames) - 5 - 3)
    
    interim_ops = []
    while(frame_pointer < len(frames) - 5 - 3):        
        with torch.no_grad():
            ip_ = ip_gen_DMBVS_recurrent(frames, interim_ops, frame_pointer)
            op_ = torch.clamp(net(ip_), min= 0.0, max= 1.0)
            interim_ops.append(op_)
            final_op_list.append((utils.tensor2img(op_)*255).astype(np.uint8))
            final_ip_list.append((utils.tensor2img(frames[frame_pointer + 2])*255).astype(np.uint8))
            
        pbar.update(1)
        frame_pointer += 1
    pbar.close()

    print(f"Writing {no_adapt_path}, {unstab_path}")
    for i in range(len(final_op_list)):
        cv2.imwrite(no_adapt_path + str(i) + ".png", final_op_list[i])
    for i in range(len(final_ip_list)):
        cv2.imwrite(unstab_path + str(i) + ".png", final_ip_list[i])
    
    print("Written unstable and prestable videos...")


def test(db, net, device, L_in, opts, video_ptr):
    final_op_list = []
    final_ip_list = []
    frame_pointer = 0

    test_batch = db[video_ptr]
    frames = test_batch['X']
    video_name = test_batch['meta_data']["video_name"].split("/")[-2]
    save_path = "./evaluation_study_DMBVSRECURRENT/output_" + opts.model_name + "_" + str(opts.epoch_to_test) + "_comparative/" + video_name
    base_path = save_path
    save_path = save_path + "/adapted/"
    os.makedirs(save_path, exist_ok= True)
    
    
    lr = opts.lr_init
    adaptation_number = opts.adpt_number
    frame_pad = 8
    optimizer = optim.Adam(net.parameters(), lr)
    pbar = tqdm(total= len(frames) - adaptation_number - frame_pad)
    while(frame_pointer < len(frames) - 5 - 5):
        interim_op = []
        USE_PREV_CROP_LOCS = False
        for _ in range(adaptation_number):
            optimizer.zero_grad()
            optimizer.zero_grad()
            ip1, spt_ip1 = ip_gen_DMBVS_recurrent_test(frames, interim_op, start_id= frame_pointer, patch_size= opts.crop_size_adap)
            spt_op1 = torch.clamp(net(ip1), min= 0.0, max= 1.0)
            interim_op.append(spt_op1.clone())
            
            ip2, spt_ip2 = ip_gen_DMBVS_recurrent_test(frames, interim_op, start_id= frame_pointer + 1, patch_size= opts.crop_size_adap)
            spt_op2 = torch.clamp(net(ip2), min= 0.0, max= 1.0)
            interim_op.append(spt_op2.clone())

            ip3, spt_ip3 = ip_gen_DMBVS_recurrent_test(frames, interim_op, start_id= frame_pointer + 2, patch_size= opts.crop_size_adap)
            spt_op3 = torch.clamp(net(ip3), min= 0.0, max= 1.0)
            interim_op.append(spt_op3.clone())

            ip4, spt_ip4 = ip_gen_DMBVS_recurrent_test(frames, interim_op, start_id= frame_pointer + 3, patch_size= opts.crop_size_adap)
            spt_op4 = torch.clamp(net(ip4), min= 0.0, max= 1.0)
            interim_op.append(spt_op4.clone())

            ip5, spt_ip5 = ip_gen_DMBVS_recurrent_test(frames, interim_op, start_id= frame_pointer + 4, patch_size= opts.crop_size_adap)
            spt_op5 = torch.clamp(net(ip5), min= 0.0, max= 1.0)
            interim_op.append(spt_op5.clone())
            #spt_loss = L_in(spt_op1, spt_op2, spt_op3, spt_ip1, spt_ip2, spt_ip3)
            spt_loss = 0.5 * L_in([spt_op1, spt_op2, spt_op3, spt_op4, spt_op5], [spt_ip1, spt_ip2, spt_ip3, spt_ip4, spt_ip5])
        
            spt_loss.backward(retain_graph= True)
            optimizer.step()
        
        
        pbar.update(1)
        frame_pointer += 1
    pbar.close()

    frame_pointer = 0
    interim_ops = []
    
    while(frame_pointer < len(frames) - 5 - 5): 
        with torch.no_grad():
            net.eval()
            ip_ = ip_gen_DMBVS_recurrent(frames, interim_ops, frame_pointer)
            op_ = torch.clamp(net(ip_), min= 0.0, max= 1.0)
            interim_ops.append(op_.clone())
            final_op_list.append((utils.tensor2img(op_)*255).astype(np.uint8))
            final_ip_list.append((utils.tensor2img(frames[frame_pointer + 1])*255).astype(np.uint8))
            frame_pointer += 1
    

    for i in range(len(final_op_list)):
        cv2.imwrite(save_path + str(i) + ".png", final_op_list[i])
    
    print("Written adapted video to:", save_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Meta Stabilization")

    ### model options
    parser.add_argument('-model',           type=str,     default="DMBVS",                          help='Model to use')
    parser.add_argument('-ckp_name',        type=str,     default="best_stab_vf_v1.h5",             help='baseline checkpoint name') 
    parser.add_argument('-model_name',      type=str,     default='rec_eval',                       help='path to save model') #### Choices: {"rec_eval": DMBVSr, "eval": DMBVS, "dif_eval": DIFRINT}
    parser.add_argument('-adpt_number',     type=int,     default=2,                                help='adaptation iterations')
    parser.add_argument('-lambda1',         type=int,     default=1,                                help='lambda1')
    parser.add_argument('-lambda2',         type=int,     default=1,                                help='lambda2')
    parser.add_argument('-lambda3',         type=int,     default=1,                                help='lambda3')

    ### dataset options
    parser.add_argument('-train_data_dir',  type=str,     default='test_data/',                     help='path to train data folder')
    parser.add_argument('-data_dir',        type=str,     default='test_data/',                     help='path to test data folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='pretrained_models/checkpoints',  help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=320,                              help='patch size')
    parser.add_argument('-crop_size_adap',  type=int,     default=320,                              help='patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=1,                                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=0,                                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.5,                              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,                              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=100,                              help='#frames for training')

    ### training options
    parser.add_argument('-solver',          type=str,     default="ADAM",                           choices=["SGD", "ADAM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,                              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,                              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,                            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                                help='weight decay')
    parser.add_argument('-batch_size',      type=int,     default=1,                                help='training batch size')
    parser.add_argument('-train_epoch_size',type=int,     default=20,                               help='train epoch size')
    parser.add_argument('-valid_epoch_size',type=int,     default=100,                              help='valid epoch size')
    parser.add_argument('-epoch_max',       type=int,     default=100,                              help='max #epochs')


    ### learning rate options
    parser.add_argument('-lr_init',         type=float,   default=1e-6,                             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,                               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=20,                               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,                              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.1,                              help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    

    ### other options
    parser.add_argument('-unstab_coeff',    type=float,   default=0.25,                             help="artificial unstabilizer set 0 for training on unstable videos only...")
    parser.add_argument('-loss',            type=str,     default="L1",                             help="Loss [Options: L1, L2]")
    parser.add_argument('-seed',            type=int,     default=9487,                             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=8,                                help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',                               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=0,                                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                                    help='use cpu?')
    


    ### testing options
    parser.add_argument('-epoch_to_test',   type=int,   default= 57,                               help="epoch to load the model from...")
    opts = parser.parse_args()


    #### adjust the testing data folder
    # opts.data_dir = "../data/"
    opts.data_dir = "../ds_bm/frames4/"

    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m
    
    ### default model name
    if opts.model_name == 'none':
        opts.model_name = "%s_%s" %(opts.model, opts.ckp_name)

    if opts.suffix != "":
        opts.model_name += "_%s" %opts.suffix


    print(opts)

    ### model saving directory
    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Taking model ckpts from: %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)


    ### initialize loss writer
    loss_dir = "./evaluation_study_DMBVSRECURRENT/output_" + opts.model_name + "_" + str(opts.epoch_to_test) + "_comparative/loss/"
    loss_writer = SummaryWriter(loss_dir)


    ### initialize model
    print('===> Initializing model from %s...' %opts.model)
    model = get_model(opts)

    ### Meta optimizer...
    ### initialize optimizer
    if opts.solver == 'SGD':
        meta_opt = optim.SGD(model.parameters(), lr=opts.lr_init, momentum=opts.momentum, weight_decay=opts.weight_decay)
    elif opts.solver == 'ADAM':
        meta_opt = optim.Adam(model.parameters(), lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))
    else:
        raise Exception("Not supported solver (%s)" %opts.solver)

    if opts.loss == 'L2':
            criterion = nn.MSELoss(size_average=True)
    elif opts.loss == 'L1':
        criterion = nn.L1Loss(size_average=True)
    else:
        raise Exception("Unsupported criterion %s" %opts.loss)
    
    num_params = utils.count_network_parameters(model)

    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')

    
    GFlowNet = GPWC().eval()
    for param in GFlowNet.parameters():
        param.requires_grad = False

    ### convert to GPU
    device = torch.device("cuda" if opts.cuda else "cpu")
    model = model.to(device)
    baseline_model = copy.deepcopy(model)
    GFlowNet = GFlowNet.to(device)
    
    ### Define losses here and pass flownet and interpolator to losses
    L_in = inner_loop_loss(criterion, GFlowNet)

    test_dataset = get_test_dataset(opts)

    
    for video in range(len(test_dataset)):
        print("Processing videos: {}/{}".format(video + 1, len(test_dataset)))
        test_no_adaptation(test_dataset, model, device, opts, video)
        
        print("Testing with adaptation...")
        model, _ = utils.load_model(model, meta_opt, opts, opts.epoch_to_test)
        test(test_dataset, baseline_model, device, L_in, opts, video)
        