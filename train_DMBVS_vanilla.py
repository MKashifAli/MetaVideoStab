import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
import torch.optim as optim
import higher
from tensorboardX import SummaryWriter
from datasets import MultiFramesDataset, create_data_loader
from losses import inner_loop_loss_perc_affine_another_try, outer_loop_loss_affine
import utils
from networks.GPWC_module.get_global_pwc_module import get_global_pwc as GPWC
import shutil

#### Some useless counters, cuz I'm lazy af...
INNER_ITER_COUNTER = 0
OUTER_ITER_COUNTER = 0
LBLS_GIVEN = True

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
        return model

    else:
        raise Exception("Model not implemented: (%s)" %opts.model)

def get_data_loaders(opts):
    bc_dir = opts.data_dir
    opts_train = opts
    opts_train.data_dir = opts.train_data_dir
    train_dataset = MultiFramesDataset(mode= 'train', opts= opts_train, get_lbls= LBLS_GIVEN, prev_iter_counter= OUTER_ITER_COUNTER)
    opts.data_dir = bc_dir
    test_dataset = MultiFramesDataset(mode= "val", opts= opts)
    train_data_loader = create_data_loader(train_dataset, mode= "train", 
        batch_size= opts.batch_size, threads= opts.threads, train_epoch_size= opts.train_epoch_size)
    test_data_loader = utils.create_data_loader(test_dataset, opts= opts, mode= "val")

    return train_data_loader, test_dataset # intentionally returning dataset instead of dataloader to gauge the performance on a single video [0]


def ip_gen_our(frames, start_id, device= "cuda", return_orig= False):
    if device == "cuda":
        ip = torch.cat([frames[start_id],
            frames[start_id + 1],
            frames[start_id + 2],
            frames[start_id + 3],
            frames[start_id + 4]], 1).cuda()
        if return_orig:
            temp = frames[start_id + 2].clone().cuda()
    else:
        ip = torch.cat([frames[start_id],
            frames[start_id + 1],
            frames[start_id + 2],
            frames[start_id + 3],
            frames[start_id + 4]], 1)
        if return_orig:
            temp = frames[start_id + 2].clone()
    if return_orig:
        return ip, temp
    else:
        return ip


def ip_gen_our_for_inner_train(frames, start_id, device= "cuda", return_orig= False, patch_size= 256):
    b, c, h, w = frames[0].shape
    h_ = np.random.randint(0, h - patch_size)
    w_ = np.random.randint(0, w - patch_size)
    if device == "cuda":
        ip = torch.cat([utils.tensor_crop(frames[start_id], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 1], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 3], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 4], h_, w_, patch_size)], 1).cuda()
        if return_orig:
            temp = utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size).clone().cuda()
    else:
        ip = torch.cat([utils.tensor_crop(frames[start_id], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 1], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 3], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 4], h_, w_, patch_size)], 1)
        if return_orig:
            temp = utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size).clone()
    if return_orig:
        return ip, temp
    else:
        return ip



def ip_gen_our_for_outer(frames, frames_lbl, start_id, device= "cuda", return_orig= True, resize= True):
    if device == "cuda":
        ip = torch.cat([frames[start_id],
            frames[start_id + 1],
            frames[start_id + 2],
            frames[start_id + 3],
            frames[start_id + 4]], 1).cuda()
        if return_orig:
            temp = frames_lbl[start_id + 2].clone().cuda()
    else:
        ip = torch.cat([frames[start_id],
            frames[start_id + 1],
            frames[start_id + 2],
            frames[start_id + 3],
            frames[start_id + 4]], 1)
        if return_orig:
            temp = frames_lbl[start_id + 2].clone()
    if return_orig:
        if resize:
            ip = nn.functional.interpolate(ip, (192, 192))
            temp = nn.functional.interpolate(temp, (192, 192))
        return ip, temp
    else:
        ip = nn.functional.interpolate(ip, (192, 192))
        return ip



def ip_gen_our_test(frames, start_id, device= "cuda", patch_size= 320):
    b, c, h, w = frames[0].shape
    h_ = np.random.randint(0, h - patch_size)
    w_ = np.random.randint(0, w - patch_size)
    
    if device == "cuda":
        ip = torch.cat([utils.tensor_crop(frames[start_id], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 1], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 3], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 4], h_, w_, patch_size)], 1).cuda()
        spt = utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size).cuda()
    else:
        ip = torch.cat([utils.tensor_crop(frames[start_id], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 1], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 3], h_, w_, patch_size),
            utils.tensor_crop(frames[start_id + 4], h_, w_, patch_size)], 1)
        spt = utils.tensor_crop(frames[start_id + 2], h_, w_, patch_size)

    return ip, spt

def ip_gen_our_test_full_image(frames, start_id, device= "cuda"): 
    if device == "cuda":
        ip = torch.cat([frames[start_id],
            frames[start_id + 1],
            frames[start_id + 2],
            frames[start_id + 3],
            frames[start_id + 4]], 1).cuda()
        spt = frames[start_id + 2].cuda()
    else:
        ip = torch.cat([frames[start_id],
            frames[start_id + 1],
            frames[start_id + 2],
            frames[start_id + 3],
            frames[start_id + 4]], 1)
        spt = frames[start_id + 2]

    return ip, spt


def test_no_adaptation(db, net, device, L_in, epoch, opts):
    if not os.path.exists("./output_" + opts.model_name + "/" + str(epoch) + "/"):
        os.makedirs("./output_" + opts.model_name + "/" + str(epoch) + "/")

    if not os.path.exists("./output_" + opts.model_name + "/target/"):
        os.makedirs("./output_" + opts.model_name + "/target/")

    net.train()
    
    final_op_list = []
    final_ip_list = []
    frame_pointer = 0

    test_batch = db[0]
    frames = test_batch['X']
    
    pbar = tqdm(total= len(frames) - 5 - 5)
    while(frame_pointer < len(frames) - 5 - 5):        
        with torch.no_grad():
            net.eval()
            ip_ = ip_gen_our(frames, frame_pointer, return_orig= False)
            op_ = torch.clamp(net(ip_), min= 0.0, max= 1.0)
            final_op_list.append((utils.tensor2img(op_)*255).astype(np.uint8))
            final_ip_list.append((utils.tensor2img(frames[frame_pointer + 2])*255).astype(np.uint8))
            net.train()
        pbar.update(1)
        frame_pointer += 1
    pbar.close()
    for i in range(len(final_op_list)):
        cv2.imwrite("./output_" + opts.model_name + "/" + str(epoch) + "/" + str(i) + ".png", final_op_list[i])
    for i in range(len(final_ip_list)):
        cv2.imwrite("./output_" + opts.model_name + "/target/" + str(i) + ".png", final_ip_list[i])

    return final_op_list


def eval_no_adaptation(db, net, device, L_in, epoch, opts):
    if not os.path.exists("./output_" + opts.model_name + "/" + str(epoch) + "_no_adapt/"):
        os.makedirs("./output_" + opts.model_name + "/" + str(epoch) + "_no_adapt/")

    if not os.path.exists("./output_" + opts.model_name + "/target/"):
        os.makedirs("./output_" + opts.model_name + "/target/")

    net.train()
    
    final_op_list = []
    final_ip_list = []
    frame_pointer = 0

    test_batch = db[0]
    frames = test_batch['X']
    
    pbar = tqdm(total= len(frames) - 5 - 5)
    while(frame_pointer < len(frames) - 5 - 5):        
        with torch.no_grad():
            net.eval()
            ip_ = ip_gen_our(frames, frame_pointer, return_orig= False)
            op_ = torch.clamp(net(ip_), min= 0.0, max= 1.0)
            final_op_list.append((utils.tensor2img(op_)*255).astype(np.uint8))
            final_ip_list.append((utils.tensor2img(frames[frame_pointer + 2])*255).astype(np.uint8))
            net.train()
        pbar.update(1)
        frame_pointer += 1
    pbar.close()
    for i in range(len(final_op_list)):
        cv2.imwrite("./output_" + opts.model_name + "/" + str(epoch) + "_no_adapt/" + str(i) + ".png", final_op_list[i])
    for i in range(len(final_ip_list)):
        cv2.imwrite("./output_" + opts.model_name + "/target/" + str(i) + ".png", final_ip_list[i])


    return final_op_list


def test(db, net, device, L_in, epoch, opts):
    if not os.path.exists("./output_" + opts.model_name + "/" + str(epoch) + "/"):
        os.makedirs("./output_" + opts.model_name + "/" + str(epoch) + "/")
    net.train()
    
    final_op_list = []
    final_ip_list = []
    frame_pointer = 0

    test_batch = db[0]
    frames = test_batch['X']
    
    lr = 5e-7
    adaptation_number = 3
    optimizer = optim.Adam(net.parameters(), lr)
    pbar = tqdm(total= len(frames) - 5 - 5)
    while(frame_pointer < len(frames) - 5 - 5):
        for _ in range(adaptation_number):
            optimizer.zero_grad()
            ip1, spt_ip1 = ip_gen_our_test(frames, start_id= frame_pointer, patch_size= opts.crop_size_adap)
            ip2, spt_ip2 = ip_gen_our_test(frames, start_id= frame_pointer + 1, patch_size= opts.crop_size_adap)
            ip3, spt_ip3 = ip_gen_our_test(frames, start_id= frame_pointer + 2, patch_size= opts.crop_size_adap)
            ip4, spt_ip4 = ip_gen_our_test(frames, start_id= frame_pointer + 3, patch_size= opts.crop_size_adap)
            ip5, spt_ip5 = ip_gen_our_test(frames, start_id= frame_pointer + 4, patch_size= opts.crop_size_adap)
            
            spt_op1 = torch.clamp(net(ip1), min= 0.0, max= 1.0)
            spt_op2 = torch.clamp(net(ip2), min= 0.0, max= 1.0)
            spt_op3 = torch.clamp(net(ip3), min= 0.0, max= 1.0)
            spt_op4 = torch.clamp(net(ip4), min= 0.0, max= 1.0)
            spt_op5 = torch.clamp(net(ip5), min= 0.0, max= 1.0)

            spt_loss = 0.5 * L_in([spt_op1, spt_op2, spt_op3, spt_op4, spt_op5], [spt_ip1, spt_ip2, spt_ip3, spt_ip4, spt_ip5])
        
            spt_loss.backward()
            optimizer.step()
        
        
        pbar.update(1)
        frame_pointer += 1
    pbar.close()
    frame_pointer = 0
    while(frame_pointer < len(frames) - 5 - 5): 
        with torch.no_grad():
            net.eval()
            ip_ = ip_gen_our(frames, frame_pointer)
            op_ = torch.clamp(net(ip_), min= 0.0, max= 1.0)
            final_op_list.append((utils.tensor2img(op_)*255).astype(np.uint8))
            final_ip_list.append((utils.tensor2img(frames[frame_pointer + 2])*255).astype(np.uint8))
            frame_pointer += 1
        net.train()
    for i in range(len(final_op_list)):
        cv2.imwrite("./output_" + opts.model_name + "/" + str(epoch) + "/" + str(i) + ".png", final_op_list[i])
        
    return final_op_list


def train(data_loader, net, device, meta_opt, L_in, L_out, epoch, opts):
    net.train()
    global INNER_ITER_COUNTER
    global OUTER_ITER_COUNTER
    if epoch > opts.lr_offset:
        current_lr = utils.learning_rate_decay_simple(opts, epoch)
    else:
        current_lr = opts.lr_init
    for param_group in meta_opt.param_groups:
        param_group['lr'] = current_lr

    loss_writer.add_scalar("hyperparameters/LR", current_lr, epoch)

    for batch_idx, batch in enumerate(data_loader, 1):
        frames = batch['X']
        if LBLS_GIVEN:
            frames_lbl = batch['Y']
        frame_inds = [0, 7, 13, 21, 29] #### for 5 inner loop iterations
        n_inner_iter = 1 #len(frame_inds) #for doing 1 inner loop optimization...
        
        inner_opt = torch.optim.Adam(net.parameters(), lr=current_lr)
        
        meta_opt.zero_grad()
        for i in range(1):
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                for ind in range(n_inner_iter):
                    ip1, unstab1 = ip_gen_our(frames, start_id= frame_inds[ind], return_orig= True)
                    ip2, unstab2 = ip_gen_our(frames, start_id= frame_inds[ind] + 1, return_orig= True)
                    ip3, unstab3 = ip_gen_our(frames, start_id= frame_inds[ind] + 2, return_orig= True)
                    ip4, unstab4 = ip_gen_our(frames, start_id= frame_inds[ind] + 3, return_orig= True)
                    ip5, unstab5 = ip_gen_our(frames, start_id= frame_inds[ind] + 4, return_orig= True)
                    spt_op1 = torch.clamp(fnet(ip1), min= 0.0, max= 1.0)
                    spt_op2 = torch.clamp(fnet(ip2), min= 0.0, max= 1.0)
                    spt_op3 = torch.clamp(fnet(ip3), min= 0.0, max= 1.0)
                    spt_op4 = torch.clamp(fnet(ip4), min= 0.0, max= 1.0)
                    spt_op5 = torch.clamp(fnet(ip5), min= 0.0, max= 1.0)
                    
                    spt_loss = L_in([spt_op1, spt_op2, spt_op3, spt_op4, spt_op5], [unstab1, unstab2, unstab3, unstab4, unstab5])
                    diffopt.step(spt_loss)
                    loss_writer.add_scalar("comb_losses/InnerLoopLoss", spt_loss.item(), INNER_ITER_COUNTER)
                    INNER_ITER_COUNTER += 1
                    print("Epoch:", epoch, "Outer_iter:", batch_idx, "Inner_iter:", ind, "L_in:", spt_loss.item(), "({}/{})".format(batch_idx, len(data_loader)))

                                
                op_qry = []
                stab_qry = []

                for q in range(6):
                    id_for_qry = frame_inds[-1] + 3 + q
                    if LBLS_GIVEN:
                        ip_qry, stab = ip_gen_our_for_outer(frames, frames_lbl, start_id= id_for_qry)
                    else:
                        ip_qry, stab = ip_gen_our_for_outer(frames, frames, start_id= id_for_qry)
                    op_qry.append(torch.clamp(fnet(ip_qry), min= 0.0, max= 1.0))
                    stab_qry.append(stab)
                
                qry_loss = L_out(op_qry, stab_qry)
                
                loss_writer.add_scalar("comb_losses/OuterLoopLoss", qry_loss.item(), OUTER_ITER_COUNTER)
                OUTER_ITER_COUNTER += 1                
                qry_loss.backward()
                print("Epoch:", epoch, "Outer_iter:", batch_idx, "Inner_iter:", ind + 1, "L_out:", qry_loss.item(), "({}/{})".format(batch_idx, len(data_loader)))

            meta_opt.step()
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return net, meta_opt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Meta Stabilization")

    ### model options
    parser.add_argument('-model',           type=str,     default="DMBVS",                          help='Model to use')
    parser.add_argument('-ckp_name',        type=str,     default="best_stab_vf_v1.h5",             help='checkpoint_name') 
    parser.add_argument('-model_name',      type=str,     default='DMBVS_vanilla',                  help='path to save model')

    ### dataset options
    parser.add_argument('-train_data_dir',  type=str,     default='../DeepStab/unstable_frames/',   help='path to train data folder')
    parser.add_argument('-train_lbl_dir',   type=str,     default='../DeepStab/stable_frames/',     help='path to train data folder')
    parser.add_argument('-data_dir',        type=str,     default='../DeepStab/unstable_frames/',   help='path to test data folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',                    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=192,                              help='patch size')
    parser.add_argument('-crop_size_adap',  type=int,     default=320,                              help='adaptation patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=0,                                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.5,                              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,                              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=63,                               help='#frames for training') #### Min number of frames in one of the deep stab scenes are 67...

    ### training options
    parser.add_argument('-solver',          type=str,     default="ADAM",                           choices=["SGD", "ADAM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,                              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,                              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,                            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                                help='weight decay')
    parser.add_argument('-batch_size',      type=int,     default=2,                                help='training batch size')
    parser.add_argument('-train_epoch_size',type=int,     default=200,                              help='train epoch size')
    parser.add_argument('-valid_epoch_size',type=int,     default=100,                              help='valid epoch size')
    parser.add_argument('-epoch_max',       type=int,     default=1000,                             help='max #epochs')


    ### learning rate options
    parser.add_argument('-lr_init',         type=float,   default=1e-5,                             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=10,                               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=1,                                help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.99,                             help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.001,                            help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    

    ### other options
    parser.add_argument('-loss',            type=str,     default="L1",                             help="Loss [Options: L1, L2]")
    parser.add_argument('-seed',            type=int,     default=9487,                             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=8,                                help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',                               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=1,                                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                                    help='use cpu?')
    
    opts = parser.parse_args()

    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m
    
    ### default model name
    if opts.model_name == 'none':
        opts.model_name = "%s_%s" %(opts.model, opts.ckp_name)

    if opts.suffix != "":
        opts.model_name += "_%s" %opts.suffix

    opts.size_multiplier = 2 ** 6
    
    print(opts)

    ### model saving directory
    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

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


    ### resume latest model
    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))
    epoch_st = 0

    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]


    if epoch_st > 0:

        print('=====================================================================')
        print('===> Resuming model from epoch %d' %epoch_st)
        print('=====================================================================')

        ### resume latest model and solver
        model, meta_opt = utils.load_model(model, meta_opt, opts, epoch_st)

    else:
        ### save epoch 0
        utils.save_model(model, meta_opt, opts, 0)


    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')


    ### initialize loss writer
    loss_dir = os.path.join(opts.model_dir, 'loss')
    loss_writer = SummaryWriter(loss_dir) 
    
    ### Load pretrained GFlowNet
    GFlowNet = GPWC().eval()
    for param in GFlowNet.parameters():
        param.requires_grad = False

    
    ### convert to GPU
    device = torch.device("cuda" if opts.cuda else "cpu")
    model = model.to(device)
    GFlowNet = GFlowNet.to(device)
    
    ### Define losses here and pass flownet and interpolator to losses
    L_in = inner_loop_loss_perc_affine_another_try(criterion, None, GFlowNet, summary_writer= loss_writer)
    L_out = outer_loop_loss_affine(GFlowNet, None, summary_writer= loss_writer)
    
    train_data_loader, test_data_loader = get_data_loaders(opts)

    print("Testing no adaptation...")
    _ = test_no_adaptation(test_data_loader, model, device, L_in, 0, opts)
    torch.cuda.empty_cache()
    for epoch in range(1, opts.epoch_max):
        train_data_loader, test_data_loader = get_data_loaders(opts)
        model, meta_opt = train(train_data_loader, model, device, meta_opt, L_in, L_out, epoch, opts)
        torch.cuda.empty_cache()
        print("Evaluating without adaptation...")
        _ = eval_no_adaptation(test_data_loader, model, device, L_in, epoch, opts)
        torch.cuda.empty_cache()
        model_for_adaptation = get_model(opts).to(device)
        model_for_adaptation.load_state_dict(model.state_dict())
        print("Testing with adaptation...")
        _ = test(test_data_loader, model_for_adaptation, device, L_in, epoch, opts)
        del model_for_adaptation
        utils.save_model(model, meta_opt, opts, epoch)
        torch.cuda.empty_cache()