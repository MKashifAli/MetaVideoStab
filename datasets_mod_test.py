### python lib
import os, sys, math, random, glob, cv2, argparse, natsort
import numpy as np

### torch lib
import torch
import torch.utils.data as data
import torchvision.transforms as tv
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

def read_img(filename, resize= False, scale_factor= 1.0):

    ## read image and convert to RGB in [0, 1]

    img = cv2.imread(filename)
    if resize:
        img = cv2.resize(img, (0,0), fx= scale_factor, fy= scale_factor) 
        #print(img.shape)

    if img is None:
        raise Exception("Image %s does not exist" %filename)

    #img = img[:, :, ::-1] ## BGR to RGB
    
    img = np.float32(img) / 255.0

    return img

crop_thingies = (0, 0, 0, 0)

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch - 0)
        self.w1 = random.randint(0, iw - self.cw - 0)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        global crop_thingies
        crop_thingies = (self.h1, self.h2, self.w1, self.w2)
        
    def __call__(self, img):
        #print("Came in to cropper...", crop_thingies)
        #sys.exit()
        if len(img.shape) == 3:
            return img[self.h1 : self.h2, self.w1 : self.w2, :]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]


class FixedCrop(object):
    def __init__(self, crop_size, mh, mw):
        self.ch, self.cw = crop_size
        self.h1 = mh
        self.w1 = mw
        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        #global crop_thingies
        #crop_thingies = (self.h1, self.h2, self.w1, self.w2)
        
    def __call__(self, img):
        #print("Came in to cropper...", crop_thingies)
        #sys.exit()
        if len(img.shape) == 3:
            return img[self.h1 : self.h2, self.w1 : self.w2, :]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]



class MultiFramesDataset(data.Dataset):

    def __init__(self, mode= 'train', opts= None, get_lbls= False, prev_iter_counter= 0):
        super(MultiFramesDataset, self).__init__()
        self.mode = mode
        self.ip_dir = opts.data_dir
        self.dataset_task_list = []
        self.num_frames = []
        self.get_lbls = get_lbls
        self.iter_counter = prev_iter_counter
        if self.mode == "train":
            self.sample_frames = opts.sample_frames
            self.geometry_aug = opts.geometry_aug
            self.scale_min = opts.scale_min
            self.scale_max = opts.scale_max
            self.crop_size = opts.crop_size
            self.order_aug = opts.order_aug
            self.size_multiplier = opts.size_multiplier

            self.dataset_task_list = natsort.natsorted(glob.glob(os.path.join(self.ip_dir + "*/")))
            self.dataset_lbls_list = natsort.natsorted(glob.glob(os.path.join(opts.train_lbl_dir + "*/")))
            self.shuffled_inds = np.arange(len(self.dataset_task_list))
            np.random.shuffle(self.shuffled_inds)
            #list_lbl = natsort.natsorted(glob.glob(os.path.join(self.lbl_dir + "*/")))
        elif self.mode == "val":
            self.prestab_dir = opts.pstb_dir
            self.dataset_task_list = natsort.natsorted(glob.glob(os.path.join(self.ip_dir + "*/")))
            self.vid_name_list = natsort.natsorted(os.listdir(self.ip_dir))
            self.dataset_task_list_pre_stable = natsort.natsorted(glob.glob(os.path.join(self.prestab_dir + "vids/*/")))
            self.dataset_transform_list_pre_stable = natsort.natsorted(glob.glob(os.path.join(self.prestab_dir + "txs/*.npy")))
            #self.dataset_task_list = [os.path.join(self.ip_dir+i+'/') for i in ['Crowd_7','Parallax_10','Regular_16','Running_16','Zooming_11']]
            self.num_frames = [len(os.listdir(dir)) for dir in self.dataset_task_list]
            # print(self.dataset_task_list)
        else:
            raise ValueError("Only train and val mode is implemented at the moment...")
        
        print("+"*20, "Total task_list:", "+"*20)
        video_names = []
        for task in self.dataset_task_list:
            video_names.append(task.split("/")[-2])
        print(video_names)
        print("+"*50)
        self.num_tasks = len(self.dataset_task_list)

        #for ip in self.dataset_task_list:
        #    self.num_frames.append(len(natsort.natsorted(os.listdir(ip))))
        global crop_thingies

        print("[%s] Total %d videos (%d frames)" %(self.__class__.__name__, len(self.dataset_task_list), sum(self.num_frames)))


    def __len__(self):
        return len(self.vid_name_list)


    def __getitem__(self, index):
        meta_data = {}
        meta_data['idx'] = index
        vid_name = self.vid_name_list[index]
        ## random select starting frame index t between [0, N - number_of_sample_frames] for "mode = "train" | "validate""
        if self.mode == "train":
            index_ = self.shuffled_inds[index]
            N = self.num_frames[index_]
            T = random.randint(0, N - self.sample_frames)
            #T = 0 #### Change this again for later experiments to randomize the frames for training
            meta_data['starting_frame'] = T
    
            input_dir = self.dataset_task_list[index_]
            if self.get_lbls:
                lbl_dir = self.dataset_lbls_list[index_]
            meta_data['unstable_video_path'] = input_dir
            if self.get_lbls:
                meta_data['stable_video_path'] = lbl_dir

            ## sample from T to T + #sample_frames - 1
            frame_ip = []
            if self.get_lbls:
                frame_lbl = []
            ip_frame_list = natsort.natsorted(glob.glob(os.path.join(input_dir, "*.*")))
            if self.get_lbls:
                lbl_frame_list = natsort.natsorted(glob.glob(os.path.join(lbl_dir, "*.*")))
            for t in range(T, T + self.sample_frames):
                if self.iter_counter < 200000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 1.0))
                    if self.get_lbls:
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 1.0))
                #elif self.iter_counter < 30000:
                #    frame_ip.append(read_img(ip_frame_list[t], True, 0.4))
                #    if self.get_lbls:
                #        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.4))
                #elif self.iter_counter < 40000:
                #    frame_ip.append(read_img(ip_frame_list[t], True, 0.45))
                #    if self.get_lbls:
                #        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.45))
                #elif self.iter_counter < 50000:
                #    frame_ip.append(read_img(ip_frame_list[t], True, 0.5))
                #    if self.get_lbls:
                #        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.5))
                #elif self.iter_counter < 60000:
                #    frame_ip.append(read_img(ip_frame_list[t], True, 0.55))
                #    if self.get_lbls:
                #        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.55))
                #elif self.iter_counter < 70000:
                #    frame_ip.append(read_img(ip_frame_list[t], True, 0.6))
                #    if self.get_lbls:
                #        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.6))
                #elif self.iter_counter < 80000:
                #    frame_ip.append(read_img(ip_frame_list[t], True, 0.7))
                #    if self.get_lbls:
                #        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.7))
                #elif self.iter_counter < 90000:
                #    frame_ip.append(read_img(ip_frame_list[t], True, 0.8))
                #    if self.get_lbls:
                #        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.8))
                #elif self.iter_counter < 100000:
                #    frame_ip.append(read_img(ip_frame_list[t], True, 0.9))
                #    if self.get_lbls:
                #        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.9))
                else:
                    frame_ip.append(read_img(ip_frame_list[t], False, 1.0))
                    if self.get_lbls:
                        frame_lbl.append(read_img(lbl_frame_list[t], False, 1.0))

            meta_data['ip_frame_paths'] = ip_frame_list[T:T + self.sample_frames]
            if self.get_lbls:
                meta_data['op_frame_paths'] = lbl_frame_list[T:T + self.sample_frames]

            self.iter_counter += 1

            ## data augmentation
            if self.geometry_aug:

                ## random scale
                H_in = frame_ip[0].shape[0]
                W_in = frame_ip[0].shape[1]

                sc = np.random.uniform(self.scale_min, self.scale_max)
                H_out = int(math.floor(H_in * sc))
                W_out = int(math.floor(W_in * sc))

                ## scaled size should be equal to opts.crop_size
                if H_out < W_out:
                    if H_out < self.crop_size:
                        H_out = self.crop_size
                        W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
                else: ## W_out < H_out
                    if W_out < self.crop_size:
                        W_out = self.crop_size
                        H_out = int(math.floor(H_in * float(W_out) / float(W_in)))

                for t in range(self.sample_frames):
                    frame_ip[t] = cv2.resize(frame_ip[t], (W_out, H_out))
                    if self.get_lbls:
                        frame_lbl[t] = cv2.resize(frame_lbl[t], (W_out, H_out))
                meta_data['scale_factor'] = sc

            ## random crop
            cropper = RandomCrop(frame_ip[0].shape[:2], (self.crop_size, self.crop_size))
            

            for t in range(self.sample_frames):
                frame_ip[t] = cropper(frame_ip[t])
                if self.get_lbls:
                    frame_lbl[t] = cropper(frame_lbl[t])
                if frame_ip[t].shape[1] != self.crop_size and frame_ip[t].shape[0] != self.crop_size:
                    print("[MultiFrameDataset]: size mismatch occured... =>", frame_ip[t].shape)
                if self.get_lbls:
                        if frame_lbl[t].shape[1] != self.crop_size and frame_lbl[t].shape[0] != self.crop_size:
                            print("[MultiFrameDataset]: size mismatch occured... =>", frame_lbl[t].shape)
            meta_data['crop_coords'] = crop_thingies
            

            if self.geometry_aug:
                #meta_data['rotation'] = False
                ### random rotate
                rotate = random.randint(0, 3)
                if rotate != 0:
                    for t in range(self.sample_frames):
                        frame_ip[t] = np.rot90(frame_ip[t], rotate)
                        if self.get_lbls:
                            frame_lbl[t] = np.rot90(frame_lbl[t], rotate)
                    #meta_data['rotation'] = True
                    
                ## horizontal flip
                if np.random.random() >= 0.5:
                    for t in range(self.sample_frames):
                        frame_ip[t] = cv2.flip(frame_ip[t], flipCode=0)
                        if self.get_lbls:
                            frame_lbl[t] = cv2.flip(frame_lbl[t], flipCode=0)
                    #meta_data['hflip'] = True


            if self.order_aug:
                ## reverse temporal order
                #meta_data['order'] = "normal"
                if np.random.random() >= 0.5:
                    #meta_data['order'] = "reversed"
                    frame_ip.reverse()
                    if self.get_lbls:
                        frame_lbl.reverse()

        elif self.mode == "val":
            #input_dir = self.dataset_task_list[index]
            #lbl_dir = self.dataset_task_list_pre_stable[index]
            #txs = np.load(self.dataset_transform_list_pre_stable[index])
            
            input_dir = self.ip_dir + vid_name + "/"
            lbl_dir = self.prestab_dir + "vids/" + vid_name + "/"
            txs = np.load(self.prestab_dir + "txs/" + vid_name + ".npy")


            meta_data['unstable_video_path'] = input_dir
            meta_data['video_name'] = vid_name
            meta_data['stable_video_path'] = lbl_dir
            meta_data['stable_txs_path'] = self.prestab_dir + "txs/" + vid_name + ".npy"

            ## sample from T to T + #sample_frames - 1
            frame_ip = []
            frame_lbl = []
            ip_frame_list = natsort.natsorted(glob.glob(os.path.join(input_dir, "*.*")))
            lbl_frame_list = natsort.natsorted(glob.glob(os.path.join(lbl_dir, "*.*")))
            for t in range(0, self.num_frames[index]):
                frame_ip.append(read_img(ip_frame_list[t]))
                frame_lbl.append(read_img(lbl_frame_list[t]))
            meta_data['ip_frame_paths'] = ip_frame_list

        ### convert (H, W, C) array to (C, H, W) tensor
        
        X = []
        Y = []

        if self.mode == "train":
            if self.get_lbls:
                for t in range(len(frame_ip)):
                    X.append(torch.from_numpy(frame_ip[t].transpose(2, 0, 1).astype(np.float32)))
                    if self.get_lbls:
                        Y.append(torch.from_numpy(frame_lbl[t].transpose(2, 0, 1).astype(np.float32)))
            
            if self.get_lbls:
                return {'X': X, 'Y': Y, 'meta_data': meta_data}
            else:
                return {'X': X, 'meta_data': meta_data}

        else:
            for t in range(len(frame_ip)):
                X.append(torch.unsqueeze(torch.from_numpy(frame_ip[t].transpose(2, 0, 1).astype(np.float32)), 0))
                Y.append(torch.unsqueeze(torch.from_numpy(frame_lbl[t].transpose(2, 0, 1).astype(np.float32)), 0))

            return {'X': X, 'Y': Y, 'A': txs, 'meta_data': meta_data}


class MultiFramesDatasetHybrid(data.Dataset):

    def __init__(self, mode= 'train', opts= None, get_lbls= True):
        super(MultiFramesDatasetHybrid, self).__init__()
        self.ip_dir = opts.data_dir
        self.ip_dir_synth = opts.data_dir_synth
        self.lbl_dir = opts.train_lbl_dir
        self.sample_frames = opts.sample_frames
        self.geometry_aug = opts.geometry_aug
        self.scale_min = opts.scale_min
        self.scale_max = opts.scale_max
        self.crop_size = opts.crop_size
        self.order_aug = opts.order_aug
        self.size_multiplier = opts.size_multiplier
        self.mode = mode
        self.dataset_task_list = []
        self.num_frames = []
        self.get_lbls = get_lbls
        self.iter_counter = 0
        if self.mode == "train":
            self.dataset_task_list = natsort.natsorted(glob.glob(os.path.join(self.ip_dir + "*/")))
            self.dataset_syn_ip_list = natsort.natsorted(glob.glob(os.path.join(self.ip_dir_synth + "*/")))
            self.dataset_syn_lbls_list = natsort.natsorted(glob.glob(os.path.join(self.lbl_dir + "*/")))
            self.shuffled_inds = np.arange(len(self.dataset_task_list))
            np.random.shuffle(self.shuffled_inds)
            #list_lbl = natsort.natsorted(glob.glob(os.path.join(self.lbl_dir + "*/")))
        elif self.mode == "val":
            self.dataset_task_list = natsort.natsorted(glob.glob(os.path.join(self.ip_dir + "*/")))
        else:
            raise ValueError("Only train and val mode is implemented at the moment...")
        
        print("+"*20, "Total task_list:", "+"*20)
        video_names = []
        for task in self.dataset_task_list:
            video_names.append(task.split("/")[-2])
        print(video_names)
        print("+"*50)
        self.num_tasks = len(self.dataset_task_list)

        for ip in self.dataset_task_list:
            self.num_frames.append(len(natsort.natsorted(os.listdir(ip))))

        for ip in self.synth_list:
            self.num_frames_synth.append(len(natsort.natsorted(os.listdir(ip))))

        global crop_thingies

        print("[%s] Total %d videos (%d frames)" %(self.__class__.__name__, len(self.dataset_task_list), sum(self.num_frames)))


    def __len__(self):
        return len(self.dataset_task_list)


    def __getitem__(self, index):
        meta_data = {}
        meta_data['idx'] = index
        ## random select starting frame index t between [0, N - number_of_sample_frames] for "mode = "train" | "validate""
        if self.mode == "train":
            index_ = self.shuffled_inds[index]
            rand_num = random.randint(0, len(self.synth_list))
            N = self.num_frames[index_]
            Ns = self.num_frames_synth[rand_num]
            T = random.randint(0, N - self.sample_frames)
            Ts = random.randint(0, Ns - self.sample_frames)
            #T = 0 #### Change this again for later experiments to randomize the frames for training
            meta_data['starting_frame'] = T
    
            input_dir = self.dataset_task_list[index_]
            if self.get_lbls:
                syn_ip_dir = self.dataset_syn_ip_list[rand_num]
                lbl_dir = self.dataset_syn_lbls_list[rand_num]
            meta_data['unstable_video_path'] = input_dir
            if self.get_lbls:
                meta_data['stable_video_path'] = lbl_dir

            ## sample from T to T + #sample_frames - 1
            frame_ip = []
            if self.get_lbls:
                frame_syn = []
                frame_lbl = []
            ip_frame_list = natsort.natsorted(glob.glob(os.path.join(input_dir, "*.*")))
            if self.get_lbls:
                syn_ip_list = natsort.natsorted(glob.glob(os.path.join(syn_ip_dir, "*.*")))
                lbl_frame_list = natsort.natsorted(glob.glob(os.path.join(lbl_dir, "*.*")))
            
            for t in range(T, T + self.sample_frames):
                if self.iter_counter < 1000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 0.3))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 0.3))
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.3))
                elif self.iter_counter < 2000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 0.4))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 0.4))
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.4))
                elif self.iter_counter < 3000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 0.45))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 0.45))
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.45))
                elif self.iter_counter < 4000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 0.5))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 0.5))
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.5))
                elif self.iter_counter < 5000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 0.55))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 0.55))
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.55))
                elif self.iter_counter < 6000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 0.6))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 0.6))
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.6))
                elif self.iter_counter < 7000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 0.7))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 0.7))
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.7))
                elif self.iter_counter < 8000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 0.8))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 0.8))
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.8))
                elif self.iter_counter < 9000:
                    frame_ip.append(read_img(ip_frame_list[t], True, 0.9))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 0.9))
                        frame_lbl.append(read_img(lbl_frame_list[t], True, 0.9))
                else:
                    frame_ip.append(read_img(ip_frame_list[t], False, 1.0))
                    if self.get_lbls:
                        frame_syn.append(read_img(syn_ip_list[t], True, 1.0))
                        frame_lbl.append(read_img(lbl_frame_list[t], False, 1.0))

            meta_data['ip_frame_paths'] = ip_frame_list[T:T + self.sample_frames]
            if self.get_lbls:
                meta_data['op_frame_paths'] = lbl_frame_list[T:T + self.sample_frames]

            self.iter_counter += 1

            ## data augmentation
            if self.geometry_aug:

                ## random scale
                H_in = frame_ip[0].shape[0]
                W_in = frame_ip[0].shape[1]

                sc = np.random.uniform(self.scale_min, self.scale_max)
                H_out = int(math.floor(H_in * sc))
                W_out = int(math.floor(W_in * sc))

                ## scaled size should be equal to opts.crop_size
                if H_out < W_out:
                    if H_out < self.crop_size:
                        H_out = self.crop_size
                        W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
                else: ## W_out < H_out
                    if W_out < self.crop_size:
                        W_out = self.crop_size
                        H_out = int(math.floor(H_in * float(W_out) / float(W_in)))

                for t in range(self.sample_frames):
                    frame_ip[t] = cv2.resize(frame_ip[t], (W_out, H_out))
                    if self.get_lbls:
                        frame_syn[t] = cv2.resize(frame_syn[t], (W_out, H_out))
                        frame_lbl[t] = cv2.resize(frame_lbl[t], (W_out, H_out))

                meta_data['scale_factor'] = sc

            ## random crop
            cropper = RandomCrop(frame_ip[0].shape[:2], (self.crop_size, self.crop_size))
            

            for t in range(self.sample_frames):
                frame_ip[t] = cropper(frame_ip[t])
                if self.get_lbls:
                    frame_lbl[t] = cropper(frame_lbl[t])
                    frame_syn[t] = cropper(frame_syn[t])
                if frame_ip[t].shape[1] != self.crop_size and frame_ip[t].shape[0] != self.crop_size:
                    print("[MultiFrameDataset]: size mismatch occured... =>", frame_ip[t].shape)
                if self.get_lbls:
                        if frame_lbl[t].shape[1] != self.crop_size and frame_lbl[t].shape[0] != self.crop_size:
                            print("[MultiFrameDataset]: size mismatch occured... =>", frame_lbl[t].shape)
                        if frame_syn[t].shape[1] != self.crop_size and frame_syn[t].shape[0] != self.crop_size:
                            print("[MultiFrameDataset]: size mismatch occured... =>", frame_syn[t].shape)
            meta_data['crop_coords'] = crop_thingies
            

            if self.geometry_aug:
                #meta_data['rotation'] = False
                ### random rotate
                rotate = random.randint(0, 3)
                if rotate != 0:
                    for t in range(self.sample_frames):
                        frame_ip[t] = np.rot90(frame_ip[t], rotate)
                        if self.get_lbls:
                            frame_syn[t] = np.rot90(frame_syn[t], rotate)
                            frame_lbl[t] = np.rot90(frame_lbl[t], rotate)
                    #meta_data['rotation'] = True
                    
                ## horizontal flip
                if np.random.random() >= 0.5:
                    for t in range(self.sample_frames):
                        frame_ip[t] = cv2.flip(frame_ip[t], flipCode=0)
                        if self.get_lbls:
                            frame_syn[t] = cv2.flip(frame_syn[t], flipCode=0)
                            frame_lbl[t] = cv2.flip(frame_lbl[t], flipCode=0)
                    #meta_data['hflip'] = True


            if self.order_aug:
                ## reverse temporal order
                #meta_data['order'] = "normal"
                if np.random.random() >= 0.5:
                    #meta_data['order'] = "reversed"
                    frame_ip.reverse()
                    if self.get_lbls:
                        frame_syn.reverse()
                        frame_lbl.reverse()

        elif self.mode == "val":
            input_dir = self.dataset_task_list[index]
            #lbl_dir = self.dataset_task_list[index][1]
            meta_data['unstable_video_path'] = input_dir
            meta_data['video_name'] = input_dir
            #meta_data['stable_video_path'] = lbl_dir

            ## sample from T to T + #sample_frames - 1
            frame_ip = []
            #frame_lbl = []
            ip_frame_list = natsort.natsorted(glob.glob(os.path.join(input_dir, "*.*")))
            #lbl_frame_list = natsort.natsorted(glob.glob(os.path.join(lbl_dir, "*.*")))
            for t in range(0, self.sample_frames):
                frame_ip.append(read_img(ip_frame_list[t]))
                #frame_lbl.append(read_img(lbl_frame_list[t]))
            meta_data['ip_frame_paths'] = ip_frame_list

        ### convert (H, W, C) array to (C, H, W) tensor
        X = []
        if self.mode == "train":
            if self.get_lbls:
                X2 = []
                Y = []
            for t in range(len(frame_ip)):
                X.append(torch.from_numpy(frame_ip[t].transpose(2, 0, 1).astype(np.float32)))
                if self.get_lbls:
                    X2.append(torch.from_numpy(frame_syn[t].transpose(2, 0, 1).astype(np.float32)))
                    Y.append(torch.from_numpy(frame_lbl[t].transpose(2, 0, 1).astype(np.float32)))

            
            if self.get_lbls:
                return {'X': X, 'X2': X2, 'Y': Y, 'meta_data': meta_data}
            else:
                return {'X': X, 'meta_data': meta_data}

        else:
            for t in range(len(frame_ip)):
                X.append(torch.unsqueeze(torch.from_numpy(frame_ip[t].transpose(2, 0, 1).astype(np.float32)), 0))

            return {'X': X, 'meta_data': meta_data}

class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def create_data_loader(data_set, mode, train_epoch_size= 1000, batch_size= 4, threads= 8):

    ### generate random index
    if mode == 'train':
        total_samples = train_epoch_size * batch_size
    else:
        raise ValueError("Only train mode is implemented at the moment...")

    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    ### generate data sampler and loader
    sampler = SubsetSequentialSampler(indices)
    if mode == "train":
        data_loader = DataLoader(dataset= data_set, num_workers= threads, batch_size= batch_size, sampler= sampler, pin_memory= True)

    return data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta Stabilization")

    ### model options
    parser.add_argument('-model',           type=str,     default="ours",                           help='Model to use')
    parser.add_argument('-ckp_name',        type=str,     default="best_stab_vf_v1.h5",             help='TransformNet') 
    parser.add_argument('-model_name',      type=str,     default='meta_stab_w_homo',               help='path to save model')

    ### dataset options
    parser.add_argument('-train_data_dir',  type=str,     default='data/',                          help='path to train data folder')
    parser.add_argument('-data_dir',        type=str,     default='data/',                          help='path to test data folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',                    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=192,                              help='patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=0,                                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.5,                              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,                              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=70,                               help='#frames for training')

    ### training options
    parser.add_argument('-solver',          type=str,     default="ADAM",                           choices=["SGD", "ADAM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,                              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,                              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,                            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                                help='weight decay')
    parser.add_argument('-batch_size',      type=int,     default=2,                                help='training batch size')
    parser.add_argument('-train_epoch_size',type=int,     default=20,                               help='train epoch size')
    parser.add_argument('-valid_epoch_size',type=int,     default=100,                              help='valid epoch size')
    parser.add_argument('-epoch_max',       type=int,     default=100,                              help='max #epochs')


    ### learning rate options
    parser.add_argument('-lr_init',         type=float,   default=5e-5,                             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,                               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=20,                               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,                              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.1,                              help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    

    ### other options
    parser.add_argument('-loss',            type=str,     default="L1",                             help="Loss [Options: L1, L2]")
    parser.add_argument('-seed',            type=int,     default=9487,                             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=8,                                help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',                               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=1,                                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                                    help='use cpu?')
    
    opts = parser.parse_args()
    opts.size_multiplier = 2 ** 6 ## Inputs to FlowNet need to be divided by 64

    train_dataset = MultiFramesDataset(mode= "train", opts= opts)
    data_loader = create_data_loader(train_dataset, mode= "train", batch_size= 1, threads= 0)
    sane_path = "./sanitycheck/"

    for iteration, (data) in enumerate(data_loader, 1):
        x = data['X']
        if iteration > 3:
            break
        if iteration == 1:
            if not os.path.exists(sane_path):
                os.mkdir(sane_path)
            i = 0
            for xs in x:
                img = xs[0]
                img = np.transpose(img.numpy(), (1, 2, 0))
                cv2.imwrite(sane_path + "X-" + str(iteration) + "-" + str(i) + ".png", img * 255)
                i += 1
        else:
            print(x[0].shape)

        