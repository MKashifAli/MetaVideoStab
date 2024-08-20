import torch
import torch.nn as nn
#from networks.raft_module import get_raft_module, estimate_flow
from networks.spynet_module import SpyNet
import kornia

def rotate_and_translate(ten, r, t):
    rotated = kornia.geometry.transform.rotate(ten, r)
    translated = kornia.geometry.transform.translate(rotated, t)
    return translated

class CoarseStabilizer(nn.Module):
    def __init__(self, in_channels=2):
        super(CoarseStabilizer, self).__init__()
        self.E0 = nn.Sequential(
                        nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1), 
                        nn.LeakyReLU(0.2))
        self.E1 = nn.Sequential(
                        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), 
                        nn.LeakyReLU(0.2))
        self.E2 = nn.Sequential(
                        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), 
                        nn.LeakyReLU(0.2))
        
        self.ups = torch.nn.functional.interpolate # usage: ups(ip, scale_factor= 2)

        self.D0 = nn.Sequential(
                        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), 
                        nn.LeakyReLU(0.2))
        self.D1 = nn.Sequential(
                        nn.Conv2d(16 + 16, 16, kernel_size=3, stride=1, padding=1), 
                        nn.LeakyReLU(0.2))
        
        self.D2 = nn.Sequential(
                        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), 
                        nn.LeakyReLU(0.2))

        self.flat = nn.Flatten()

        self.L0 = nn.Linear(65536, 1000)
        self.L1 = nn.Linear(1000, 100)
        self.L2 = nn.Linear(100, 3)


        

    def forward(self, x):
        out = self.E0(x)
        E1 = self.E1(out)
        out = self.E2(E1)
        out = self.ups(out, scale_factor= 2)
        out = self.D0(out)
        out = torch.cat([E1, out], 1)
        out = self.D1(out)
        out = self.ups(out, scale_factor= 2)
        out = self.D2(out)
        out = self.flat(out)
        out = self.L0(out)
        out = self.L1(out)
        out = self.L2(out)
        return out


class CoarseStabilizerInferReady(nn.Module):
    def __init__(self, flownet, path= "./pretrained_models/CoarseStabilizerLatest.pth"):
        super(CoarseStabilizerInferReady, self).__init__()
        self.model = CoarseStabilizer().cuda()
        self.model.load_state_dict(torch.load(path)["model_state_dict"])
        self.model.eval()
        for params in self.model.parameters():
            params.requires_grad = False
        self.flownet = flownet
        #self.flownet = self.flownet.cuda()
        self.warp_fn = rotate_and_translate
    
    def lerp_angles(self, final_pt, n= 3):
        initial_pt = torch.zeros_like(final_pt)
        dx = (final_pt - initial_pt)/(n + 1)
        interim_pts = []
        interim_pts.append(initial_pt)
        for i in range(1, n + 2):
            interim_pts.append(interim_pts[i - 1] + dx)

        return interim_pts

    def lerp_translations(self, final_pt, n= 3):
        initial_pt = torch.zeros_like(final_pt)
        dx = (final_pt[:, 0] - initial_pt[:, 0])/(n + 1)
        dy = (final_pt[:, 1] - initial_pt[:, 1])/(n + 1)
        interim_pts = []
        interim_pts.append(initial_pt)
        for i in range(1, n + 2):
            x_ = interim_pts[i - 1][:, 0] + dx
            y_ = interim_pts[i - 1][:, 1] + dy
            temp = torch.stack([x_, y_], 1)
            interim_pts.append(temp)

        return interim_pts


    def intermediate_est(self, angles, translation, n_pts):
        angles_all = self.lerp_angles(angles, n_pts)
        translations_all = self.lerp_translations(translation, n_pts)
        return angles_all, translations_all


    def forward(self, x0, x1, intermediary= False, n_pts= 3):
        b, c, h, w = x0.shape
        denorm_factor_h = h/64
        denorm_factor_w = w/64
        f01 = self.flownet(x0.clone().detach(), x1.clone().detach())
        f01 = nn.functional.interpolate(f01, (64, 64))
        op = self.model(f01)
        angles = op[:, 0]*360
        w_t = op[:, 1]*denorm_factor_w
        h_t = op[:, 2]*denorm_factor_h
        translation = torch.stack([w_t, h_t], 1)
        if intermediary:
            angles_all, translations_all = self.intermediate_est(angles, translation, n_pts)
            #print(len(translations_all), translations_all[0].shape)
            op_interim = []
            for i in range(len(angles_all)):
                op_interim.append(torch.stack([angles_all[i], translations_all[i][:, 0], translations_all[i][:, 1]], 1))

        t_x1 = self.warp_fn(x1, angles, translation)
        op_ = torch.stack([angles, w_t, h_t], 1)
        if intermediary:
            return t_x1, op_, op_interim
        else:
            return t_x1, op_
 
def rescale_half(ten):
    img = ten2img(ten)
    #img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, (0, 0), fx= 0.5, fy= 0.5)
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)/255.0
    return torch.tensor(img.astype(np.float32))

def rescale_back(ten):
    img = ten2img(ten)
    #img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, (0, 0), fx= 2.0, fy= 2.0)
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)/255.0
    return torch.tensor(img.astype(np.float32))

def stabilize(model, loi):
    with torch.no_grad():
        lot = []
        lot.append(ten2img(loi[0])) # first frame remains the same...
        pbar = tqdm(total=len(loi))
        for i in range(1, len(loi)):
            i0 = rescale_half(loi[0]).cuda()
            it = rescale_half(loi[i]).cuda()
            tx, _ = model(i0, it)
            #print(tx.shape)
            it_ = rescale_back(tx)
            lot.append(ten2img(it_))
            pbar.update(1)
        pbar.close()

    return lot


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    import cv2
    import numpy as np
    import natsort
    from networks.GPWC_module.get_global_pwc_module import get_global_pwc as GPWC
    from tqdm import tqdm
    #from PWC_Module.PWC_Module import FlowNet_PWC as Flownet

    def read_img_as_tensor(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (640, 320))
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)/255.0
        img_t = torch.tensor(img.astype(np.float32))
        return img_t

    def ten2img(ten):
        img_t = ten[0]*255
        img_t = img_t.cpu().detach().numpy()
        img = np.transpose(img_t, (1, 2, 0))
        return img
    
    # FlowNet = Flownet().cuda().eval() #### <------ Loads PWC with weights...
    # for param in FlowNet.parameters():
    #     param.requires_grad = False

    GFlowNet = GPWC().eval()
    for param in GFlowNet.parameters():
        param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoarseStabilizerInferReady(GFlowNet, path= "./pretrained_models/GPWC_Stabilizer.pth").to(device)

    vid_name = "Crowd_21"
    folder_to_test = "../ds_bm/frames2/PX/" + vid_name + "/"
    save_dir = "../ds_bm/frames2/NF/" + vid_name + "_gpwc/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok= True)

    lof = natsort.natsorted(os.listdir(folder_to_test))
    loi = []
    
    for f in lof:
        loi.append(read_img_as_tensor(folder_to_test + f).to(device))

    print("Stabilizing...")
    tx_imgs = stabilize(model, loi)

    print("Saving...")
    for i in range(len(tx_imgs)):
        cv2.imwrite(save_dir + str(i) + ".png", tx_imgs[i])

    
    #tx, op_, op_interim = model(img0, img1, True)