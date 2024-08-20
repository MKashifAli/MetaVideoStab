import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
# end

Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow, cuda_flag):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        if cuda_flag:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
        else:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1)
    # end

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

##########################################################

class SpyNet(torch.nn.Module):
    def __init__(self, cuda_flag= True, weights_dir= "./networks/models_spynet/", model_name= "kitti-final", n_levels= 5):
        super(SpyNet, self).__init__()
        self.cuda_flag = cuda_flag
        self.weights_dir = weights_dir
        self.model_name = model_name
        self.n_levels = n_levels

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            # end

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)

        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(self.n_levels + 1)])

        self.load_state_dict(torch.load(self.weights_dir + 'network-' + self.model_name + '.pytorch'), strict=False)

    
    def forward(self, tenOne, tenTwo):
        assert(tenOne.shape[1] == tenTwo.shape[1])
        
        intWidth = tenOne.shape[3]
        intHeight = tenOne.shape[2]

        tenPreprocessedOne = tenOne
        tenPreprocessedTwo = tenTwo

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tensorFirst = [tenPreprocessedOne]
        tensorSecond = [tenPreprocessedTwo]

        for intLevel in range(self.n_levels):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))

        tensorFlow = tensorFirst[0].new_zeros(tensorFirst[0].size(0), 2,
                                              int(math.floor(tensorFirst[0].size(2) / 2.0)),
                                              int(math.floor(tensorFirst[0].size(3) / 2.0)))

        for intLevel in range(len(tensorFirst)):
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            # if the sizes of upsampling and downsampling are not the same, apply zero-padding.
            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 1, 0, 0], mode='replicate')

            # input ：[first picture of corresponding level,
            #          the output of w with input second picture of corresponding level and upsampling flow,
            #          upsampling flow]
            # then we obtain the final flow.
            x = tensorFirst[intLevel]
            y = Backward(tensorInput=tensorSecond[intLevel], tensorFlow=tensorUpsampled, cuda_flag=self.cuda_flag)
            z = tensorUpsampled
            tensorFlow = self.moduleBasic[intLevel](torch.cat([tensorFirst[intLevel],
                                                               Backward(tensorInput=tensorSecond[intLevel],
                                                                        tensorFlow=tensorUpsampled,
                                                                        cuda_flag=self.cuda_flag),
                                                               tensorUpsampled], 1)) + tensorUpsampled

        tenFlow = torch.nn.functional.interpolate(input= tensorFlow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow


##########################################################
if __name__ == '__main__':
    import flow_vis as fv
    import cv2
    arguments_strOne = './15.png'
    arguments_strTwo = './14.png'
    models = ['sintel-final', 'sintel-clean', 'chairs-final', 'chairs-clean', 'kitti-final']
    for arguments_strModel in models:
        if not os.path.exists("./comp_imgs_" + arguments_strModel + "/"):
            os.mkdir("./comp_imgs_" + arguments_strModel + "/")
        n_levels = 50
        print("Doing:", arguments_strModel)
        for int_level in range(n_levels):
            spynet = SpyNet(model_name= arguments_strModel, weights_dir= "./models_spynet/", n_levels= int_level).cuda().eval()
            
            tenOne = torch.FloatTensor(numpy.expand_dims(cv2.imread(arguments_strOne)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0), 0)).cuda()
            tenTwo = torch.FloatTensor(numpy.expand_dims(cv2.imread(arguments_strTwo)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0), 0)).cuda()

            #tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))).cuda()
            #tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))).cuda()
            with torch.no_grad():
                tenOutput = spynet(tenOne, tenTwo) #estimate(tenOne, tenTwo)
            tenOutput = tenOutput.cpu()
            #print(tenOutput.shape)
            np_flow = numpy.transpose(tenOutput.numpy()[0, :, :, :], (1, 2, 0))
            fc = fv.flow_to_color(np_flow, convert_to_bgr= True)
            cv2.imwrite("./comp_imgs_" + arguments_strModel + "/current_flow_" + arguments_strModel + "_" + str(int_level) + ".png", fc)
