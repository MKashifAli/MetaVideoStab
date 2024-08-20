import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math
import pdb

import getopt
import numpy
import os
import PIL
import PIL.Image
import sys
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0

#torch.set_grad_enabled(True) # make sure to not compute gradients for computational performance

#torch.cuda.device(0) # change this if you have a multiple graphics cards and you want to utilize them

#torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################
def custom_grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

##########################################################

arguments_strModel = 'sintel'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './output/out.flo'
arguments_strOutWarp = './output/result.png'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--model':
        arguments_strModel = strArgument # which model to use, see below

    elif strOption == '--first':
        arguments_strFirst = strArgument # path to the first frame

    elif strOption == '--second':
        arguments_strSecond = strArgument # path to the second frame

    elif strOption == '--out':
        arguments_strOut = strArgument # path to where the output should be stored

    elif strOption == '--outwarp':
        arguments_strOutWarp = strArgument # path to where the output should be stored

    # end
# end


import cupy
import re

kernel_Correlation_rearrange = '''
	extern "C" __global__ void kernel_Correlation_rearrange(
		const int n,
		const float* input,
		float* output
	) {
	  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	  if (intIndex >= n) {
	    return;
	  }

	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;

	  float dblValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

	  __syncthreads();

	  int intPaddedY = (intIndex / SIZE_3(input)) + 4;
	  int intPaddedX = (intIndex % SIZE_3(input)) + 4;
	  int intRearrange = ((SIZE_3(input) + 8) * intPaddedY) + intPaddedX;

	  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = dblValue;
	}
'''

kernel_Correlation_updateOutput = '''
	extern "C" __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const float* rbot0,
	  const float* rbot1,
	  float* top
	) {
	  extern __shared__ char patch_data_char[];
	  
	  float *patch_data = (float *)patch_data_char;
	  
	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = blockIdx.x + 4;
	  int y1 = blockIdx.y + 4;
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;
	  
	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = ((j * 1) + i) * SIZE_3(rbot0);
	      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }
	  
	  __syncthreads();
	  
	  __shared__ float sum[32];
	  
	  // Compute correlation
	  for(int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
	    sum[ch_off] = 0;
	  
	    int s2o = (top_channel % 9) - 4;
	    int s2p = (top_channel / 9) - 4;
	    
	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = ((j * 1) + i) * SIZE_3(rbot0);
	        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;
	          
	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;
	          
	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }
	    
	    __syncthreads();
	    
	    if (ch_off == 0) {
	      float total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = SIZE_3(rbot0);
	      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
	      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
	    }
	  } 
	}
'''

def cupy_kernel(strFunction, objectVariables):
	strKernel = globals()[strFunction]

	while True:
		objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArg = int(objectMatch.group(2))

		strTensor = objectMatch.group(4)
		intSizes = objectVariables[strTensor].size()

		strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
	# end

	while True:
		objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupy._util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class FunctionCorrelation(torch.autograd.Function):
	def __init__(self):
		super(FunctionCorrelation, self).__init__()
	# end

	def forward(self, first, second):
		self.save_for_backward(first, second)

		assert(first.is_contiguous() == True)
		assert(second.is_contiguous() == True)

		self.rbot0 = first.new(first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)).zero_()
		self.rbot1 = first.new(first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)).zero_()

		output = first.new(first.size(0), 81, first.size(2), first.size(3)).zero_()

		if first.is_cuda == True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream
			# end

			n = first.size(2) * first.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'input': first,
				'output': self.rbot0
			}))(
				grid=tuple([ int((n + 16 - 1) / 16), first.size(1), first.size(0) ]),
				block=tuple([ 16, 1, 1 ]),
				args=[ n, first.data_ptr(), self.rbot0.data_ptr() ],
				stream=Stream
			)

			n = second.size(2) * second.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'input': second,
				'output': self.rbot1
			}))(
				grid=tuple([ int((n + 16 - 1) / 16), second.size(1), second.size(0) ]),
				block=tuple([ 16, 1, 1 ]),
				args=[ n, second.data_ptr(), self.rbot1.data_ptr() ],
				stream=Stream
			)

			n = output.size(1) * output.size(2) * output.size(3)
			cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
				'rbot0': self.rbot0,
				'rbot1': self.rbot1,
				'top': output
			}))(
				grid=tuple([ first.size(3), first.size(2), first.size(0) ]),
				block=tuple([ 32, 1, 1 ]),
				shared_mem=first.size(1) * 4,
				args=[ n, self.rbot0.data_ptr(), self.rbot1.data_ptr(), output.data_ptr() ],
				stream=Stream
			)

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	def backward(self, gradOutput):
		first, second = self.saved_tensors

		assert(gradOutput.is_contiguous() == True)

		gradFirst = first.new(first.size()).zero_() if self.needs_input_grad[0] == True else None
		gradSecond = first.new(first.size()).zero_() if self.needs_input_grad[1] == True else None

		if first.is_cuda == True:
			raise NotImplementedError()

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return gradFirst, gradSecond
	# end
# end

class ModuleCorrelation(torch.nn.Module):
	def __init__(self):
		super(ModuleCorrelation, self).__init__()
	# end

	def forward(self, tensorFirst, tensorSecond):
		return FunctionCorrelation.apply(tensorFirst, tensorSecond)
	# end
# end

##########################################################

# import torch
# from torch.nn.modules.module import Module
# from torch.autograd import Function
# import correlation_cuda

# class FunctionCorrelation(Function):

#     @staticmethod
#     def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
#         ctx.save_for_backward(input1, input2)

#         ctx.pad_size = pad_size
#         ctx.kernel_size = kernel_size
#         ctx.max_displacement = max_displacement
#         ctx.stride1 = stride1
#         ctx.stride2 = stride2
#         ctx.corr_multiply = corr_multiply

#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()
#             output = input1.new()

#             correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
#                 ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input1, input2 = ctx.saved_tensors

#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()

#             grad_input1 = input1.new()
#             grad_input2 = input2.new()

#             correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
#                 ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

#         return grad_input1, grad_input2 #, None, None, None, None, None, None


# class ModuleCorrelation(Module):
#     def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
#         super(ModuleCorrelation, self).__init__()
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.max_displacement = max_displacement
#         self.stride1 = stride1
#         self.stride2 = stride2
#         self.corr_multiply = corr_multiply

#     def forward(self, input1, input2):

#         result = FunctionCorrelation.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)

#         return result

##############################################


class PwcNet(torch.nn.Module):
    def __init__(self, strModel='sintel'):
        super(PwcNet, self).__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            # end

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]
            # end
        # end

        class Backward(torch.nn.Module):
            def __init__(self):
                super(Backward, self).__init__()
            # end

            def forward(self, tensorInput, tensorFlow):
                if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
                    self.tensorPartial = torch.FloatTensor().resize_(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)).fill_(1.0).cuda()
                # end

                if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
                    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

                    self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
                # end

                tensorInput = torch.cat([ tensorInput, self.tensorPartial ], 1)
                tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

                #tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
                tensorOutput = custom_grid_sample(tensorInput, (self.tensorGrid + tensorFlow).permute(0, 2, 3, 1))

                tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

                return tensorOutput[:, :-1, :, :] * tensorMask
            # end
        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
                intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

                if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)

                if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]
                if intLevel < 6: self.moduleBackward = Backward()

                self.moduleCorrelation = ModuleCorrelation()
                self.moduleCorreleaky = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )
            # end

            def forward(self, tensorFirst, tensorSecond, objectPrevious):
                tensorFlow = None
                tensorFeat = None

                if objectPrevious is None:
                    tensorFlow = None
                    tensorFeat = None

                    tensorVolume = self.moduleCorreleaky(self.moduleCorrelation.forward(tensorFirst, tensorSecond))

                    tensorFeat = torch.cat([ tensorVolume ], 1)

                elif objectPrevious is not None:
                    tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
                    tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

                    tensorVolume = self.moduleCorreleaky(self.moduleCorrelation(tensorFirst, self.moduleBackward(tensorSecond, tensorFlow * self.dblBackward)))

                    tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow, tensorFeat ], 1)

                # end

                tensorFeat = torch.cat([ self.moduleOne(tensorFeat), tensorFeat ], 1)
                tensorFeat = torch.cat([ self.moduleTwo(tensorFeat), tensorFeat ], 1)
                tensorFeat = torch.cat([ self.moduleThr(tensorFeat), tensorFeat ], 1)
                tensorFeat = torch.cat([ self.moduleFou(tensorFeat), tensorFeat ], 1)
                tensorFeat = torch.cat([ self.moduleFiv(tensorFeat), tensorFeat ], 1)

                tensorFlow = self.moduleSix(tensorFeat)

                return {
                    'tensorFlow': tensorFlow,
                    'tensorFeat': tensorFeat
                }
            # end
        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1,  dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2,  dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,  dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8,  dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16,  dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1,  dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1,  dilation=1)
                )
            # end

            def forward(self, tensorInput):
                return self.moduleMain(tensorInput)
            # end
        # end

        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

        #self.load_state_dict(torch.load('./trained_models/' + strModel + '.pytorch'))
    # end

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])
    # end
# end

moduleNetwork = PwcNet().cuda()

##########################################################

def estimate(tensorInputFirst, tensorInputSecond):
    tensorOutput = torch.FloatTensor()

    assert(tensorInputFirst.size(1) == tensorInputSecond.size(1))
    assert(tensorInputFirst.size(2) == tensorInputSecond.size(2))

    intWidth = tensorInputFirst.size(2)
    intHeight = tensorInputFirst.size(1)

    assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    if True:
        tensorInputFirst = tensorInputFirst.cuda()
        tensorInputSecond = tensorInputSecond.cuda()
        tensorOutput = tensorOutput.cuda()
    # end

    if True:
        tensorPreprocessedFirst = tensorInputFirst.view(1, 3, intHeight, intWidth)
        tensorPreprocessedSecond = tensorInputSecond.view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0)) # Due to Pyramid method?
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        #pdb.set_trace()
        tensorFlow = 20.0 * torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        tensorOutput.resize_(2, intHeight, intWidth).copy_(tensorFlow[0, :, :, :])
    # end

    if True:
        tensorInputFirst = tensorInputFirst.cpu()
        tensorInputSecond = tensorInputSecond.cpu()
        tensorOutput = tensorOutput.cpu()
    # end

    return tensorOutput
# end

##########################################################

# if __name__ == '__main__':
#     tensorInputFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) / 255.0)
#     tensorInputSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) / 255.0)

#     tensorOutput = estimate(tensorInputFirst, tensorInputSecond)
#     #pdb.set_trace()
#     # Output *.flo file
#     objectOutput = open(arguments_strOut, 'wb')

#     numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
#     numpy.array([ tensorOutput.size(2), tensorOutput.size(1) ], numpy.int32).tofile(objectOutput)
#     numpy.array(tensorOutput.permute(1, 2, 0), numpy.float32).tofile(objectOutput)

#     objectOutput.close()

#     # Visualize *.flo file
#     #flowlib.show_flow(arguments_strOut)
#     #pdb.set_trace()

#     # Output warped *.png file
#     tensorInputFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst)).transpose(2, 0, 1).astype(numpy.float32) / 255.0)
#     tensorInputSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond)).transpose(2, 0, 1).astype(numpy.float32) / 255.0)
#     tensorOutput = tensorOutput.permute(1, 2, 0)[None,:,:,:]
#     #pdb.set_trace()
#     tensorOutput[:,:,:,0] = tensorOutput[:,:,:,0]/(1024/2)
#     tensorOutput[:,:,:,1] = tensorOutput[:,:,:,1]/(436/2)


#     scale = 0.5
#     mesh = numpy.array(numpy.meshgrid(numpy.linspace(-1,1,1024), numpy.linspace(-1,1,436)))
#     mesh = torch.FloatTensor(mesh)
#     outwarp = F.grid_sample(tensorInputSecond[None,:,:,:], mesh.permute(1, 2, 0)[None,:,:,:] + tensorOutput*scale)
#     outwarp = outwarp.squeeze().permute(1, 2, 0)

#     plt.imshow(outwarp.numpy())
#     plt.show()
# end

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
                super(Encoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.ReLU()
                )
                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class Decoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
                super(Decoder, self).__init__()
                
                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
                )

                if tanh:
                    self.activ = nn.Tanh()
                else:
                    self.activ = nn.ReLU()
                
                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                s = self.seq(x)
                s = self.activ(s)
                return s * self.GateConv(x)


        self.enc0 = Encoder(16, 32, stride=1)
        self.enc1 = Encoder(32, 32, stride=2)
        self.enc2 = Encoder(32, 32, stride=2)
        self.enc3 = Encoder(32, 32, stride=2)

        self.dec0 = Decoder(32, 32, stride=1)
        # up-scaling + concat
        self.dec1 = Decoder(32+32, 32, stride=1)
        self.dec2 = Decoder(32+32, 32, stride=1)
        self.dec3 = Decoder(32+32, 32, stride=1)

        self.dec4 = Decoder(32, 3, stride=1, tanh=True)

    def forward(self, w1, w2, flo1, flo2, fr1, fr2):
        s0 = self.enc0(torch.cat([w1, w2, flo1, flo1, fr1, fr2],1).cuda())
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        s4 = self.dec0(s3)
        # up-scaling + concat
        s4 = F.interpolate(s4, scale_factor=2, mode='nearest')
        s5 = self.dec1(torch.cat([s4, s2],1).cuda())
        s5 = F.interpolate(s5, scale_factor=2, mode='nearest')
        s6 = self.dec2(torch.cat([s5, s1],1).cuda())
        s6 = F.interpolate(s6, scale_factor=2, mode='nearest')
        s7 = self.dec3(torch.cat([s6, s0],1).cuda())

        out = self.dec4(s7)
        return out


class DIFNet2(nn.Module):
    def __init__(self):
        super(DIFNet2, self).__init__()

        class Backward(torch.nn.Module):
            def __init__(self):
                super(Backward, self).__init__()
            # end

            def forward(self, tensorInput, tensorFlow, scale=1.0):
                if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
                    self.tensorPartial = torch.FloatTensor().resize_(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)).fill_(1.0).cuda()
                # end

                if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
                    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

                    self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
                # end
                #pdb.set_trace()
                tensorInput = torch.cat([ tensorInput, self.tensorPartial ], 1)
                tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

                #tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow*scale).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
                tensorOutput = custom_grid_sample(tensorInput, (self.tensorGrid + tensorFlow*scale).permute(0, 2, 3, 1))

                tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

                return tensorOutput[:, :-1, :, :] * tensorMask

        # PWC
        self.pwc = PwcNet()
        # self.pwc.load_state_dict(torch.load('./trained_models/sintel.pytorch'))
        self.pwc.eval()

        # Warping layer
        self.warpLayer = Backward()
        self.warpLayer.eval()

        # UNets
        self.UNet2 = UNet2()
        self.ResNet2 = ResNet2()

    def warpFrame(self, fr_1, fr_2, scale=1.0):
        with torch.no_grad():
            temp_w = int(math.floor(math.ceil(fr_1.size(3) / 64.0) * 64.0)) # Due to Pyramid method?
            temp_h = int(math.floor(math.ceil(fr_1.size(2) / 64.0) * 64.0))

            temp_fr_1 = torch.nn.functional.interpolate(input=fr_1, size=(temp_h, temp_w), mode='nearest')
            temp_fr_2 = torch.nn.functional.interpolate(input=fr_2, size=(temp_h, temp_w), mode='nearest')

            flo = 20.0 * torch.nn.functional.interpolate(input=self.pwc(temp_fr_1, temp_fr_2), size=(fr_1.size(2), fr_1.size(3)), mode='bilinear', align_corners=False)
            return self.warpLayer(fr_2, flo, scale), flo

    def forward(self, fr1, fr2, f3, fs2, fs1, scale):
        w1, flo1 = self.warpFrame(fs2, fr1, scale=scale)
        w2, flo2 = self.warpFrame(fs1, fr2, scale=scale)

        I_int = self.UNet2(w1, w2, flo1, flo2, fr1, fr2)
        f_int, flo_int = self.warpFrame(I_int, f3)

        fhat = self.ResNet2(I_int, f_int, flo_int, f3)
        return fhat# , I_int



class DIFNet_ours(nn.Module):
    def __init__(self):
        super(DIFNet_ours, self).__init__()

        class Backward(torch.nn.Module):
            def __init__(self):
                super(Backward, self).__init__()
            # end

            def forward(self, tensorInput, tensorFlow, scale=1.0):
                if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
                    self.tensorPartial = torch.FloatTensor().resize_(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)).fill_(1.0).cuda()
                # end

                if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
                    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

                    self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
                # end
                #pdb.set_trace()
                tensorInput = torch.cat([ tensorInput, self.tensorPartial ], 1)
                tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

                #tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow*scale).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
                tensorOutput = custom_grid_sample(tensorInput, (self.tensorGrid + tensorFlow*scale).permute(0, 2, 3, 1))

                tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

                return tensorOutput[:, :-1, :, :] * tensorMask

        # PWC
        self.pwc = PwcNet()
        # self.pwc.load_state_dict(torch.load('./trained_models/sintel.pytorch'))
        self.pwc.eval()
        for param in self.pwc.parameters():
             param.requires_grad = False

        # Warping layer
        self.warpLayer = Backward()
        self.warpLayer.eval()

        # UNets
        self.UNet2 = UNet2()
        self.ResNet2 = ResNet2()

    def warpFrame(self, fr_1, fr_2, scale=1.0):
        with torch.no_grad():
            temp_w = int(math.floor(math.ceil(fr_1.size(3) / 64.0) * 64.0)) # Due to Pyramid method?
            temp_h = int(math.floor(math.ceil(fr_1.size(2) / 64.0) * 64.0))

            temp_fr_1 = torch.nn.functional.interpolate(input=fr_1, size=(temp_h, temp_w), mode='nearest')
            temp_fr_2 = torch.nn.functional.interpolate(input=fr_2, size=(temp_h, temp_w), mode='nearest')

            flo = 20.0 * torch.nn.functional.interpolate(input=self.pwc(temp_fr_1, temp_fr_2), size=(fr_1.size(2), fr_1.size(3)), mode='bilinear', align_corners=False)
            return self.warpLayer(fr_2, flo, scale), flo

    def forward(self, ip, scale= 0.5):
        prev_op, future_ip, curr_ip = ip[:, :3, :, :], ip[:, 3:6, :, :], ip[:, 6:, :, :] 
        fr1 = prev_op
        fr2 = future_ip
        f3 = curr_ip
        fs2 = future_ip.clone()
        fs1 = prev_op.clone()
        w1, flo1 = self.warpFrame(fs2, fr1, scale=scale)
        w2, flo2 = self.warpFrame(fs1, fr2, scale=scale)

        I_int = self.UNet2(w1, w2, flo1, flo2, fr1, fr2)
        f_int, flo_int = self.warpFrame(I_int, f3)

        fhat = self.ResNet2(I_int, f_int, flo_int, f3)
        return fhat# , I_int


class ResNet2(nn.Module):
    def __init__(self):
        super(ResNet2, self).__init__()

        class ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super(ConvBlock, self).__init__()

                self.seq = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                    nn.ReLU()
                )

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)


        class ResBlock(nn.Module):
            def __init__(self, num_ch):
                super(ResBlock, self).__init__()

                self.seq = nn.Sequential(
                    nn.Conv2d(num_ch, num_ch, kernel_size=1, stride=1, padding=0),
                    nn.ReLU()
                )

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x) + x


        self.seq = nn.Sequential(
            ConvBlock(11, 32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ConvBlock(32, 3),
            nn.Tanh()
        )

    def forward(self, I_int, f_int, flo_int, f3):
        return self.seq(torch.cat([I_int, f_int, flo_int, f3],1).cuda())



#############################################################################################################
class UNetFlow(nn.Module):
    def __init__(self):
        super(UNetFlow, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
                super(Encoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.LeakyReLU(0.2)
                )
                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class Decoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
                super(Decoder, self).__init__()
                
                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
                )

                if tanh:
                    self.activ = nn.Tanh()
                else:
                    self.activ = nn.LeakyReLU(0.2)
                
                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                s = self.seq(x)
                s = self.activ(s)
                return s * self.GateConv(x)


        self.enc0 = Encoder(4, 32, stride=1)
        self.enc1 = Encoder(32, 32, stride=2)
        self.enc2 = Encoder(32, 32, stride=2)
        self.enc3 = Encoder(32, 32, stride=2)

        self.dec0 = Decoder(32, 32, stride=1)
        # up-scaling + concat
        self.dec1 = Decoder(32+32, 32, stride=1)
        self.dec2 = Decoder(32+32, 32, stride=1)
        self.dec3 = Decoder(32+32, 32, stride=1)

        self.dec4 = Decoder(32, 2, stride=1)

    def forward(self, x1, x2):
        s0 = self.enc0(torch.cat([x1, x2],1).cuda())
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        s4 = self.dec0(s3)
        # up-scaling + concat
        s4 = F.interpolate(s4, scale_factor=2, mode='nearest')
        s5 = self.dec1(torch.cat([s4, s2],1).cuda())
        s5 = F.interpolate(s5, scale_factor=2, mode='nearest')
        s6 = self.dec2(torch.cat([s5, s1],1).cuda())
        s6 = F.interpolate(s6, scale_factor=2, mode='nearest')
        s7 = self.dec3(torch.cat([s6, s0],1).cuda())

        out = self.dec4(s7)
        return out