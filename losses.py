import os
import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import networks
import contextual_loss as cx
from model_cs import CoarseStabilizerInferReady


def gradient_loss(gen_frames, gt_frames, alpha=1):

    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)


def gram(x):
	b,c,h,w = x.size();
	x = x.view(b*c, -1);
	return torch.mm(x, x.t())


class outer_loop_loss_affine(nn.Module):
	def __init__(self, flownet, interpolator, summary_writer= None):
		super(outer_loop_loss_affine, self).__init__()
		self.crit = cx.ContextualLoss(use_vgg= True, vgg_layer='relu3_4', band_width= 0.1, loss_type= 'cosine').cuda()
		self.flownet = flownet
		self.summary_writer = summary_writer
		self.writer_counter = 0
		self.interpolator = interpolator
		self.l2 = nn.MSELoss(size_average=True)
		self.l1 = nn.L1Loss()
		#self.affine = CoarseStabilizerInferReady(self.flownet)
		

	def forward(self, semi_stable, stable):	
		#### Flow loss: 	loss between semi stable and transformed (Working: minimize the motion (flow) between the generated and transformed (coarsely stable) frames)...
		#### Pixel loss:	pixel space loss between the generated and transformed stable...
		
		motion_between_gen_and_transformed = 0
		pixel_loss = 0


		for i in range(1, len(semi_stable)):
			#### Flow loss
			flow_stable = self.flownet.estimateFlowFull(stable[i - 1], stable[i])
			flow_unstable = self.flownet.estimateFlowFull(semi_stable[i - 1], semi_stable[i])
			motion_between_gen_and_transformed = motion_between_gen_and_transformed + self.l2(flow_unstable, flow_stable)
			

			#motion_between_gen_and_transformed = motion_between_gen_and_transformed + self.l2(torch.zeros_like(flow).cuda(), flow)
			
			#### Pixel loss
			pixel_loss += self.crit(semi_stable[i], stable[i])

		curr_loss = (10.0*torch.abs(motion_between_gen_and_transformed) + 100.0*pixel_loss)/(len(semi_stable) )



		if self.summary_writer is not None:
			self.summary_writer.add_scalar('outer_loop_loss/pixel_loss', pixel_loss.item(), self.writer_counter)
			self.summary_writer.add_scalar('outer_loop_loss/motion_loss', torch.abs(motion_between_gen_and_transformed).item(), self.writer_counter)
			self.writer_counter += 1
		
		return curr_loss

	def epe(self, tenFlow, tenTruth):
		tenEpe = torch.mean(torch.sqrt((tenFlow[:, 0:1, :, :] - tenTruth[:, 0:1, :, :])**2 + (tenFlow[:, 1:2, :, :] - tenTruth[:, 1:2, :, :])**2))
		return tenEpe


class inner_loop_loss_perc_affine_another_try(nn.Module):
	def __init__(self, loss_fn, interpolator, flownet, summary_writer= None, use_vgg= True):
		super(inner_loop_loss_perc_affine_another_try, self).__init__()
		self.summary_writer = summary_writer
		self.writer_counter = 0
		self.crit = loss_fn
		self.cx = cx.ContextualLoss(use_vgg= True, vgg_layer='relu3_4', band_width= 0.1, loss_type= 'cosine').cuda()
		self.interpolator = interpolator
		self.flownet = flownet
		if use_vgg:
			self.use_vgg = True
			import networks
			import utils
			self.norm_vgg = utils.normalize_ImageNet_stats
			self.VGG = networks.Vgg16(requires_grad= False)
			self.VGG = self.VGG.cuda()
			VGGLayers = [int(layer) for layer in list("4")]
			VGGLayers.sort()
			self.VGGLayers = [layer-1 for layer in list(VGGLayers)]

		self.affine = CoarseStabilizerInferReady(self.flownet)


	def forward(self, semi_stable, unstable):
		#with torch.no_grad():
		unstable_txs = []
		unstable_affs = []
		for i in range(1, len(unstable)):
			tx, op = self.affine(unstable[0], unstable[i])
			unstable_txs.append(tx)
			unstable_affs.append(op)

		#### Flow loss: 	loss between semi stable and transformed (Working: minimize the motion (flow) between the generated and transformed (coarsely stable) frames)...
		#### Affine loss: 	loss between the original and lerped transforms...
		#### Pixel loss:	pixel space loss between the generated and transformed stable...
		motion_between_gen_and_transformed = 0
		aff_loss = 0
		pixel_loss = 0
		vgg_loss = 0
		for i in range(1, len(semi_stable)):
			#### Flow loss
			flow = self.flownet(unstable_txs[i - 1], semi_stable[i])
			
			motion_between_gen_and_transformed = motion_between_gen_and_transformed + (torch.mean(flow))
			
			#### Pixel loss + Affine loss
			pixel_loss = pixel_loss + self.cx(semi_stable[i], unstable[i])
			
			f_x = self.norm_vgg(semi_stable[i])
			f_y = self.norm_vgg(unstable_txs[i - 1])
			f_yy = self.norm_vgg(unstable[i])
			for l in self.VGGLayers:
				f_x = self.VGG(f_x, self.VGGLayers[-1])
				f_y = self.VGG(f_y, self.VGGLayers[-1])
				f_yy = self.VGG(f_yy, self.VGGLayers[-1])
				vgg_loss = vgg_loss + self.crit(f_x[l], f_y[l])
				vgg_loss = vgg_loss + self.crit(gram(f_x[l]), gram(f_yy[l]))
			#if i < len(semi_stable) - 1:
			#	y_, _ = self.interpolator(semi_stable[i - 1].clone().detach(), semi_stable[i + 1].clone().detach())
			#	pixel_loss = pixel_loss + self.crit(semi_stable[i], y_)

		if self.summary_writer is not None:
			self.summary_writer.add_scalar('inner_loop_loss/pixel_loss', pixel_loss.item(), self.writer_counter)
			self.summary_writer.add_scalar('inner_loop_loss/motion_loss', torch.abs(motion_between_gen_and_transformed).item(), self.writer_counter)
			self.summary_writer.add_scalar('inner_loop_loss/perc_loss', vgg_loss.item(), self.writer_counter)
			self.writer_counter += 1
		curr_loss = (pixel_loss) + 10*(torch.abs(motion_between_gen_and_transformed) + vgg_loss)

		return curr_loss