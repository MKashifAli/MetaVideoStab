import os
import numpy as np
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import networks
import contextual_loss as cx
from model_cs import CoarseStabilizerInferReady



def gram(x):
	b,c,h,w = x.size()
	x = x.view(b*c, -1)
	return torch.mm(x, x.t())



class inner_loop_loss_perc_affine(nn.Module): 
	def __init__(self, loss_fn, flownet, summary_writer= None, use_vgg= True):
		super(inner_loop_loss_perc_affine, self).__init__()
		self.summary_writer = summary_writer
		self.writer_counter = 0
		self.crit = loss_fn
		self.cx = cx.ContextualLoss(use_vgg= True, vgg_layer='relu3_4', band_width= 0.1, loss_type= 'cosine').cuda()
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
		unstable_txs = []
		unstable_affs = []
		for i in range(1, len(unstable)):
			tx, op = self.affine(unstable[0], unstable[i])
			unstable_txs.append(tx)
			unstable_affs.append(op)

		#### Flow loss: 	loss between semi stable and transformed (Working: minimize the motion (flow) between the regressed and transformed (affine aligned) frames)...
		#### Affine loss: 	loss between the original and lerped transforms...
		#### Pixel loss:	pixel space loss between the regressed and transformed stable...
		
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
			

		if self.summary_writer is not None:
			self.summary_writer.add_scalar('inner_loop_loss/pixel_loss', pixel_loss.item(), self.writer_counter)
			self.summary_writer.add_scalar('inner_loop_loss/motion_loss', torch.abs(motion_between_gen_and_transformed).item(), self.writer_counter)
			self.summary_writer.add_scalar('inner_loop_loss/perc_loss', vgg_loss.item(), self.writer_counter)
			self.writer_counter += 1
		curr_loss = (pixel_loss) + 10*(torch.abs(motion_between_gen_and_transformed) + vgg_loss)
		return curr_loss


