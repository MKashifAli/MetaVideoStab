import cv2
import numpy as np
import torch
import imageio
import flow_vis as fv


def decode_flow(flow_tensor, convert_to_bgr= False):
	flow = flow_tensor.clone().detach().cpu().numpy()[0]
	flow = np.transpose(flow, (1, 2, 0))
	flow_color = fv.flow_to_color(flow, convert_to_bgr= convert_to_bgr)
	return flow_color


def decode_frame(frame_tensor):
	frame = frame_tensor.clone().detach().cpu().numpy()[0]
	frame = np.transpose(frame, (1, 2, 0)) * 255.0
	return frame


def save_frame(f, name)
	f = decode_frame(f)
	cv2.imwrite(name, f)


def save_flow(f, name):
	f = decode_flow(f, True)
	cv2.imwrite(name, f)


def save_as_gif(lof, name):
	imageio.mimsave(name, lof, fps= 1)


def save_flow_list(lot, name):
	lof = []
	for t in lot:
		lof.append(cv2.cvtColor(decode_flow(t), cv2.COLOR_BGR2RGB))
	save_as_gif(lof, name)


def save_frame_list(lot, name):
	lof = []
	for t in lot:
		lof.append(decode_frame(t))
	save_as_gif(lof, name)
