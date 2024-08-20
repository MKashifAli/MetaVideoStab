from torchvision.models.optical_flow import raft_large
import torch
def get_raft_module(path= "./networks/models_raft/raft_ckpt.pth"):
	flownet = raft_large()
	flownet.load_state_dict(torch.load(path))
	flownet.eval()
	for params in flownet.parameters():
		params.requires_grad = False

	return flownet

def read_img_as_tensor(path):
	img = cv2.imread(path)
	img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)/255.0
	img_t = torch.tensor(img.astype(np.float32))
	return img_t

def estimate_flow(flownet, i1, i2):
	list_of_flows = flownet(i1, i2)
	predicted_flows = list_of_flows[-1]
	return predicted_flows

if __name__ == '__main__':
	import os 
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	import torch
	import cv2
	import numpy as np
	import flow_vis

	flownet = get_raft_module("./models_raft/raft_ckpt.pth")

	i1 = read_img_as_tensor("./models_raft/r0.png")
	i2 = read_img_as_tensor("./models_raft/r1.png")

	list_of_flows = flownet(i1, i2)
	predicted_flows = list_of_flows[-1]
	flow_uv = predicted_flows.cpu().detach().numpy()
	flow_uv = np.transpose(flow_uv[0], (1, 2, 0))
	np.save("./models_raft/flow_raw_raft.npy", flow_uv)

	i1 = read_img_as_tensor("./models_raft/p0.png")
	i2 = read_img_as_tensor("./models_raft/p1.png")

	list_of_flows = flownet(i1, i2)
	predicted_flows = list_of_flows[-1]
	flow_uv = predicted_flows.cpu().detach().numpy()
	flow_uv = np.transpose(flow_uv[0], (1, 2, 0))
	np.save("./models_raft/flow_pro_raft.npy", flow_uv)
	#flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=True)
	#cv2.imwrite("./models_raft/flow.png", flow_color)


