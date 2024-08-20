import torch
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()


def get_raft_module():
	flownet = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
	flownet = flownet.eval()

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



class RAFT(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(RAFT, self).__init__()
        self.flownet = get_raft_module()
        self.estimate = estimate_flow

    def forward(self, i1, i2):
    	lof = self.flownet(i1, i2)
    	return lof[-1]



if __name__ == '__main__':
	import torch
	import cv2
	import numpy as np
	import flow_vis

	flownet = get_raft_module()

	i1 = read_img_as_tensor("./models_raft/1.png")
	i2 = read_img_as_tensor("./models_raft/2.png")

	list_of_flows = flownet(i1, i2)
	predicted_flows = list_of_flows[-1]
	flow_uv = predicted_flows.cpu().detach().numpy()
	flow_uv = np.transpose(flow_uv[0], (1, 2, 0))
	flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=True)
	cv2.imwrite("./models_raft/flow.png", flow_color)


