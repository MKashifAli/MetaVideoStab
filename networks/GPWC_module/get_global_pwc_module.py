from .GlobalFlowNets.GlobalPWCNets import getGlobalPWCModel_mine

def get_global_pwc(path= "./pretrained_models/GFlowNet.pth"):
    OptNet = getGlobalPWCModel_mine(path)
    return OptNet
