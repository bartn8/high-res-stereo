import numpy as np
import cv2
import torch
import math
from torch.autograd import Variable

import torchvision.transforms as transforms

import importlib
hsm = importlib.import_module("thirdparty.high-res-stereo.models.hsm")
hsm_submodule = importlib.import_module("thirdparty.high-res-stereo.models.submodule")
hsm_preprocess = importlib.import_module("thirdparty.high-res-stereo.utils.preprocess")


class HSMBlock:
    def __init__(self, max_disparity = 192, clean = -1, level = 1, device = "cpu", verbose=False):
        self.logName = "HSM Block"
        self.verbose = verbose

        if max_disparity % 16 != 0:
            max_disparity = 16 * math.floor(max_disparity/16)
            max_disparity = int(max_disparity)

        self.max_disparity = max_disparity
        self.clean = clean
        self.level = level if level in [1,2,3] else 1
        self.device = device
        self.disposed = False
        self.processed = hsm_preprocess.get_transform()

    def log(self, x):
        if self.verbose:
            print(f"{self.logName}: {x}")

    def build_model(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(f"Building Model...")
        self.model = hsm.hsm(self.max_disparity,self.clean,level=self.level)
        self.model = torch.nn.DataParallel(self.model)

    def load(self, model_path):
        # load the checkpoint file specified by model_path.loadckpt
        print("loading model {}".format(model_path))
        pretrained_dict = torch.load(model_path)
        pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
        self.model.load_state_dict(pretrained_dict['state_dict'],strict=False)

        tmpdisp = int(self.max_disparity*1//64*64)
        
        if (self.max_disparity*1/64*64) > tmpdisp:
            self.model.module.maxdisp = tmpdisp + 64
        else:
            self.model.module.maxdisp = tmpdisp

        if self.model.module.maxdisp == 64: 
            self.model.module.maxdisp=128

        self.model.module.disp_reg8 =  hsm_submodule.disparityregression(self.model.module.maxdisp,16)
        self.model.module.disp_reg16 = hsm_submodule.disparityregression(self.model.module.maxdisp,16)
        self.model.module.disp_reg32 = hsm_submodule.disparityregression(self.model.module.maxdisp,32)
        self.model.module.disp_reg64 = hsm_submodule.disparityregression(self.model.module.maxdisp,64)
        print(self.model.module.maxdisp)

    def dispose(self):
        if not self.disposed:
            del self.model
            self.disposed = True


    def _conv_image(self, img):
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)

        h,w = img.shape[:2]
        img = self.processed(img).numpy()
        img = np.reshape(img,[1,3,img.shape[1],img.shape[2]])

        max_h = int(img.shape[2] // 64 * 64)
        max_w = int(img.shape[3] // 64 * 64)

        if max_h < img.shape[2]:
             max_h += 64
        
        if max_w < img.shape[3]:
             max_w += 64

        top_pad = max_h-img.shape[2]
        left_pad = max_w-img.shape[3]
        img = np.lib.pad(img,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        self.log(f"Original shape: {(h,w)}, padding: {(top_pad, left_pad)}, new shape: {img.shape}")

        top_pad   = max_h-h
        left_pad  = max_w-w

        return torch.from_numpy(np.expand_dims(img, axis=0)).to(self.device), top_pad, left_pad

    def test(self, left_vpp, right_vpp):
        #Input conversion
        left_vpp, top_pad, left_pad = self._conv_image(left_vpp)
        right_vpp, _, _ = self._conv_image(right_vpp)

        left_vpp = Variable(torch.FloatTensor(left_vpp))
        right_vpp = Variable(torch.FloatTensor(right_vpp))

        self.model.eval()
        with torch.no_grad():
            pred_disp,_  = self.model(left_vpp, right_vpp)
            pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
            #entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
            pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]    
            
            return pred_disp