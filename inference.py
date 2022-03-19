from seg import U2NETP
from GeoTr import GeoTr
from IllTr import IllTr
from inference_ill import rec_ill

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import numpy as np
import cv2
import glob
import os
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')


class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)
        
    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x

        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        
        return bm
        

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        

def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        

def rec(opt):
    # print(torch.__version__) # 1.5.1
    img_list = os.listdir(opt.distorrted_path)  # distorted images list

    if not os.path.exists(opt.gsave_path):  # create save path
        os.mkdir(opt.gsave_path)
    if not os.path.exists(opt.isave_path):  # create save path
        os.mkdir(opt.isave_path)
    
    GeoTr_Seg_model = GeoTr_Seg().cuda()
    # reload segmentation model
    reload_segmodel(GeoTr_Seg_model.msk, opt.Seg_path)
    # reload geometric unwarping model
    reload_model(GeoTr_Seg_model.GeoTr, opt.GeoTr_path)
    
    IllTr_model = IllTr().cuda()
    # reload illumination rectification model
    reload_model(IllTr_model, opt.IllTr_path)
    
    # To eval mode
    GeoTr_Seg_model.eval()
    IllTr_model.eval()
  
    for img_path in img_list:
        name = img_path.split('.')[-2]  # image name

        img_path = opt.distorrted_path + img_path  # read image and to tensor
        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255. 
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)
        
        with torch.no_grad():
            # geometric unwarping
            bm = GeoTr_Seg_model(im.cuda())
            bm = bm.cpu()
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
            
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2,0,1).unsqueeze(0).float(), lbl, align_corners=True)
            img_geo = ((out[0]*255).permute(1, 2, 0).numpy()).astype(np.uint8)
            io.imsave(opt.gsave_path + name + '_geo' + '.png', img_geo)  # save
            
            # illumination rectification
            if opt.ill_rec:
                ill_savep = opt.isave_path + name + '_ill' + '.png'
                rec_ill(IllTr_model, img_geo, saveRecPath=ill_savep)
        
        print('Done: ', img_path)


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--distorrted_path',  default='./distorted/')
    parser.add_argument('--gsave_path',  default='./geo_rec/')
    parser.add_argument('--isave_path',  default='./ill_rec/')
    parser.add_argument('--Seg_path',  default='./model_pretrained/seg.pth')
    parser.add_argument('--GeoTr_path',  default='./model_pretrained/geotr.pth')
    parser.add_argument('--IllTr_path',  default='./model_pretrained/illtr.pth')
    parser.add_argument('--ill_rec',  default=False)
    
    opt = parser.parse_args()

    rec(opt)


if __name__ == '__main__':
    main()
