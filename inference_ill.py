import cv2
import numpy as np
import torch
from skimage.filters.rank import mean_bilateral
from skimage import morphology
from PIL import Image
from PIL import ImageEnhance


def padCropImg(img):
    H = img.shape[0]
    W = img.shape[1]

    patchRes = 128
    pH = patchRes
    pW = patchRes
    ovlp = int(patchRes * 0.125)  # 32

    padH = (int((H - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - W

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)

    ynum = int((padImg.shape[0] - pH) / (pH - ovlp)) + 1
    xnum = int((padImg.shape[1] - pW) / (pW - ovlp)) + 1

    totalPatch = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)

    for j in range(0, ynum):
        for i in range(0, xnum):
            x = int(i * (pW - ovlp))
            y = int(j * (pH - ovlp))
            
            if j == (ynum-1) and i == (xnum-1):
                totalPatch[j, i] = img[-patchRes:, -patchRes:]
            elif j == (ynum-1):
                totalPatch[j, i] = img[-patchRes:, x:int(x + patchRes)]
            elif i == (xnum-1):
                totalPatch[j, i] = img[y:int(y + patchRes), -patchRes:]
            else:
                totalPatch[j, i] = padImg[y:int(y + patchRes), x:int(x + patchRes)]

    return totalPatch, padH, padW


def illCorrection(model, totalPatch):
    totalPatch = totalPatch.astype(np.float32) / 255.0

    ynum = totalPatch.shape[0]
    xnum = totalPatch.shape[1]

    totalResults = np.zeros((ynum, xnum, 128, 128, 3), dtype=np.float32)

    for j in range(0, ynum):
        for i in range(0, xnum):
            patchImg = torch.from_numpy(totalPatch[j, i]).permute(2,0,1)
            patchImg = patchImg.cuda().view(1, 3, 128, 128)

            output = model(patchImg)
            output = output.permute(0, 2, 3, 1).data.cpu().numpy()[0]

            output = output * 255.0
            output = output.astype(np.uint8)

            totalResults[j, i] = output

    return totalResults


def composePatch(totalResults, padH, padW, img):
    ynum = totalResults.shape[0]
    xnum = totalResults.shape[1]
    patchRes = totalResults.shape[2]

    ovlp = int(patchRes * 0.125)
    step = patchRes - ovlp

    resImg = np.zeros((patchRes + (ynum - 1) * step, patchRes + (xnum - 1) * step, 3), np.uint8)
    resImg = np.zeros_like(img).astype('uint8')

    for j in range(0, ynum):
        for i in range(0, xnum):
            sy = int(j * step)
            sx = int(i * step)
            
            if j == 0 and i != (xnum-1):
                resImg[sy:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i]
            elif i == 0 and j != (ynum-1):
                resImg[sy+10:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i,10:]
            elif j == (ynum-1) and i == (xnum-1):
                resImg[-patchRes+10:, -patchRes+10:] = totalResults[j, i,10:,10:]
            elif j == (ynum-1) and i == 0:
                resImg[-patchRes+10:, sx:(sx + patchRes)] = totalResults[j, i,10:]
            elif j == (ynum-1) and i != 0:
                resImg[-patchRes+10:, sx+10:(sx + patchRes)] = totalResults[j, i,10:,10:]
            elif i == (xnum-1) and j == 0:
                resImg[sy:(sy + patchRes), -patchRes+10:] = totalResults[j, i,:,10:]
            elif i == (xnum-1) and j != 0:
                resImg[sy+10:(sy + patchRes), -patchRes+10:] = totalResults[j, i,10:,10:]
            else:
                resImg[sy+10:(sy + patchRes), sx+10:(sx + patchRes)] = totalResults[j, i,10:,10:]

    resImg[0,:,:] = 255

    return resImg


def preProcess(img):
    img[:,:,0] = mean_bilateral(img[:,:,0], morphology.disk(20), s0=10, s1=10)
    img[:,:,1] = mean_bilateral(img[:,:,1], morphology.disk(20), s0=10, s1=10)
    img[:,:,2] = mean_bilateral(img[:,:,2], morphology.disk(20), s0=10, s1=10)
    
    return img


def postProcess(img):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    factor = 2.0
    img = enhancer.enhance(factor)

    return img


def rec_ill(net, img, saveRecPath):

    totalPatch, padH, padW = padCropImg(img)

    totalResults = illCorrection(net, totalPatch)

    resImg = composePatch(totalResults, padH, padW, img)
    #resImg = postProcess(resImg)
    resImg = Image.fromarray(resImg)
    resImg.save(saveRecPath)  
