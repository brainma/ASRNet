import argparse
import os
import sys
import re
import cv2
import pydicom as pyd
import torch
import torch.nn as nn
import numpy as np
from math import log10, sqrt
from skimage.measure import compare_ssim


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
parser.add_argument("--infer_num", type=bool, default=True, help='Number of intermediate frame')
parser.add_argument("--batch_size", type=int, default=4, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--outFolder", type=str, required=True, default="./", help='Specify the folder of output files. Default: ./')
parser.add_argument("--labelFolder", type=str, required=True, default="./", help='Specify the folder of output dicom files. Default: ./')
parser.add_argument("--inFolder", type=str, required=True, help='Specify the folder of input files.')
args = parser.parse_args()

root = args.inFolder
outFolder = args.outFolder
checkpoint = args.checkpoint
labelFolder = args.labelFolder
inferNum = args.infer_num
batchSize = args.batch_size
infolders = os.listdir(root)



if not os.path.isdir(outFolder):
    os.mkdir(outFolder)

for folder in infolders:
    inPath = os.path.join(root,folder)
    outPath = os.path.join(outFolder,folder)
    outDcmPath = os.path.join(outDcmFolder,folder)
    if not os.path.isdir(outPath):
        os.mkdir(outPath)
    if not os.path.isdir(os.path.join(outPath, 'DICOM')):
        os.mkdir(os.path.join(outPath,'DICOM'))
    if not os.path.isdir(os.path.join(outPath, 'PNG')):
        os.mkdir(os.path.join(outPath, 'PNG'))
    cmd = 'python test.py --batch_szie ' + batchSize + ' --infer_num '+ inferNum + ' --checkpoint ' + checkpoint + ' --inFolder ' + inPath + ' --outFolder ' + os.path.join(outPath,'PNG') + ' --outDicomFolder ' + os.path.join(outPath,'DICOM')
    os.system(cmd)


predictedFilePattern = re.compile('(\d+)_p\.png')
psnr_total = float(0)
ie_total = float(0)
ssim_total = float(0)
print('Case Num\tFile Name\tPSNR\tIE\tSSIM')
for folder in infolders:
    outPath = os.path.join(outFolder, folder, 'PNG')

    files = os.listdir(outPath)

    pre_num = 0
    psnr_case = float(0)
    ssim_case = float(0)
    ie_case = float(0)
    print(outPath)
    for file in files:
        m = predictedFilePattern.match(file)
        if m:
            predicted_img_path = os.path.join(outPath, file)
            label_img_path = os.path.join(labelFolder, folder, 'img'+m.group(1) + '.dcm')
            pre_img = cv2.imread(predicted_img_path, cv2.IMREAD_GRAYSCALE)
            label_dicom_img = pyd.dcmread(label_img_path)
            label_dicom_img.NumberOfFrames = 1
            label_img = label_dicom_img.pixel_array
            psnr = float(0)
            ie = float(0)
            for i in range(0,512):
                for j in range(0,512):
                    p = float(pre_img[i][j])
                    l = float(label_img[i][j])
                    psnr += np.square(p - l)

            ie = psnr = psnr / float((512*512))
            ie = sqrt(ie)
            ie_case  += ie
            psnr = (10 * log10(np.square(255) / psnr))
            psnr_case += psnr
            ssim = float(0)
            ssim = compare_ssim(pre_img,label_img)
            ssim_case += ssim
            pre_num += 1
            print(folder + '\t'+ file + '\t' + str(psnr) + '\t'+ str(ie) +'\t'+ str(ssim))

        else:
            continue
    
    print('Case Num: ' + folder) 
    psnr_case = psnr_case/float(pre_num)
    ie_case = ie_case/float(pre_num)
    ssim_case = ssim_case/float(pre_num)
    print(str(folder) + '\t' + str(psnr_case) + '\t' + str(ie_case) + '\t' + str(ssim_case))
    psnr_total += psnr_case
    ie_total += ie_case
    ssim_total += ssim_case
psnr_total = psnr_total/float(len(infolders))
print('\nAve psnr of total cases: ' + str(psnr_total))
ie_total = ie_total/float(len(infolders))
print('Ave ie of total cases: ' + str(ie_total))
ssim_total = ssim_total/float(len(infolders))
print('Ave ssim of total cases: ' + str(ssim_total))
