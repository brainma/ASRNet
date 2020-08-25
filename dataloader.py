import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import re
import pydicom as pyd

def _sort_files(files):
    pattern = re.compile('^img(\d+)\.[a-z]+')
    tmp = [''] * 133
    for f in files:
        m = pattern.match(f)
        tmp[int(m.group(1)) - 1] = f
    i = 0
    for f in tmp:
        if(f != ''):
            files[i] = f
            i += 1

def _make_dataset(dir):
    framesPath = []
    for index, folder in enumerate(os.listdir(dir)):
        inputFolderPath = os.path.join(dir, folder)
        if not (os.path.isdir(inputFolderPath)):
            continue
        framesPath.append([])
        images = list(os.listdir(inputFolderPath))
        _sort_files(images)
        for image in images:
            framesPath[index].append(os.path.join(inputFolderPath, image))
    return framesPath

def _make_test_dataset(dir):
    framesPath = []
    images = list(os.listdir(dir))
    _sort_files(images)
    for image in images:
        framesPath.append(os.path.join(dir, image))
    return framesPath

def _dicom_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    dicom_img = pyd.dcmread(path)
    dicom_img.NumberOfFrames = 1
    img = dicom_img.pixel_array
    img = Image.fromarray(img)
    resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
    cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
    flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
    
    return flipped_img, dicom_img

class AsrNet(data.Dataset):
    
    def __init__(self, root, transform=None, dim=(512, 512), randomCropSize=(448, 448), inferNum, train=True):
       
        framesPath = _make_dataset(root)
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train
        self.framesPath     = framesPath
        self.inferNum = inferNum

    def __getitem__(self, index):
        
        sample = []
        totalFrameNum = len(self.framesPath[index])
        if (self.train):
            firstFrame = random.randint(0, totalFrameNum - (self.inferNum + 2))
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            IFrameIndex = random.randint(firstFrame + 1, firstFrame + self.inferNum)
            if (random.randint(0, 1)):
                frameRange = [firstFrame, IFrameIndex, firstFrame + (self.inferNum + 1)]
                returnIndex = IFrameIndex - firstFrame - 1
            else:
                frameRange = [firstFrame + (self.inferNum + 1), IFrameIndex, firstFrame]
                returnIndex = firstFrame - IFrameIndex + self.inferNum
            randomFrameFlip = random.randint(0, 1)
        else:
            firstFrame = 0
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            IFrameIndex = ((index) % self.inferNum  + 1)
            returnIndex = IFrameIndex - 1
            frameRange = [0, IFrameIndex, (self.inferNum + 1)]
            randomFrameFlip = 0

        for frameIndex in frameRange:
            image = _dicom_loader(self.framesPath[index][frameIndex], cropArea=cropArea, frameFlip=randomFrameFlip)
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)

        return sample, returnIndex


class AsrNetTest(data.Dataset):

    def __init__(self, root, transform=None):
        
        framesPath = _make_test_dataset(root)
        frame, self.dicom_template = _dicom_loader(framesPath[0])
        self.origDim = frame.size
        self.dim = int(self.origDim[0] / 32) * 32, int(self.origDim[1] / 32) * 32
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in: " + root + "\n"))

        self.root           = root
        self.framesPath     = framesPath
        self.transform      = transform

    def __getitem__(self, index):
        
        sample = []
        for framePath in [self.framesPath[index], self.framesPath[index + 1]]:
            image, _ = _dicom_loader(framePath, resizeDim=self.dim)
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample
