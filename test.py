import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import model
import dataloader
import platform
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
parser.add_argument("--infer_num", type=bool, default=True, help='Number of inferred frame')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--outFolder", type=str, required=True, default="./", help='Specify the folder of output files. Default: ./')
parser.add_argument("--outDicomFolder", type=str, required=True, default="./", help='Specify the folder of output dicom files. Default: ./')
parser.add_argument("--inFolder", type=str, required=True, help='Specify the folder of input files.')
args = parser.parse_args()

def check():

    error = ""
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    return error


def main():
    error = check()
    if error:
        print(error)
        exit(1)
    outputPath = args.outFolder
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    mean = [0.5]
    std  = [1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Data Loader
    frames = dataloader.AsrNetTest(root=args.inFolder, transform=transform)
    framesLoader = torch.utils.data.DataLoader(frames, batch_size=args.batch_size, shuffle=False)

    # Network arch
    flowComp = model.UNet(2, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = model.UNet(12, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backWarp(512, 512, device)
    flowBackWarp = flowBackWarp.to(device)

    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])
    dicomTemplate = frames.dicom_template
    dicomTemplate.NumberOfFrames = 1

    frameCounter = 1

    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(framesLoader, 0):

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            flowIn = torch.cat((I0, I1), dim=1)
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            for batchIndex in range(args.batch_size):
                (TP(frame0[batchIndex].detach())).save(os.path.join(outputPath, 'img' + str(frameCounter) + ".png"), compress_level = 0)
                copy(os.path.join(args.inFolder, 'img'+str(frameCounter) +'.dcm' ), args.outDicomFolder)
            frameCounter += 1

            for intermediateIndex in range(1, args.infer_num + 1):
                t = intermediateIndex / (args.infer_num + 1)
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)
                intrpIn = torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1)
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0
                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save output frame
                for batchIndex in range(args.batch_size):
                    img = TP(Ft_p[batchIndex].cpu().detach())
                    img.save(os.path.join(outputPath, 'img' + str(frameCounter) + "_p.png"), compress_level = 0)
                    img_array = np.asarray(img)
                    dicomTemplate.PixelData = img_array.tobytes()
                    dicomTemplate.save_as(os.path.join(args.outDicomFolder, 'img' + str(frameCounter)+ '_p.dcm'))
                frameCounter += 1

            frameCounter += (args.infer_num + 1) * (args.batch_size - 1)

    exit(0)

main()
