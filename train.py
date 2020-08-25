import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import dataloader
from math import log10
import datetime
import time
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help='path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to folder for saving checkpoints')
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--infer_num", type=bool, default=True, help='Number of intermediate frame')
parser.add_argument("--train_continue", type=bool, default=False, help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
parser.add_argument("--train_batch_size", type=int, default=4, help='batch size for training.')
parser.add_argument("--validation_batch_size", type=int, default=10, help='batch size for validation. Default: 10.')
parser.add_argument("--init_learning_rate", type=float, default=0.0001, help='set initial learning rate. Default: 0.0001.')
parser.add_argument("--milestones", type=list, default=[100, 150], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
parser.add_argument("--progress_iter", type=int, default=100, help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
parser.add_argument("--checkpoint_epoch", type=int, default=5, help='checkpoint saving frequency. N: after every N epochs.Default: 5.')
parser.add_argument("--logfolder", type=str, required=True, default='log', help='path of log folder.')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flowComp = model.UNet(2, 4)
flowComp.to(device)
ArbTimeFlowIntrp = model.UNet(12, 5)
ArbTimeFlowIntrp.to(device)

trainFlowBackWarp      = model.backWarp(448, 448, device)
trainFlowBackWarp      = trainFlowBackWarp.to(device)
validationFlowBackWarp = model.backWarp(512, 512, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)

mean = [0.5]
std  = [1]
normalize = transforms.Normalize(mean=mean,
                                 std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

trainset = dataloader.AsrNet(root=args.dataset_root + '/train', transform=transform, inferNum=args.infer_num, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)

validationset = dataloader.AsrNet(root=args.dataset_root + '/validation', transform=transform, randomCropSize=(512, 512), inferNum=args.infer_num, train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False)

negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()
params = list(ArbTimeFlowIntrp.parameters()) + list(flowComp.parameters())
optimizer = optim.Adam(params, lr=args.init_learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

vgg16 = torchvision.models.vgg16()
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)
for param in vgg16_conv_4_3.parameters():
    param.requires_grad = False


def validate():
    psnr = 0
    tloss = 0
    flag = 1
    with torch.no_grad():
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            fCoeff = model.getFlowCoeff(validationFrameIndex, device, args.infer_num)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)

            intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1   = 1 - V_t_0

            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)

            wCoeff = model.getWarpCoeff(validationFrameIndex, device, args.infer_num)

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            if (flag):
                retImg = torchvision.utils.make_grid([revNormalize(frame0[0]), revNormalize(frameT[0]), revNormalize(Ft_p.cpu()[0]), revNormalize(frame1[0])], padding=10)
                flag = 0

            #Loss calculation
            recnLoss = L1_lossFn(Ft_p, IFrame)
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
            tloss += loss.item()

            MSE_val = MSE_LossFn(Ft_p, IFrame)
            psnr += (10 * log10(1 / MSE_val.item()))

    return (psnr / len(validationloader)), (tloss / len(validationloader)), retImg

if args.train_continue:
    dict1 = torch.load(args.checkpoint)
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])
else:
    dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}

start = time.time()
cLoss   = dict1['loss']
valLoss = dict1['valLoss']
valPSNR = dict1['valPSNR']
checkpoint_counter = 0
if not os.path.isdir(args.logfolder):
    os.mkdir(args.logfolder)
writer = SummaryWriter(args.logfolder)
if not os.path.isdir(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

for epoch in range(dict1['epoch'] + 1, args.epochs):
    print("Epoch: ", epoch)

    cLoss.append([])
    valLoss.append([])
    valPSNR.append([])
    iLoss = 0

    scheduler.step()

    for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
        frame0, frameT, frame1 = trainData
        I0 = frame0.to(device)
        I1 = frame1.to(device)
        IFrame = frameT.to(device)
        optimizer.zero_grad()
        flowOut = flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]
        fCoeff = model.getFlowCoeff(trainFrameIndex, device, args.infer_num)
        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
        g_I0_F_t_0 = trainFlowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)
        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0
        g_I0_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)
        wCoeff = model.getWarpCoeff(trainFrameIndex, device, args.infer_num)
        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

        # Loss calculation
        recnLoss = L1_lossFn(Ft_p, IFrame)
        prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
        warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)
        loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
        loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
        loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
        loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
        loss.backward()
        optimizer.step()
        iLoss += loss.item()
        if ((trainIndex % args.progress_iter) == args.progress_iter - 1):
            end = time.time()
            psnr, vLoss, valImg = validate()
            valPSNR[epoch].append(psnr)
            valLoss[epoch].append(vLoss)

            #Tensorboard
            itr = trainIndex + epoch * (len(trainloader))
            writer.add_scalars('Loss', {'trainLoss': iLoss/args.progress_iter, 'validationLoss': vLoss}, itr)
            writer.add_scalar('PSNR', psnr, itr)
            writer.add_image('Validation',valImg , itr)
            endVal = time.time()
            print(" Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (iLoss / args.progress_iter, trainIndex, len(trainloader), end - start, vLoss, psnr, endVal - end, get_lr(optimizer)))
            cLoss[epoch].append(iLoss/args.progress_iter)
            iLoss = 0
            start = time.time()
    # Save checkpoint
    if ((epoch % args.checkpoint_epoch) == args.checkpoint_epoch - 1):
        dict1 = {
                'Detail':'AsrNet.',
                'epoch':epoch,
                'timestamp':datetime.datetime.now(),
                'trainBatchSz':args.train_batch_size,
                'validationBatchSz':args.validation_batch_size,
                'learningRate':get_lr(optimizer),
                'loss':cLoss,
                'valLoss':valLoss,
                'valPSNR':valPSNR,
                'state_dictFC': flowComp.state_dict(),
                'state_dictAT': ArbTimeFlowIntrp.state_dict(),
                }
        torch.save(dict1, args.checkpoint_dir + "/ASRNet" + str(epoch) + ".ckpt")
        checkpoint_counter += 1
writer.close()
