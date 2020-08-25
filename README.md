# Angular Super-Resolution in X-Ray Projection Radiography Using Deep Neural Network: Implementation on Rotational Angiography
Code for the paper: "Angular Super-Resolution in X-Ray Projection Radiography Using Deep Neural Network: Implementation on Rotational Angiography" by Tiing Yee Siow, Cheng-Yu Ma, and Cheng Hong Toh.
NOTE: This code is not intended for clinical use.

## Prerequisites
numpy==1.18.3
Pillow==6.1.0
pydicom==1.4.2
six==1.14.0
torch==1.4.0
torchvision==0.5.0
tqdm==4.45.0

## Training
### Preparing training data
In order to train your own model using the provided code with your own dataset, the data needs to be formatted in a certain file structure. 
The data should follow the format: 
Training data folder: dataset/train/patientID/imgX.dcm
Validation data folder: dataset/validation/patientID/imgx.dcm
Testing data folder: dataset/test/patientID/imgx.dcm
Testing data label folder: dataset/test/patientID/label/imgx.dcm

### Command
```bash
python train.py --dataset_root path\to\dataset --checkpoint_dir path\to\save\checkpoints
```

## Testing
```bash
python run_test.py --inFolder path\to\test_dataset --checkpoint path\to\save\checkpoints\ASRNETx.ckpt --labelFolder path\to\labelFolder --infer_num numberOfIntermediateFrame
```

## Run the following commmand for help / more options
```bash
python train.py --h
python test.py --h
python run_test.py --h
```

## Tensorboard
```bash
tensorboard --logdir log --port 6007
```
