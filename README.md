
# POLY-GAN - Experiment 
Our implementation is inspired by https://github.com/nile649/POLY-GAN
## Step-by-step instructions:
Download pre-trained models and dataset from https://drive.google.com/drive/folders/18eu3OrNh9TbmiED0sotbzGPtLCCecSeT?usp=sharing It has pre-trained weights for Stage 1, Stage 2 and Stage 3.
Open CLI, run
```
gdown --folder https://drive.google.com/drive/folders/18eu3OrNh9TbmiED0sotbzGPtLCCecSeT
mv ./pre_trained_models/data/data.zip .
unzip data.zip
```
### For training
Open CLI, run
```
python train.py --stage "Shape" # This will train Stage 1 Poly-Gan to change structure of Reference cloth
python train.py --stage "Stitch" # This will train Stage 2 Poly-Gan to stitch clothes to missing portion as shown in paper
python train.py --stage "Refine" # This will train Stage 3 Poly-Gan to refine for missing regions in result of Stage 2
```
### For testing
You can use given pretrained without training

Open CLI, excute
```
python test.py --stage "Stage3" # This will give the complete result from Stage 1 - Stage 4.
```
Or run to get all test set's results
```
python run.py
```
