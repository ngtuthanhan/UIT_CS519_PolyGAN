from skimage.metrics import structural_similarity as ssim
import torch
import torchvision
from torchmetrics.image.inception import InceptionScore
import numpy as np
import cv2
inception = InceptionScore()
img1 = cv2.imread('./results/test/Stage3temp_res/1_000010.jpg')
img2 = cv2.imread('./data/test/image/000010_0.jpg')
img2 = cv2.resize(img2,(128,128))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
score = ssim(gray1, gray2)
img1 = torchvision.io.read_image('./results/test/Stage3temp_res/1_000010.jpg')
img2 = torchvision.io.read_image('./results/test/Stage3temp_res/1_000001.jpg')
img1 = torchvision.transforms.Resize((299,299))(img1)
img1 = img1[None, :]
img2 = torchvision.transforms.Resize((299,299))(img2)
img2 = img2[None, :]
img1 = torch.cat((img1, img2))
print(img1.shape)
img1 = torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8)
inception.update(img1)
print("IS:", inception.compute())
# 6. You can print only the score if you want
print("SSIM: {}".format(score))
