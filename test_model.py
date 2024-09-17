from train_model import Net, classes

import torchvision
import os
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import PIL

net = Net()
net.load_state_dict(torch.load('/home/kevin/dev/woob/modules/orange/pages/image_classifier.pth', weights_only=True))
net.eval()
code = []
transform = transforms.Compose([transforms.PILToTensor()])
files = [f'unsorted/{x}' for x in os.listdir("./unsorted")]
for image in files:
	pil_image = PIL.Image.open(image)
	torch_image = transform(pil_image).type(torch.float)
	torch_image = torch_image.unsqueeze(0)
	output = net(torch_image)
	pred = torch.max(output.data, 1)
	label = classes[int(pred.indices)]
	image = mpimg.imread(image)
	plt.imshow(image)
	plt.title(label)
	plt.show()
