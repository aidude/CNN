# CNN using pytorch
# amritansh


import torch
import torchvision

import load_data
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(load_data.trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % load_data.classes[labels[j]] for j in range(4)))


# lets make a CNN using Pytorch

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv_1 = nn.Conv2d(3,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv_2 = nn.Conv2d(6,16,5)
		self.fc_1 = nn.Linear(16*5*5, 120)
		self.fc_2 = nn.Linear(120, 84)
		self.fc_3 = nn.Linear(84,10)

def forward(self, x):
	x = self.pool(F.relu(self.conv_1(x)))
	x = self.pool(F.relu(self.conv_2(x)))
	x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc_1(x))
    x = F.relu(self.fc_2(x))
    x = self.fc_3(x)
    return x


convNet = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(5):

	network_loss = 0.0
	for i,data in enumerate(load_data.trainloader, 0):
		# get the inputs
		inputs, labels = data

		inputs, labels = Variable.(inputs), Variable(labels)


		optimizer.zero_grad()

		outputs = convNet(inputs)
		loss = criterion(outputs, labels)
		loss.backwards()
		optimizer.step()

		network_loss += loss.data[0]

		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, network_loss / 2000))
            network_loss = 0.0

print('Finished Training')


dataiter = iter(data_load.testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % data_load.classes[labels[j]] for j in range(4)))

