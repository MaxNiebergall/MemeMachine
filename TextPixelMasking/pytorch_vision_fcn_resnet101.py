# %% [markdown]
# ### This notebook is optionally accelerated with a GPU runtime.
# ### If you would like to use this acceleration, please select the menu option "Runtime" -> "Change runtime type", select "Hardware Accelerator" -> "GPU" and click "SAVE"
# 
# ----------------------------------------------------------------------
# 
# # FCN
# 
# *Author: Pytorch Team*
# 
# **Fully-Convolutional Network model with ResNet-50 and ResNet-101 backbones**
# 
# _ | _
# - | -
# ![alt](https://pytorch.org/assets/images/deeplab1.png) | ![alt](https://pytorch.org/assets/images/fcn2.png)

# %%
import torch
import torchvision
import torchvision.transforms as transforms
model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
# or
# model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
model.eval()

# %% [markdown]
# All pre-trained models expect input images normalized in the same way,
# i.e. mini-batches of 3-channel RGB images of shape `(N, 3, H, W)`, where `N` is the number of images, `H` and `W` are expected to be at least `224` pixels.
# The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
# and `std = [0.229, 0.224, 0.225]`.
# 
# The model returns an `OrderedDict` with two Tensors that are of the same height and width as the input Tensor, but with 21 classes.
# `output['out']` contains the semantic masks, and `output['aux']` contains the auxillary loss values per-pixel. In inference mode, `output['aux']` is not useful.
# So, `output['out']` is of shape `(N, 21, H, W)`. More documentation can be found [here](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection).

# %%
# Download an example image from the pytorch website
"""
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
"""

# Select an image from the dataset
from torch.utils.data import DataLoader
from generate_training_validation_data import CustomImageDataset
import numpy as np
import matplotlib.pyplot as plt

train_data_dir = 'D:/MemeMachine_ProjectData/dataset/training'
validation_data_dir = 'D:/MemeMachine_ProjectData/dataset/validation'
img_width, img_height, n_channels = 257, 257, 3 #TODO change dimensions to be wider, to better support text

epochs = 1 #50 TODO
batch_size = 1

#TODO change image_with_text_functions.generate_text_on_image_and_pixel_mask_from_path to place the text properly
train_dataset = CustomImageDataset(train_data_dir, img_width, img_height)
test_dataset = CustomImageDataset(validation_data_dir, img_width, img_height)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
input_image = train_features[0].squeeze()
input_image = np.moveaxis(input_image.numpy(), 0, -1)
label = train_labels[0].reshape((img_width, img_height))

plt.imshow(input_image, cmap="gray")
plt.show()
plt.imshow(label, cmap="gray")
plt.show()

# %%
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
import cv2 as cv
# input_image = Image.open(filename)
input_image2 = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
input_image2 = Image.fromarray(np.uint8(input_image2))
input_image2 = input_image2.convert("RGB")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0] #zero refers to the batch number?
output_predictions = output.argmax(0)
print(output_predictions)
print(output_predictions.shape)
print(output)
print(output.shape)


# %% [markdown]
# The output here is of shape `(21, H, W)`, and at each location, there are unnormalized probabilities corresponding to the prediction of each class.
# To get the maximum prediction of each class, and then use it for a downstream task, you can do `output_predictions = output.argmax(0)`.
# 
# Here's a small snippet that plots the predictions, with each color being assigned to each class (see the visualized image on the left).

# %%
# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image2.size)
r.putpalette(colors)

import matplotlib.pyplot as plt
plt.imshow(r)
# plt.show()

# %% [markdown]
# ### Model Description
# 
# FCN-ResNet is constructed by a Fully-Convolutional Network model, using a ResNet-50 or a ResNet-101 backbone.
# The pre-trained models have been trained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
# 
# Their accuracies of the pre-trained models evaluated on COCO val2017 dataset are listed below.
# 
# | Model structure |   Mean IOU  | Global Pixelwise Accuracy |
# | --------------- | ----------- | --------------------------|
# |  fcn_resnet50   |   60.5      |   91.4                    |
# |  fcn_resnet101  |   63.7      |   91.9                    |
# 
# ### Resources
# 
#  - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211)

# %%
import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=10**-4, momentum=0.99)
model.train()

# %%
# Train the model
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(inputs)['out'][0]
        # print(output.shape)
        output = output[0]
        # print(output.shape)
        output = torch.unsqueeze(output, 0)
        output = torch.unsqueeze(output, 0)
        # print(output.shape)

        # output_predictions = output.argmax(0)
        # print(labels.shape)
        labels = torch.reshape(labels, (1,257,257))
        labels = labels.long()
        # print(labels.shape)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# %%
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %% [markdown]
# Show a test of the newly trained (fine tuned) model below

# %%
model.eval()

# Display image and label.
test_features, test_labels = next(iter(test_dataloader))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")
input_image = test_features[0].squeeze()
input_image = np.moveaxis(input_image.numpy(), 0, -1)
label = test_labels[0].reshape((img_width, img_height))

plt.imshow(input_image, cmap="gray")
plt.show()
plt.imshow(label, cmap="gray")
plt.show()

# %%
# input_image = Image.open(filename)
input_image2 = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
input_image2 = Image.fromarray(np.uint8(input_image2))
input_image2 = input_image2.convert("RGB")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0] #zero refers to the batch number?
output_predictions = output.argmax(0)
print(output_predictions)
print(output_predictions.shape)
print(output)
print(output.shape)

# %%
test(test_dataloader, model, criterion)

# %%



