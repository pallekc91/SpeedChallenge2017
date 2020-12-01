import numpy as np
import torch as t
from optical_flow import optical_flow
from model import get_nvidia_model
import cv2
import os

def get_frames(path):
    assert os.path.isfile(path), 'The path argument to get_frames is invalid'
    v = cv2.VideoCapture(path)
    images = {}
    i = 0
    while v.isOpened():
        success, img = v.read()
        if not success:
            break
        images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        i += 1
    v.release()
    return images, i


def get_targets(path):
    targets = {}
    with open(path,'r') as f:
        for i, line in enumerate(f.readlines()):
            targets[i] = float(line)
    i += 1
    return targets, i


def add_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5))


def change_brightness(image, bright_factor=0.2 + np.random.uniform()):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * bright_factor
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def reshape(image):
    image_cropped = image[25:375, :]  # clipping the sky and and the interior of car
    image = cv2.resize(image_cropped, (220, 66), interpolation=cv2.INTER_AREA)
    return image


def preprocess(image):
    img = change_brightness(image)
    img = reshape(img)
    return img


def split_data(data, seed_val=44):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    np.random.seed(seed_val)
    for i in range(len(data)):
        rand_int = np.random.randint(9)
        if 0 <= rand_int <= 1:
            val_x.append(data[i][0])
            val_y.append(data[i][1])
        else:
            train_x.append(data[i][0])
            train_y.append(data[i][1])
    return train_x, train_y, val_x, val_y

'''
def split_data(X, y, seed_val=55):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    np.random.seed(seed_val)
    for i in range(len(y)):
        rand_int = np.random.randint(9)
        if 0 <= rand_int <= 1:
            val_x.append(X[i])
            val_y.append(y[i])
        else:
            train_x.append(X[i])
            train_y.append(y[i])
    return train_x, train_y, val_x, val_y
'''

def process_frames(frames_map, targets):
    # you get a map of frames where key is the frame number and the images are RGB
    op_flow_out = {}
    prev = preprocess(frames_map[0])
    for i in range(len(frames_map)-1):
        cur = preprocess(frames_map[i+1])
        of_val = optical_flow(prev, cur)
        y = (targets[i] + targets[i+1])/2
        op_flow_out[i] = (of_val, y)
    return op_flow_out


class SpeedDataset(t.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
