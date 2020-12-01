import cv2
import numpy as np


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def optical_flow(img_1, img_2):
    img_1_gray = to_gray(img_1)
    img_2_gray = to_gray(img_2)

    # parameters
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # return
    hsv = np.zeros((66, 220, 3))
    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(img_2, cv2.COLOR_RGB2HSV)[:, :, 1]

    flow = cv2.calcOpticalFlowFarneback(img_1_gray, img_2_gray, flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        extra)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)
    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_flow
