import os
import numpy as np
import cv2
import skimage.data
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d
from filters import Dx, Dy, gaussian_kernel_2d

from paths import RESULTS


# can't use np.arctan2 so write my own
def my_atan(z):
    # polynomial approx of arctan, only works for |z| <= 1
    z2 = z * z
    return z * (0.9998660 + z2 * (-0.3302995 + z2 * (0.1801410 + z2 * (-0.0851330 + z2 * 0.0208351))))


def my_atan2(y, x):
    ax, ay = np.abs(x), np.abs(y)
    big = np.maximum(ax, ay)
    small = np.minimum(ax, ay)
    ratio = np.where(big > 0, small / np.where(big > 0, big, 1), 0)
    a = my_atan(ratio)
    a = np.where(ay > ax, np.pi / 2 - a, a)
    a = np.where(x < 0, np.pi - a, a)
    a = np.where(y < 0, -a, a)
    return a


def orientation_image(image, sigma=2, mag_scale=None):
    g = gaussian_kernel_2d(sigma)
    blurred = convolve2d(image, g, mode="same", boundary="symm")
    gx = convolve2d(blurred, Dx, mode="same", boundary="symm")
    gy = convolve2d(blurred, Dy, mode="same", boundary="symm")
    mag = np.sqrt(gx ** 2 + gy ** 2)
    theta = my_atan2(gy, gx)
    hue = (theta % (2 * np.pi)) / (2 * np.pi)
    val = np.clip(mag / (mag_scale or np.percentile(mag, 99)), 0, 1)
    return hsv_to_rgb(np.dstack([hue, np.ones_like(hue), val])), theta, mag


if __name__ == "__main__":
    cam = skimage.data.camera() / 255.0

    # check against numpy
    ys, xs = np.random.uniform(-1, 1, (2, 100000))
    print("max |my_atan2 - np.arctan2| = %.2e" % np.abs(my_atan2(ys, xs) - np.arctan2(ys, xs)).max())

    rgb, theta, mag = orientation_image(cam, sigma=2)
    cv2.imwrite(os.path.join(RESULTS, "bw_orientation.jpg"), (rgb[:, :, ::-1] * 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # colour wheel legend
    n = 200
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r = np.sqrt(xx ** 2 + yy ** 2)
    hue = (my_atan2(yy, xx) % (2 * np.pi)) / (2 * np.pi)
    wheel = hsv_to_rgb(np.dstack([hue, np.ones_like(hue), np.ones_like(hue)]))
    wheel[r > 1] = 1.0
    cv2.imwrite(os.path.join(RESULTS, "bw_wheel.jpg"), (wheel[:, :, ::-1] * 255).astype(np.uint8))
