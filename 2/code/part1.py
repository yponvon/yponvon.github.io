import os
import time
import numpy as np
import cv2
import skimage.data
from scipy.signal import convolve2d
from filters import (conv2d_4loops, conv2d_2loops, Dx, Dy, box_filter,
                     gaussian_kernel_2d)

from paths import MINE, RESULTS


def save(name, image):
    cv2.imwrite(os.path.join(RESULTS, name), np.clip(image * 255, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def stretch(image):
    return (image - image.min()) / (image.max() - image.min())


# 1.1
selfie = cv2.imread(os.path.join(MINE, "portrait.jpg"), cv2.IMREAD_GRAYSCALE) / 255.0
selfie = cv2.resize(selfie, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# test on a small crop since 4 loops is slow
crop = selfie[100:228, 100:228]
box9 = box_filter(9)
print("--- Part 1.1: runtime on a 128x128 crop, 9x9 box ---")
for mode in ["full", "same", "valid"]:
    t = time.time(); a = conv2d_4loops(crop, box9, mode); t4 = time.time() - t
    t = time.time(); b = conv2d_2loops(crop, box9, mode); t2 = time.time() - t
    t = time.time(); c = convolve2d(crop, box9, mode=mode, boundary="fill", fillvalue=0); ts = time.time() - t
    print("%-5s 4-loop %.3fs | 2-loop %.3fs | scipy %.5fs | max|4-scipy| %.1e | max|2-scipy| %.1e"
          % (mode, t4, t2, ts, np.abs(a - c).max(), np.abs(b - c).max()))
for name, k in [("Dx", Dx), ("Dy", Dy)]:
    c = convolve2d(crop, k, mode="same")
    print(name, "same: max diff (2-loop vs scipy) %.1e" % np.abs(conv2d_2loops(crop, k, "same") - c).max())

# whole photo
t = time.time(); box_out = conv2d_2loops(selfie, box9, "same"); print("selfie box 2-loop: %.2fs" % (time.time() - t))
t = time.time(); scipy_out = convolve2d(selfie, box9, mode="same"); print("selfie box scipy:  %.3fs" % (time.time() - t))
print("selfie box max diff vs scipy: %.1e" % np.abs(box_out - scipy_out).max())
dx_out = conv2d_2loops(selfie, Dx, "same")
dy_out = conv2d_2loops(selfie, Dy, "same")
save("p11_gray.jpg", selfie)
save("p11_box.jpg", box_out)
save("p11_dx.jpg", stretch(dx_out))
save("p11_dy.jpg", stretch(dy_out))

# 1.2
cam = skimage.data.camera() / 255.0
save("cameraman.jpg", cam)
gx = convolve2d(cam, Dx, mode="same", boundary="symm")
gy = convolve2d(cam, Dy, mode="same", boundary="symm")
mag = np.sqrt(gx ** 2 + gy ** 2)
save("p12_dx.jpg", stretch(gx))
save("p12_dy.jpg", stretch(gy))
save("p12_mag.jpg", mag / mag.max())
for t in [0.1, 0.15, 0.2, 0.3]:
    save("p12_thresh_%02d.jpg" % int(t * 100), (mag > t).astype(float))
print("--- Part 1.2: mag max %.3f" % mag.max())

# 1.3
SIGMA = 2
g2 = gaussian_kernel_2d(SIGMA)
print("gaussian ksize", g2.shape)
blurred = convolve2d(cam, g2, mode="same", boundary="symm")
bx = convolve2d(blurred, Dx, mode="same", boundary="symm")
by = convolve2d(blurred, Dy, mode="same", boundary="symm")
bmag = np.sqrt(bx ** 2 + by ** 2)
save("p13_blur.jpg", blurred)
save("p13_dx.jpg", stretch(bx))
save("p13_dy.jpg", stretch(by))
save("p13_mag.jpg", bmag / bmag.max())
for t in [0.03, 0.05, 0.08]:
    save("p13_thresh_%02d.jpg" % int(t * 100), (bmag > t).astype(float))
print("blurred mag max %.3f" % bmag.max())

# DoG
dog_x = convolve2d(g2, Dx, mode="full")
dog_y = convolve2d(g2, Dy, mode="full")
save("p13_dogx_filter.jpg", cv2.resize(stretch(dog_x), None, fx=20, fy=20, interpolation=cv2.INTER_NEAREST))
save("p13_dogy_filter.jpg", cv2.resize(stretch(dog_y), None, fx=20, fy=20, interpolation=cv2.INTER_NEAREST))
dx_dog = convolve2d(cam, dog_x, mode="same", boundary="symm")
dy_dog = convolve2d(cam, dog_y, mode="same", boundary="symm")
dmag = np.sqrt(dx_dog ** 2 + dy_dog ** 2)
save("p13_dogmag.jpg", dmag / dmag.max())
save("p13_dogthresh.jpg", (dmag > 0.05).astype(float))
save("p13_dogdx.jpg", stretch(dx_dog))
save("p13_dogdy.jpg", stretch(dy_dog))
m = 20
print("DoG vs blur-then-diff: max diff whole image %.2e, interior %.2e"
      % (np.abs(dmag - bmag).max(), np.abs(dmag - bmag)[m:-m, m:-m].max()))
print("dx: whole %.2e, interior %.2e" % (np.abs(dx_dog - bx).max(), np.abs(dx_dog - bx)[m:-m, m:-m].max()))
