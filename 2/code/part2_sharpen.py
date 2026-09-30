import os
import numpy as np
import cv2
from filters import blur, unsharp, unsharp_kernel

from paths import COURSE, MINE, RESULTS


def save(name, image):
    cv2.imwrite(os.path.join(RESULTS, name), np.clip(image * 255, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 95])


def load(path, width=None):
    image = cv2.imread(path).astype(float) / 255.0
    if width is not None:
        image = cv2.resize(image, None, fx=width / image.shape[1], fy=width / image.shape[1],
                           interpolation=cv2.INTER_AREA)
    return image



def sharpen_report(tag, image, sigma, alphas):
    blurred = blur(image, sigma)
    high = image - blurred
    save("sh_%s_orig.jpg" % tag, image)
    save("sh_%s_blur.jpg" % tag, blurred)
    save("sh_%s_high.jpg" % tag, np.clip(high * 4 + 0.5, 0, 1))  # x4 so you can see it
    for a in alphas:
        save("sh_%s_a%g.jpg" % (tag, a), unsharp(image, sigma, a))
    print(tag, "sigma", sigma, "kernel shape", unsharp_kernel(sigma, 1).shape)


# 2.1
taj = load(os.path.join(COURSE, "taj.jpg"))
sharpen_report("taj", taj, 2, [0.5, 1, 2, 4])

# blur.jpg is already out of focus
blurry = load(os.path.join(MINE, "blur.jpg"), width=700)
sharpen_report("blurry", blurry, 4, [1, 2, 4, 8])

# blur a sharp photo then sharpen it back
sharp = load(os.path.join(MINE, "mountain.jpg"), width=600)
SIGMA_BLUR = 2
blurred = blur(sharp, SIGMA_BLUR)
save("sh_eval_orig.jpg", sharp)
save("sh_eval_blur.jpg", blurred)
for a in [0.5, 1, 2, 4]:
    restored = unsharp(blurred, SIGMA_BLUR, a)
    save("sh_eval_a%g.jpg" % a, restored)

# zoom in
def crop_zoom(image, y, x, h, w, z=4):
    return cv2.resize(image[y:y + h, x:x + w], None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)

for name, img in [("orig", sharp), ("blur", blurred), ("a1", unsharp(blurred, SIGMA_BLUR, 1)),
                  ("a4", unsharp(blurred, SIGMA_BLUR, 4))]:
    save("sh_eval_zoom_%s.jpg" % name, crop_zoom(img, 345, 440, 90, 155, z=3))
