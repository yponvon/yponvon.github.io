import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from filters import gaussian_stack, laplacian_stack, blend

from paths import COURSE, MINE, DOWNLOADED, RESULTS


def save(name, image):
    cv2.imwrite(os.path.join(RESULTS, name), np.clip(image * 255, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def load(path, width=None):
    image = cv2.imread(path).astype(float) / 255.0
    if width is not None:
        image = cv2.resize(image, None, fx=width / image.shape[1], fy=width / image.shape[1],
                           interpolation=cv2.INTER_AREA)
    return image


def align_to(image, src_pts, dst_pts, shape):
    M, _ = cv2.estimateAffinePartial2D(np.float32(src_pts), np.float32(dst_pts))
    return cv2.warpAffine(image, M, (shape[1], shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def show_band(band, last, gain):
    # laplacian levels can be negative so shift to gray
    return np.clip(band, 0, 1) if last else np.clip(band * gain + 0.5, 0, 1)


LEVELS = 5
GAINS = [4, 3, 3, 3, 1]

# 2.3 stacks
apple_path = os.path.join(COURSE, "apple.jpg")
orange_path = os.path.join(COURSE, "orange.jpg")
if os.path.exists(apple_path) and os.path.exists(orange_path):
    apple = load(apple_path)
    orange = load(orange_path)
    h, w = apple.shape[:2]

    for name, img in [("apple", apple), ("orange", orange)]:
        g = gaussian_stack(img, LEVELS)
        l = laplacian_stack(img, LEVELS)
        err = np.abs(sum(l) - img).max()
        print(name, "sum of laplacian stack - original: max err %.1e" % err)
        fig, axes = plt.subplots(2, LEVELS, figsize=(3 * LEVELS, 6.2))
        for i in range(LEVELS):
            axes[0, i].imshow(np.clip(g[i], 0, 1)[:, :, ::-1])
            axes[0, i].set_title("Gaussian level %d" % i)
            axes[1, i].imshow(show_band(l[i], i == LEVELS - 1, GAINS[i])[:, :, ::-1])
            axes[1, i].set_title("Laplacian level %d" % i)
            axes[0, i].axis("off"); axes[1, i].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "st_%s.jpg" % name), dpi=60)
        plt.close()

    # fig 3.42
    mask = np.zeros((h, w, 3))
    mask[:, : w // 2] = 1.0
    la, lb, gm = laplacian_stack(apple, LEVELS), laplacian_stack(orange, LEVELS), gaussian_stack(mask, LEVELS)
    oraple, bands = blend(apple, orange, mask, LEVELS)
    save("bl_oraple.jpg", oraple)
    columns = [
        ("apple x mask", [la[i] * gm[i] for i in range(LEVELS)]),
        ("orange x (1 - mask)", [lb[i] * (1 - gm[i]) for i in range(LEVELS)]),
        ("blend", bands),
    ]
    shown = [0, 2, 4]
    fig, axes = plt.subplots(4, 3, figsize=(9, 12.4))
    for c, (title, parts) in enumerate(columns):
        for r, i in enumerate(shown):
            axes[r, c].imshow(show_band(parts[i], i == LEVELS - 1, GAINS[i])[:, :, ::-1])
            axes[r, c].set_title("%s, level %d" % (title, i), fontsize=9)
        axes[3, c].imshow(np.clip(sum(parts), 0, 1)[:, :, ::-1])
        axes[3, c].set_title("%s, all levels summed" % title, fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "bl_fig342.jpg"), dpi=65)
    plt.close()

    hard = np.where(mask > 0.5, apple, orange)
    save("bl_hard.jpg", hard)
else:
    print("SKIPPING apple/orange (Part 2.3 stacks + oraple + Fig 3.42): "
          "put the course's apple.jpg and orange.jpg in %s" % COURSE)


# 2.4 blends
def ellipse_mask(shape, center, axes, angle=0):
    m = np.zeros(shape[:2])
    cv2.ellipse(m, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), angle, 0, 360, 1.0, -1)
    return np.dstack([m] * 3)


def save_blend(name, img_a, img_b, mask, levels=6):
    out, bands = blend(img_a, img_b, mask, levels)
    save("bl_%s_a.jpg" % name, img_a)
    save("bl_%s_b.jpg" % name, img_b)
    save("bl_%s_mask.jpg" % name, mask)
    save("bl_%s.jpg" % name, out)
    save("bl_%s_naive.jpg" % name, np.where(mask > 0.5, img_a, img_b))  # no blending
    return out, bands


# singapore skyline with calgary's mountains behind it
def skyline_mask(image):
    # walk down each column until it stops looking like sky, everything below is city
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(float)
    hue, sat, val = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    sky = (val > 0.62) & ((sat < 0.3) | ((hue > 95) & (hue < 125)))
    sky[:, 150:420] |= (sat[:, 150:420] < 0.45) & (val[:, 150:420] > 0.45)  # dark cloud above the towers
    h, w = sky.shape
    top = np.zeros(w, int)
    for x in range(w):
        y = 0
        while y < h - 15 and (sky[y, x] or sky[y:y + 12, x].mean() > 0.5):
            y += 1
        top[x] = y
    top = median_filter(top, 5)
    city = (np.arange(h)[:, None] >= top[None, :]).astype(float)
    return np.dstack([city] * 3)


singapore = load(os.path.join(DOWNLOADED, "singapore.jpg"))
calgary = load(os.path.join(DOWNLOADED, "calgary.jpg"))
calgary = calgary[30:30 + singapore.shape[0]]  # line the mountains up behind the lower buildings
city_mask = skyline_mask(singapore)
save_blend("skyline", singapore, calgary, city_mask)

# cheetah + cat
cat = load(os.path.join(MINE, "cheetahcat.jpg"))
cheetah = load(os.path.join(MINE, "cheetah.jpg"))
cat_eyes = [(285, 352), (450, 345)]
cheetah_eyes = [(320, 265), (498, 228)]
cheetah_al = align_to(cheetah, cheetah_eyes, cat_eyes, cat.shape[:2])
cat_face = ellipse_mask(cat.shape, (368, 420), (160, 150))
cheetahcat, cheetahcat_bands = save_blend("cheetahcat", cheetah_al, cat, cat_face)

# stack figure for cheetah cat
lc = laplacian_stack(cheetah_al, 6)
lt = laplacian_stack(cat, 6)
gm_face = gaussian_stack(cat_face, 6)
crop = (slice(200, 640), slice(150, 590))  # crop to face
fig, axes = plt.subplots(3, 6, figsize=(18, 9.4))
for i in range(6):
    last, g = i == 5, (3 if i < 5 else 1)
    axes[0, i].imshow(show_band((lc[i] * gm_face[i])[crop], last, g)[:, :, ::-1])
    axes[1, i].imshow(show_band((lt[i] * (1 - gm_face[i]))[crop], last, g)[:, :, ::-1])
    axes[2, i].imshow(show_band(cheetahcat_bands[i][crop], last, g)[:, :, ::-1])
    axes[0, i].set_title("cheetah x mask, level %d" % i, fontsize=10)
    axes[1, i].set_title("cat x (1 - mask), level %d" % i, fontsize=10)
    axes[2, i].set_title("sum, level %d" % i, fontsize=10)
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "bl_cheetahcat_stack.jpg"), dpi=55)
plt.close()

# b&w: gray vs colour
def to_gray3(image):
    g = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(float)
    return np.dstack([g] * 3)

skyline_gray, _ = blend(to_gray3(singapore), to_gray3(calgary), city_mask, 6)
save("bl_skyline_gray.jpg", skyline_gray)
