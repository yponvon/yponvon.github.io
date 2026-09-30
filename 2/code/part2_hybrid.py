import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from filters import blur, hybrid

from paths import COURSE, MINE, RESULTS


def save(name, image):
    cv2.imwrite(os.path.join(RESULTS, name), np.clip(image * 255, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def align_to(image, src_pts, dst_pts, shape):
    # line up two points (the eyes)
    M, _ = cv2.estimateAffinePartial2D(np.float32(src_pts), np.float32(dst_pts))
    return cv2.warpAffine(image, M, (shape[1], shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def gray(image):
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(float)


def fft_log(image):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray(image) if image.ndim == 3 else image))) + 1e-8)


def with_small(image, scale=0.2):
    return np.clip(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), 0, 1)


def make_hybrid(name, low, high, sigma_low, sigma_high):
    _, _, hy = hybrid(low, high, sigma_low, sigma_high)
    save("hy_%s.jpg" % name, hy)
    save("hy_%s_far.jpg" % name, with_small(hy))
    return hy


# derek + nutmeg
derek = cv2.imread(os.path.join(COURSE, "DerekPicture.jpg")) / 255.0
nutmeg = cv2.imread(os.path.join(COURSE, "nutmeg.jpg")) / 255.0
derek = cv2.resize(derek, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

derek_eyes = [(159, 173), (222.5, 166.5)]
nutmeg_eyes = [(599, 286), (750, 366)]
nutmeg_al = align_to(nutmeg, nutmeg_eyes, derek_eyes, derek.shape[:2])
save("hy_derek.jpg", derek)
save("hy_nutmeg.jpg", nutmeg)
save("hy_nutmeg_aligned.jpg", nutmeg_al)

SIGMA_LOW, SIGMA_HIGH = 8, 4
low, high, hy1 = hybrid(derek, nutmeg_al, SIGMA_LOW, SIGMA_HIGH)
save("hy_derek_nutmeg.jpg", hy1)
save("hy_derek_nutmeg_far.jpg", with_small(hy1))
save("hy_derek_low.jpg", low)
save("hy_nutmeg_high.jpg", high + 0.5)

# try a few cutoffs
for sl, sh in [(4, 2), (8, 4), (16, 8)]:
    _, _, h = hybrid(derek, nutmeg_al, sl, sh)
    save("hy_cutoff_%d_%d.jpg" % (sl, sh), h)
    save("hy_cutoff_%d_%d_far.jpg" % (sl, sh), with_small(h))

# fft
panels = [("Derek (aligned)", derek), ("Nutmeg (aligned)", nutmeg_al),
          ("Derek low-passed", low), ("Nutmeg high-passed", high), ("hybrid", hy1)]
fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
for k, (title, img) in enumerate(panels):
    g = gray(img)
    axes[0, k].imshow(g, cmap="gray", vmin=0 if k != 3 else -0.2, vmax=1 if k != 3 else 0.2)
    axes[0, k].set_title(title)
    axes[1, k].imshow(fft_log(img), cmap="viridis")
    axes[1, k].set_title("log |FFT|")
    axes[0, k].axis("off")
    axes[1, k].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "hy_fourier.jpg"), dpi=70)
plt.close()

# b&w: where to put colour
low_c = blur(derek, SIGMA_LOW)
high_c = nutmeg_al - blur(nutmeg_al, SIGMA_HIGH)
low_g = blur(gray(derek), SIGMA_LOW)
high_g = gray(nutmeg_al) - blur(gray(nutmeg_al), SIGMA_HIGH)
combos = {
    "gray": np.dstack([np.clip(low_g + high_g, 0, 1)] * 3),
    "color_low": np.clip(low_c + high_g[:, :, None], 0, 1),
    "color_high": np.clip(low_g[:, :, None] + high_c, 0, 1),
    "color_both": np.clip(low_c + high_c, 0, 1),
}
for k, v in combos.items():
    save("hy_color_%s.jpg" % k, v)
    save("hy_color_%s_far.jpg" % k, with_small(v))

# jisoo + wonyoung
jisoo = cv2.imread(os.path.join(MINE, "jisoo.jpg")) / 255.0
wonyoung = cv2.imread(os.path.join(MINE, "wonyoung.jpg")) / 255.0
jisoo_eyes = [(185, 138), (258, 128)]
wonyoung_eyes = [(282, 232), (388, 228)]
wonyoung_al = align_to(wonyoung, wonyoung_eyes, jisoo_eyes, jisoo.shape[:2])
save("hy_jisoo.jpg", jisoo)
save("hy_wonyoung.jpg", wonyoung)
save("hy_wonyoung_aligned.jpg", wonyoung_al)
make_hybrid("jisoo_wonyoung", jisoo, wonyoung_al, 9, 4)

# harry + hermione
potter = cv2.imread(os.path.join(MINE, "potter.jpg"))[:, 115:570] / 255.0  # crop so hermione covers the whole frame
hermione = cv2.imread(os.path.join(MINE, "hermione.jpg")) / 255.0
potter_eyes = [(197, 143), (283, 140)]
hermione_eyes = [(260, 215), (370, 210)]
hermione_al = align_to(hermione, hermione_eyes, potter_eyes, potter.shape[:2])
save("hy_potter.jpg", potter)
save("hy_hermione.jpg", hermione)
save("hy_hermione_aligned.jpg", hermione_al)
make_hybrid("potter_hermione", potter, hermione_al, 9, 4)

# cutoff frequencies
for name, s in [("hybrid 1 low", SIGMA_LOW), ("hybrid 1 high", SIGMA_HIGH),
               ("hybrid 2/3 low", 9), ("hybrid 2/3 high", 4)]:
    print("%s: sigma %g -> cutoff ~ %.4f cycles/pixel" % (name, s, 1 / (2 * np.pi * s)))
