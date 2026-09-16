import numpy as np
import skimage.io as skio
from skimage import img_as_float
from scipy.ndimage import convolve


SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32) / 8.0
SOBEL_Y = SOBEL_X.T


def sobel_magnitude(ch):
    gx = convolve(ch, SOBEL_X, mode="nearest")
    gy = convolve(ch, SOBEL_Y, mode="nearest")
    return np.hypot(gx, gy)


def interior(img, frac):
    h, w = img.shape
    dh, dw = int(h * frac), int(w * frac)
    return img[dh:h - dh, dw:w - dw]


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


BLUR_1D = np.array([1, 2, 1], dtype=np.float32) / 4.0


def downsample(img):
    blurred = convolve(img, BLUR_1D[None, :], mode="nearest")
    blurred = convolve(blurred, BLUR_1D[:, None], mode="nearest")
    return blurred[::2, ::2]


def build_pyramid(img, min_size):
    pyr = [img]
    while min(pyr[-1].shape) > min_size:
        pyr.append(downsample(pyr[-1]))
    return pyr


def search(moving, fixed, center, radius, frac):
    fixed_in = interior(fixed, frac)
    cy, cx = center
    best_score, best_shift = -np.inf, center
    for dy in range(cy - radius, cy + radius + 1):
        for dx in range(cx - radius, cx + radius + 1):
            shifted = np.roll(moving, (dy, dx), axis=(0, 1))
            score = ncc(interior(shifted, frac), fixed_in)
            if score > best_score:
                best_score, best_shift = score, (dy, dx)
    return best_shift


def align_edges(moving, fixed, coarse_radius=15, refine_radius=2,
                min_size=256, frac=0.15):
    pyr_m = build_pyramid(moving, min_size)
    pyr_f = build_pyramid(fixed, min_size)

    shift = (0, 0)
    for level in reversed(range(len(pyr_m))):
        coarsest = level == len(pyr_m) - 1
        if coarsest:
            radius = coarse_radius
        else:
            shift = (shift[0] * 2, shift[1] * 2)
            radius = refine_radius
        edges_m = sobel_magnitude(pyr_m[level])
        edges_f = sobel_magnitude(pyr_f[level])
        shift = search(edges_m, edges_f, shift, radius, frac)
    return shift


def colorize_edges(path, out_path=None):
    im = img_as_float(skio.imread(path)).astype(np.float32)
    h = im.shape[0] // 3
    b, g, r = im[:h], im[h:2 * h], im[2 * h:3 * h]

    g_shift = align_edges(g, b)
    r_shift = align_edges(r, b)
    g_aligned = np.roll(g, g_shift, axis=(0, 1))
    r_aligned = np.roll(r, r_shift, axis=(0, 1))

    print(f"{path}: G (x, y) = ({g_shift[1]}, {g_shift[0]}), "
          f"R (x, y) = ({r_shift[1]}, {r_shift[0]})")

    rgb = np.dstack([r_aligned, g_aligned, b])
    if out_path:
        out = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
        skio.imsave(out_path, out, quality=95)
    return rgb, g_shift, r_shift


if __name__ == "__main__":
    colorize_edges("CS180_fa2026_proj1_data/emir.tif", "out_path/out_emir_edges.jpg")
