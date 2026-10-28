import os
import json
import numpy as np
import cv2

from paths import MINE, RESULTS, POINTS

WIDTH = 1200  # photos are shrunk to this width before clicking points, so points and warps match


def load(name, width=WIDTH):
    image = cv2.imread(os.path.join(MINE, name)).astype(float) / 255.0
    if width is not None and image.shape[1] > width:
        s = width / image.shape[1]
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return image


def save(name, image):
    cv2.imwrite(os.path.join(RESULTS, name), np.clip(image * 255, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def load_points(name):
    with open(os.path.join(POINTS, name + ".json")) as f:
        d = json.load(f)
    pts2 = np.array(d["pts2"], float) if "pts2" in d else None
    return d["images"], np.array(d["pts1"], float), pts2


# A.2 homographies
def system(im1_pts, im2_pts):
    # h = [a b c d e f g h], last entry of H fixed to 1
    # x' = (ax + by + c) / (gx + hy + 1)  ->  ax + by + c - gxx' - hyx' = x'
    A, b = [], []
    for (x, y), (xp, yp) in zip(im1_pts, im2_pts):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
        b.append(xp)
        b.append(yp)
    return np.array(A, float), np.array(b, float)


def computeH(im1_pts, im2_pts):
    # least squares, so more than 4 points averages out clicking error
    A, b = system(im1_pts, im2_pts)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)


def apply_H(H, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:]


# A.3 warping
def warped_box(shape, H):
    # where the 4 corners land = size of the output
    h, w = shape[:2]
    corners = apply_H(H, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float))
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    return x0, y0, x1, y1


def source_coords(H, box):
    # inverse warping: every output pixel looks up where it came from, so no holes
    x0, y0, x1, y1 = box
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    src = apply_H(np.linalg.inv(H), np.column_stack([xs.ravel(), ys.ravel()]))
    return src[:, 0].reshape(xs.shape), src[:, 1].reshape(xs.shape)


def warpImageNearestNeighbor(im, H, box=None):
    # (0, 0) is the center of the top-left pixel
    if box is None:
        box = warped_box(im.shape, H)
    sx, sy = source_coords(H, box)
    h, w = im.shape[:2]
    cx, cy = np.round(sx).astype(int), np.round(sy).astype(int)
    alpha = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    out = np.zeros(sx.shape + im.shape[2:])
    out[alpha] = im[cy[alpha], cx[alpha]]
    return out, alpha.astype(float)


def warpImageBilinear(im, H, box=None):
    if box is None:
        box = warped_box(im.shape, H)
    sx, sy = source_coords(H, box)
    h, w = im.shape[:2]
    alpha = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    x, y = sx[alpha], sy[alpha]
    # min so x0 + 1 doesn't go past the last column/row
    x0 = np.minimum(np.floor(x).astype(int), w - 2)
    y0 = np.minimum(np.floor(y).astype(int), h - 2)
    fx, fy = x - x0, y - y0
    if im.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    top = im[y0, x0] * (1 - fx) + im[y0, x0 + 1] * fx
    bottom = im[y0 + 1, x0] * (1 - fx) + im[y0 + 1, x0 + 1] * fx
    out = np.zeros(sx.shape + im.shape[2:])
    out[alpha] = top * (1 - fy) + bottom * fy
    return out, alpha.astype(float)


# A.4 mosaics
def edge_weight(shape):
    # 1 in the middle of the photo, linearly down to 0 at its edges
    h, w = shape[:2]
    wx = 1 - np.abs(np.linspace(-1, 1, w))
    wy = 1 - np.abs(np.linspace(-1, 1, h))
    return np.minimum(wy[:, None], wx[None, :])


def mosaic_box(images, Hs):
    boxes = np.array([warped_box(im.shape, H) for im, H in zip(images, Hs)])
    return boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()


def warp_all(images, Hs):
    # everything goes into one canvas big enough for all of them
    box = mosaic_box(images, Hs)
    warped, weights = [], []
    for im, H in zip(images, Hs):
        out, alpha = warpImageBilinear(im, H, box)
        wt, _ = warpImageBilinear(edge_weight(im.shape), H, box)
        warped.append(out)
        weights.append(wt * alpha + 1e-6 * alpha)  # tiny bit so the outer edge isn't 0/0
    return warped, weights


def blend_overwrite(warped, weights):
    # no blending, later images just cover earlier ones
    out = np.zeros_like(warped[0])
    for im, wt in zip(warped, weights):
        out[wt > 0] = im[wt > 0]
    return out


def blend_average(warped, weights):
    total = np.maximum(sum(weights), 1e-12)
    return sum(im * wt[..., None] for im, wt in zip(warped, weights)) / total[..., None]


def blur(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)


def blend_two_band(warped, weights, sigma=4):
    # 2 level laplacian pyramid:
    # low freqs -> weighted average so brightness changes fade smoothly
    # high freqs -> only from the photo with the biggest weight, so edges don't ghost
    lows, highs = [], []
    for im, wt in zip(warped, weights):
        a = (wt > 0).astype(float)
        # blur only inside the photo, otherwise the black outside leaks in
        low = blur(im * a[..., None], sigma) / np.maximum(blur(a, sigma), 1e-12)[..., None]
        low *= a[..., None]
        lows.append(low)
        highs.append(im - low)
    low = blend_average(lows, weights)
    best = np.argmax(np.stack(weights), axis=0)
    high = np.zeros_like(low)
    for i, hi in enumerate(highs):
        high[best == i] = hi[best == i]
    return np.clip(low + high, 0, 1)
