import numpy as np
import cv2
from scipy.signal import convolve2d


# part 1.1 convolution
def pad_zeros(image, ph, pw):
    padded = np.zeros((image.shape[0] + 2 * ph, image.shape[1] + 2 * pw), dtype=float)
    padded[ph:ph + image.shape[0], pw:pw + image.shape[1]] = image
    return padded


def output_window(shape, kshape, mode):
    # which part of the full output to keep
    h, w = shape
    kh, kw = kshape
    if mode == "full":
        return 0, 0, h + kh - 1, w + kw - 1
    if mode == "same":
        return (kh - 1) // 2, (kw - 1) // 2, h, w
    if mode == "valid":
        return kh - 1, kw - 1, h - kh + 1, w - kw + 1
    raise ValueError(mode)


def conv2d_4loops(image, kernel, mode="same"):
    kh, kw = kernel.shape
    flipped = kernel[::-1, ::-1]
    padded = pad_zeros(image, kh - 1, kw - 1)
    r0, c0, out_h, out_w = output_window(image.shape, kernel.shape, mode)
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            total = 0.0
            for u in range(kh):
                for v in range(kw):
                    total += padded[r0 + i + u, c0 + j + v] * flipped[u, v]
            out[i, j] = total
    return out


def conv2d_2loops(image, kernel, mode="same"):
    kh, kw = kernel.shape
    flipped = kernel[::-1, ::-1]
    padded = pad_zeros(image, kh - 1, kw - 1)
    r0, c0, out_h, out_w = output_window(image.shape, kernel.shape, mode)
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(padded[r0 + i:r0 + i + kh, c0 + j:c0 + j + kw] * flipped)
    return out


# filters
Dx = np.array([[1, -1]], dtype=float)
Dy = np.array([[1], [-1]], dtype=float)


def box_filter(n):
    return np.ones((n, n)) / (n * n)


def gaussian_kernel_1d(sigma):
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.getGaussianKernel(ksize, sigma)


def gaussian_kernel_2d(sigma):
    g = gaussian_kernel_1d(sigma)
    return g @ g.T


# part 2
def blur(image, sigma):
    # symm so the edges don't go dark
    g = gaussian_kernel_1d(sigma)
    if image.ndim == 3:
        return np.dstack([blur(image[:, :, c], sigma) for c in range(image.shape[2])])
    out = convolve2d(image, g, mode="same", boundary="symm")
    return convolve2d(out, g.T, mode="same", boundary="symm")


def unsharp_kernel(sigma, alpha):
    # (1 + alpha) * delta - alpha * G
    g = gaussian_kernel_2d(sigma)
    delta = np.zeros_like(g)
    delta[g.shape[0] // 2, g.shape[1] // 2] = 1.0
    return (1 + alpha) * delta - alpha * g


def unsharp(image, sigma, alpha):
    kernel = unsharp_kernel(sigma, alpha)
    if image.ndim == 3:
        chans = [convolve2d(image[:, :, c], kernel, mode="same", boundary="symm")
                 for c in range(image.shape[2])]
        return np.clip(np.dstack(chans), 0, 1)
    return np.clip(convolve2d(image, kernel, mode="same", boundary="symm"), 0, 1)


def hybrid(low_img, high_img, sigma_low, sigma_high):
    low = blur(low_img, sigma_low)
    high = high_img - blur(high_img, sigma_high)
    return low, high, np.clip(low + high, 0, 1)


def gaussian_stack(image, levels, sigma0=2):
    # no downsampling, just blur more each level
    stack = [image]
    for i in range(1, levels):
        stack.append(blur(image, sigma0 * 2 ** (i - 1)))
    return stack


def laplacian_stack(image, levels, sigma0=2):
    g = gaussian_stack(image, levels, sigma0)
    lap = [g[i] - g[i + 1] for i in range(levels - 1)]
    lap.append(g[-1])
    return lap


def blend(img_a, img_b, mask, levels=6, sigma0=2):
    la = laplacian_stack(img_a, levels, sigma0)
    lb = laplacian_stack(img_b, levels, sigma0)
    gm = gaussian_stack(mask, levels, sigma0)
    bands = [la[i] * gm[i] + lb[i] * (1 - gm[i]) for i in range(levels)]
    return np.clip(sum(bands), 0, 1), bands
