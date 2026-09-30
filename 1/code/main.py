import os
import numpy as np
import skimage as sk
import skimage.io as skio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "images", "course")   # the provided data.zip, unzipped
OUT = os.path.join(HERE, "..", "images", "output")
jpg_filenames = [
    os.path.join(DATA, 'cathedral.jpg'),
    os.path.join(DATA, 'monastery.jpg'),
    os.path.join(DATA, 'tobolsk.jpg'),
    os.path.join(DATA, 'ornament.jpg'),
    os.path.join(DATA, 'specimens.jpg'),
    os.path.join(DATA, 'iconostasis.jpg'),
]
tif_filenames = [
    os.path.join(DATA, 'church.tif'),
    os.path.join(DATA, 'emir.tif'),
    os.path.join(DATA, 'harvesters.tif'),
    os.path.join(DATA, 'icon.tif'),
    os.path.join(DATA, 'ilemselga.tif'),
    os.path.join(DATA, 'melons.tif'),
    os.path.join(DATA, 'religous_painting.tif'),
    os.path.join(DATA, 'self_portrait.tif'),
    os.path.join(DATA, 'siren.tif'),
    os.path.join(DATA, 'three_generations.tif'),
    os.path.join(DATA, 'wharf.tif'),
]

os.makedirs(OUT, exist_ok=True)


# ---- alignment metric (SSD, equivalent to L2 for picking the best shift) ----
def alignment_metric(name, image1, image2):
        if name == "SSD":
            difference = image1 - image2
            sq_difference = np.square(difference)
            final = np.sum(sq_difference)
            return final
        if name == "NCC":
            mean_1 = np.mean(image1)
            mean_2 = np.mean(image2)
            norm_1 = np.linalg.norm(image1)
            norm_2 = np.linalg.norm(image2)
            normalize_1 = (image1 - mean_1) / norm_1
            normalize_2 = (image2 - mean_2) / norm_2
            similarity = np.dot(normalize_1.flatten(), normalize_2.flatten())
            return similarity


# ---- single-scale alignment: exhaustive +/-15px search ----
def align(image, image_fixed):
        score = float('inf')
        best_dx = None
        best_dy = None
        m = 15

        for dx in range(-15, 16):
            for dy in range(-15, 16):
                shifted_image = np.roll(image, dx, axis=1)
                shifted_image = np.roll(shifted_image, dy, axis=0)

                cropped_image = shifted_image[m:-m, m:-m]
                cropped_fixed = image_fixed[m:-m, m:-m]
                new_score = alignment_metric("SSD", cropped_image, cropped_fixed)
                if new_score < score:
                    score = new_score
                    best_dx = dx
                    best_dy = dy
        best_image = np.roll(image, best_dx, axis=1)
        best_image = np.roll(best_image, best_dy, axis=0)
        print(best_dx, best_dy)
        return best_image, (best_dx, best_dy)


# ---- pyramid downsampling ----
def blur_axis(image, axis):
    left = np.roll(image, 1, axis=axis)
    right = np.roll(image, -1, axis=axis)
    return (left + 2 * image + right) / 4.0


def downsample(image):
    blurred = blur_axis(image, axis=0)
    blurred = blur_axis(blurred, axis=1)
    return blurred[::2, ::2]


# ---- coarse-to-fine pyramid alignment ----
def pyramid_align(image, image_fixed, threshold=400):
    if min(image.shape) < threshold:
        _, (best_dx, best_dy) = align(image, image_fixed)
        return best_dx, best_dy
    else:
        small_image = downsample(image)
        small_fixed = downsample(image_fixed)

        coarse_dx, coarse_dy = pyramid_align(small_image, small_fixed, threshold)

        scaled_dx = coarse_dx * 2
        scaled_dy = coarse_dy * 2

        window = 2
        m = int(0.1 * min(image.shape))
        cropped_fixed = image_fixed[m:-m, m:-m]

        score = float('inf')
        best_dx, best_dy = scaled_dx, scaled_dy
        for dx in range(scaled_dx - window, scaled_dx + window + 1):
            for dy in range(scaled_dy - window, scaled_dy + window + 1):
                shifted_image = np.roll(image, dx, axis=1)
                shifted_image = np.roll(shifted_image, dy, axis=0)
                cropped_image = shifted_image[m:-m, m:-m]
                new_score = alignment_metric("SSD", cropped_image, cropped_fixed)
                if new_score < score:
                    score = new_score
                    best_dx = dx
                    best_dy = dy
        return best_dx, best_dy


# ---- run pyramid alignment on the full-resolution tifs ----
for imname in tif_filenames:
    im = skio.imread(imname)
    im = sk.img_as_float(im)
    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]

    g_dx, g_dy = pyramid_align(g, b)
    r_dx, r_dy = pyramid_align(r, b)
    ag = np.roll(np.roll(g, g_dx, axis=1), g_dy, axis=0)
    ar = np.roll(np.roll(r, r_dx, axis=1), r_dy, axis=0)
    print(imname, "G:", (g_dx, g_dy), "R:", (r_dx, r_dy))

    im_out = np.dstack([ar, ag, b])
    base = imname.split('/')[-1].rsplit('.', 1)[0]
    fname = os.path.join(OUT, f"out_{base}.jpg")
    im_out = (np.clip(im_out, 0, 1) * 255).round().astype(np.uint8)
    skio.imsave(fname, im_out)


# ---- run single-scale alignment on the low-res jpgs ----
for imname in jpg_filenames:
    im = skio.imread(imname)
    im = sk.img_as_float(im)

    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]

    ag, _ = align(g, b)
    ar, _ = align(r, b)

    im_out = np.dstack([ar, ag, b])
    fname = os.path.join(OUT, f"out_{os.path.basename(imname)}")
    im_out = (np.clip(im_out, 0, 1) * 255).round().astype(np.uint8)
    skio.imsave(fname, im_out)
