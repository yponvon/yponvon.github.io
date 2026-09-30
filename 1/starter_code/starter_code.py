# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

# name of the input file
jpg_filenames = [
    'CS180_fa2026_proj1_data/cathedral.jpg',
    'CS180_fa2026_proj1_data/monastery.jpg',
    'CS180_fa2026_proj1_data/tobolsk.jpg',
    'CS180_fa2026_proj1_data/ornament.jpg',
    'CS180_fa2026_proj1_data/specimens.jpg',
    'CS180_fa2026_proj1_data/iconostasis.jpg',
]
tif_filenames = [
    'CS180_fa2026_proj1_data/church.tif',
    'CS180_fa2026_proj1_data/emir.tif',
    'CS180_fa2026_proj1_data/harvesters.tif',
    'CS180_fa2026_proj1_data/icon.tif',
    'CS180_fa2026_proj1_data/ilemselga.tif',
    'CS180_fa2026_proj1_data/melons.tif',
    'CS180_fa2026_proj1_data/religous_painting.tif',
    'CS180_fa2026_proj1_data/self_portrait.tif',
    'CS180_fa2026_proj1_data/siren.tif',
    'CS180_fa2026_proj1_data/three_generations.tif',
    'CS180_fa2026_proj1_data/wharf.tif',
]

def alignment_metric(name, image1, image2):
        if name == "SSD":
            difference = image1 - image2 
            sq_difference = np.square(difference)
            final = np.sum(sq_difference)
            return final
        if name == "NCC":
            # measure similarity of direction
            mean_1 = np.mean(image1)
            mean_2 = np.mean(image2)
            norm_1 = np.linalg.norm(image1)
            norm_2 = np.linalg.norm(image2)
            normalize_1 = (image1 - mean_1) / norm_1
            normalize_2 = (image2 - mean_2) / norm_2
            similarity = np.dot(normalize_1.flatten(), normalize_2.flatten())
            return similarity

def align(image, image_fixed):
        score = float('inf') 
        best_dx = None
        best_dy = None
        m = 15

        for dx in range(-15, 16):
            for dy in range(-15, 16):
                #apply candidate shift
                shifted_image = np.roll(image, dx, axis=1)
                shifted_image = np.roll(shifted_image, dy, axis=0)

                cropped_image = shifted_image[m:-m, m:-m]
                cropped_fixed = image_fixed[m:-m, m:-m]
                new_score = alignment_metric("SSD", cropped_image, cropped_fixed)
                #score = max(alignment_metric("NCC", cropped_image, cropped_fixed), score)
                if new_score < score:
                    score = new_score
                    best_dx = dx
                    best_dy = dy
        best_image = np.roll(image, best_dx, axis=1)
        best_image = np.roll(best_image, best_dy, axis=0)
        print(best_dx,best_dy)
        return best_image, (best_dx,best_dy)

print(skio.imread('CS180_fa2026_proj1_data/church.tif').shape) #(9607, 3634)

def blur_axis(image, axis):
    left = np.roll(image, 1, axis=axis)
    right = np.roll(image, -1, axis=axis)
    return (left + 2 * image + right) / 4.0

def downsample(image):
    blurred = blur_axis(image, axis=0)
    blurred = blur_axis(blurred, axis=1)
    return blurred[::2, ::2]

def pyramid_align(image, image_fixed, threshold=400):
    if min(image.shape) < threshold:
        _,(best_dx, best_dy) = align(image, image_fixed)
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
     

for imname in tif_filenames:
    im = skio.imread(imname)
    im = sk.img_as_float(im)
    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    g_dx, g_dy = pyramid_align(g, b)
    r_dx, r_dy = pyramid_align(r, b)
    ag = np.roll(np.roll(g, g_dx, axis=1), g_dy, axis=0)
    ar = np.roll(np.roll(r, r_dx, axis=1), r_dy, axis=0)
    print(imname, "G:", (g_dx, g_dy), "R:", (r_dx, r_dy))

    im_out = np.dstack([ar, ag, b])
    base = imname.split('/')[-1].rsplit('.', 1)[0]
    fname = f"out_path/out_{base}.jpg"
    im_out = (np.clip(im_out, 0, 1) * 255).round().astype(np.uint8)
    skio.imsave(fname, im_out)
    skio.imshow(im_out)
    skio.show()


# read in the image
for imname in jpg_filenames:
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    ag, _ = align(g, b)
    ar, _ = align(r, b)
            
    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = f"out_path/out_{imname.split('/')[-1]}"
    print(im.dtype, im.shape) #float64 (1024, 390)
    print(im_out.dtype, im_out.shape) #float64 (341, 390, 3)
    print(im_out.min(), im_out.max()) #0.0 1.0

    im_out = (np.clip(im_out, 0, 1) * 255).round().astype(np.uint8) # convert to uint8 and values 0-255
    skio.imsave(fname, im_out)

    # display the image
    skio.imshow(im_out)
    skio.show()
