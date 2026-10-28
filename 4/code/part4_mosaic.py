# A.4: warp every photo into the reference photo's view and blend
import os
import numpy as np
from warp import (load, save, load_points, computeH, warp_all,
                  blend_overwrite, blend_average, blend_two_band)

from paths import MINE, POINTS

# name, photos left to right, point file for each photo (None for the reference photo in the middle)
# each point file maps a photo to its neighbour one step closer to the reference, e.g.
#   python pick_points.py terrace_1.jpg terrace_2.jpg terrace_1_2
MOSAICS = [
    ("anthro", ["anthro_left.jpg", "anthro_mid.jpg", "anthro_right.jpg"],
     ["anthro_left_mid", None, "anthro_right_mid"]),
    ("sunshade", ["sunshade_left.jpg", "sunshade_right.jpg"],
     ["sunshade_left_right", None]),
    ("terrace", ["terrace_%d.jpg" % i for i in range(1, 6)],
     ["terrace_1_2", "terrace_2_3", None, "terrace_4_3", "terrace_5_4"]),
]


def homographies(point_files):
    # the far photos don't overlap the reference, so chain: H(1 -> 3) = H(2 -> 3) @ H(1 -> 2)
    ref = point_files.index(None)
    Hs = [None] * len(point_files)
    Hs[ref] = np.eye(3)
    for i in list(range(ref - 1, -1, -1)) + list(range(ref + 1, len(point_files))):
        _, pts1, pts2 = load_points(point_files[i])
        neighbour = i + 1 if i < ref else i - 1
        Hs[i] = Hs[neighbour] @ computeH(pts1, pts2)
    return Hs


for name, files, point_files in MOSAICS:
    missing = [f for f in files if not os.path.exists(os.path.join(MINE, f))]
    missing += [p + ".json" for p in point_files if p and not os.path.exists(os.path.join(POINTS, p + ".json"))]
    if missing:
        print("SKIPPING %s: missing %s" % (name, ", ".join(missing)))
        continue
    images = [load(f) for f in files]
    Hs = homographies(point_files)

    warped, weights = warp_all(images, Hs)
    print("%s: %d photos, canvas %dx%d" % (name, len(files), warped[0].shape[1], warped[0].shape[0]))
    for i, im in enumerate(images):
        save("a4_%s_src%d.jpg" % (name, i + 1), im)
    over = blend_overwrite(warped, weights)
    avg = blend_average(warped, weights)
    two = blend_two_band(warped, weights)
    save("a4_%s_overwrite.jpg" % name, over)
    save("a4_%s_average.jpg" % name, avg)
    save("a4_%s_blend.jpg" % name, two)

    # zoom on the seam: where the photo on top changes (inside the overlap)
    best = np.argmax(np.stack(weights), axis=0)
    covered = sum(wt > 0 for wt in weights)
    seam = (covered >= 2) & (np.abs(np.diff(best, axis=1, prepend=best[:, :1])) > 0)
    ys, xs = np.nonzero(seam)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    r = 200
    cy = min(max(cy, r), over.shape[0] - r)
    cx = min(max(cx, r), over.shape[1] - r)
    crops = [x[cy - r:cy + r, cx - r:cx + r] for x in (over, avg, two)]
    gap = np.ones((2 * r, 8, 3))
    save("a4_%s_seam.jpg" % name, np.hstack([crops[0], gap, crops[1], gap, crops[2]]))
    # weights for the webpage, each photo in its own colour channel
    wmap = np.zeros(warped[0].shape)
    for i, wt in enumerate(weights):
        wmap[:, :, i % 3] += wt
    save("a4_%s_weights.jpg" % name, wmap)
