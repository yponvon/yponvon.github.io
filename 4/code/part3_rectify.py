# A.3: rectify flat rectangles, nearest neighbor vs bilinear
import os
import time
import numpy as np
import cv2
from warp import load, save, load_points, computeH, warpImageNearestNeighbor, warpImageBilinear

from paths import POINTS

# point file (4 corners from pick_points.py) -> width / height of the real object
RECTIFY = [
    ("rect_map", 0.75),        # campus map sign (measured 0.746 from the photo, so an 18x24 sign)
    ("rect_box", 1 / 3),       # hand cream box front (measured 0.319 from the photo)
]
MARGIN = 0.3  # how much around the rectangle to keep, as a fraction of its size

for name, aspect in RECTIFY:
    if not os.path.exists(os.path.join(POINTS, name + ".json")):
        print("SKIPPING %s: click its 4 corners with pick_points.py first" % name)
        continue
    files, corners, _ = load_points(name)
    im = load(files[0])

    # rectangle about as wide as the clicked top/bottom edges
    W = (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])) / 2
    Hh = W / aspect
    # small things get blown up to ~600px so the page images aren't tiny
    up = max(1, 600 / max(W, Hh))
    W, Hh = W * up, Hh * up
    rect = np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]])
    H = computeH(corners, rect)
    print("==== %s H:\n%s" % (name, H))
    # only keep the rectangle plus a margin, the full warp blows up near the horizon
    box = (int(-MARGIN * W), int(-MARGIN * Hh), int((1 + MARGIN) * W), int((1 + MARGIN) * Hh))

    t = time.time(); nn, _ = warpImageNearestNeighbor(im, H, box); t_nn = time.time() - t
    t = time.time(); bil, _ = warpImageBilinear(im, H, box); t_bil = time.time() - t
    print("%s: nearest %.2fs, bilinear %.2fs (%dx%d output)" % (name, t_nn, t_bil, nn.shape[1], nn.shape[0]))

    marked = im.copy()
    cv2.polylines(marked, [np.round(corners).astype(np.int32)], True, (0, 0, 1), 3)
    save("a3_%s_input.jpg" % name, marked)
    save("a3_%s_nn.jpg" % name, nn)
    save("a3_%s_bilinear.jpg" % name, bil)

    # zoom into the middle to see the difference: nn is jaggy, bilinear is smoother but softer
    cy, cx = nn.shape[0] // 2, nn.shape[1] // 2
    r = min(60, cy, cx)
    zoom = [cv2.resize(x[cy - r:cy + r, cx - r:cx + r], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            for x in (nn, bil)]
    save("a3_%s_zoom.jpg" % name, np.hstack([zoom[0], np.ones((zoom[0].shape[0], 8, 3)), zoom[1]]))
