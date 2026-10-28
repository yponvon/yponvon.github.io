# click correspondences and save them to points/<name>.json
#   pair:          python pick_points.py left.jpg mid.jpg left_mid
#   rectification: python pick_points.py tiles.jpg tiles   (4 corners: top-left, top-right, bottom-right, bottom-left)
# left click adds a point, right click (two finger click) removes the last one on that image.
# points are saved after every click, enter or closing the window finishes.
# click the points in the same order on both images. running it again on a name keeps the old points.
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from warp import load

from paths import POINTS

files, name = sys.argv[1:-1], sys.argv[-1]
out_path = os.path.join(POINTS, name + ".json")
images = [load(f) for f in files]
pts = [[] for _ in files]
if os.path.exists(out_path):
    with open(out_path) as f:
        old = json.load(f)
    pts = [old.get("pts%d" % (i + 1), []) for i in range(len(files))]

fig, axes = plt.subplots(1, len(files), figsize=(8 * len(files), 7))
axes = list(np.atleast_1d(axes))
for ax, im, f in zip(axes, images, files):
    ax.imshow(im[:, :, ::-1])
    ax.set_title(f)
    ax.axis("off")
drawn = []


def redraw():
    for d in drawn:
        d.remove()
    drawn.clear()
    for ax, p in zip(axes, pts):
        for i, (x, y) in enumerate(p):
            drawn.append(ax.plot(x, y, "r+", markersize=12, mew=2)[0])
            drawn.append(ax.text(x + 8, y - 8, str(i + 1), color="yellow", fontsize=11, weight="bold"))
    fig.suptitle(" | ".join("%s: %d pts" % (f, len(p)) for f, p in zip(files, pts)))
    fig.canvas.draw_idle()


def save_points():
    n = min(map(len, pts)) if len(files) == 2 else len(pts[0])
    d = {"images": files}
    for i, p in enumerate(pts):
        d["pts%d" % (i + 1)] = p[:n]  # only finished pairs
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1)
    return n


def on_key(event):
    if event.key == "enter":
        plt.close(fig)


def on_click(event):
    # ignore clicks while zooming/panning with the toolbar
    if event.inaxes not in axes or fig.canvas.toolbar.mode:
        return
    i = axes.index(event.inaxes)
    if event.button == 1:
        pts[i].append([float(event.xdata), float(event.ydata)])
    elif event.button == 3 and pts[i]:
        pts[i].pop()
    redraw()
    save_points()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
redraw()
plt.show()

if len(files) == 2 and len(pts[0]) != len(pts[1]):
    print("WARNING: %d points on %s but %d on %s, only the first %d pairs are kept"
          % (len(pts[0]), files[0], len(pts[1]), files[1], min(map(len, pts))))
print("saved %d points to %s" % (save_points(), out_path))
