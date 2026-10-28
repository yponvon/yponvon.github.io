# A.2: for every pair in points/, show the correspondences, the system of equations and H
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from warp import load, load_points, system, computeH, apply_H

from paths import POINTS, RESULTS

np.set_printoptions(precision=4, suppress=True, linewidth=150)

pairs = []
for path in sorted(glob.glob(os.path.join(POINTS, "*.json"))):
    name = os.path.basename(path)[:-5]
    files, pts1, pts2 = load_points(name)
    if pts2 is not None:
        pairs.append((name, files, pts1, pts2))
if not pairs:
    print("SKIPPING: no pairs in %s yet, click some with pick_points.py" % POINTS)

summary = []
for name, files, pts1, pts2 in pairs:
    A, b = system(pts1, pts2)
    H = computeH(pts1, pts2)
    err = np.linalg.norm(apply_H(H, pts1) - pts2, axis=1)
    print("==== %s: %s -> %s, %d points" % (name, files[0], files[1], len(pts1)))
    print("A (first 2 points):\n", A[:4])
    print("b (first 2 points):", b[:4])
    print("H:\n", H)
    print("reprojection error: mean %.2f px, max %.2f px" % (err.mean(), err.max()))

    # how good is H on a point it hasn't seen? leave one out, with all the others vs only 4 of them
    loo_all, loo_4 = [], []
    for i in range(len(pts1)):
        rest = [j for j in range(len(pts1)) if j != i]
        H_all = computeH(pts1[rest], pts2[rest])
        H_4 = computeH(pts1[rest[:4]], pts2[rest[:4]])
        loo_all.append(np.linalg.norm(apply_H(H_all, pts1[i:i + 1]) - pts2[i]))
        loo_4.append(np.linalg.norm(apply_H(H_4, pts1[i:i + 1]) - pts2[i]))
    print("held out point error: %d points %.2f px, 4 points %.2f px (median)"
          % (len(pts1) - 1, np.median(loo_all), np.median(loo_4)))
    summary.append((name, len(pts1), err.mean(), np.median(loo_all), np.median(loo_4)))
    with open(os.path.join(RESULTS, "a2_%s_H.txt" % name), "w") as f:
        f.write("A =\n%s\n\nb = %s\n\nH =\n%s\n\nreprojection error: mean %.2f px, max %.2f px\n"
                % (A, b, H, err.mean(), err.max()))

    # red + = clicked, cyan o on the right = left points sent through H (should sit on the red ones)
    im1, im2 = load(files[0]), load(files[1])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, im, p, f in [(axes[0], im1, pts1, files[0]), (axes[1], im2, pts2, files[1])]:
        ax.imshow(np.clip(im, 0, 1)[:, :, ::-1])
        ax.plot(p[:, 0], p[:, 1], "r+", markersize=10, mew=2)
        for i, (x, y) in enumerate(p):
            ax.text(x + 8, y - 8, str(i + 1), color="yellow", fontsize=9, weight="bold")
        ax.set_title(f)
        ax.axis("off")
    mapped = apply_H(H, pts1)
    axes[1].plot(mapped[:, 0], mapped[:, 1], "co", mfc="none", markersize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "a2_%s_points.jpg" % name), dpi=80)
    plt.close()

if summary:
    with open(os.path.join(RESULTS, "a2_summary.txt"), "w") as f:
        f.write("pair, points, fit error (px), held out error n-1 points (px), held out error 4 points (px)\n")
        for row in summary:
            f.write("%s, %d, %.2f, %.2f, %.2f\n" % row)
    print(open(os.path.join(RESULTS, "a2_summary.txt")).read())
