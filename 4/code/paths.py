import os

HERE = os.path.dirname(os.path.abspath(__file__))
# images/ sits next to code/ in the website repo, and inside code/ in the submission zip
ROOT = HERE if os.path.isdir(os.path.join(HERE, "images")) else os.path.join(HERE, "..")
MINE = os.path.join(ROOT, "images", "mine")
RESULTS = os.path.join(ROOT, "images", "results")
POINTS = os.path.join(ROOT, "points")  # correspondences clicked with pick_points.py
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(POINTS, exist_ok=True)
