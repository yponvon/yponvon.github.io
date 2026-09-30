import os

HERE = os.path.dirname(os.path.abspath(__file__))
# images/ sits next to code/ in the website repo, and inside code/ in the submission zip
ROOT = HERE if os.path.isdir(os.path.join(HERE, "images")) else os.path.join(HERE, "..")
COURSE = os.path.join(ROOT, "images", "course")
MINE = os.path.join(ROOT, "images", "mine")
DOWNLOADED = os.path.join(ROOT, "images", "downloaded")
RESULTS = os.path.join(ROOT, "images", "results")
os.makedirs(RESULTS, exist_ok=True)
