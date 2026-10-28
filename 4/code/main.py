# runs every part of project 4A in order; all figures are written to images/results/
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))

for script in ["part2_homography.py",  # A.2 correspondences + H
               "part3_rectify.py",     # A.3 rectification, nearest vs bilinear
               "part4_mosaic.py"]:     # A.4 mosaics
    print("==== running", script)
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")
