# runs every part of project 2 in order; all figures are written to images/results/
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))

for script in ["part1.py",          # 1.1 convolution, 1.2 finite differences, 1.3 DoG
               "bells_whistles.py", # part 1 b&w: gradient orientation
               "part2_sharpen.py",  # 2.1 unsharp mask
               "part2_hybrid.py",   # 2.2 hybrid images (+ b&w colour)
               "part2_blend.py"]:   # 2.3 stacks, 2.4 blending (+ b&w colour)
    print("==== running", script)
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")
