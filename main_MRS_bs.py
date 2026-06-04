#!/usr/bin/env python3
"""
MRS (Multi-Robot System) B-spline Trajectory Visualization.

Interactive visualization of an MRS center trajectory (B-spline) and the
resulting robot trajectories.  Control points are draggable in the x-y plane;
parameters can be tuned via text boxes and a t_inp range axis.
"""

from mrs_bs_viz import MRSVisualizer

if __name__ == "__main__":
    vis = MRSVisualizer()
    vis.run()
