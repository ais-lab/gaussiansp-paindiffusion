This is the repo for gaussian splatting avatars with driving model from paindiffusion.

It's fast, high quality rendering, have teeth...

Based on:
- Paindiffusion 
- GaussianAvatars
  - CHAP for face model tracking
  - GaussianAvatars for animation

---

## Install

To keep the installation simple, each component will be installed with different environment.

1. Clone this repo
2. GaussianAvattars
   1. Clone the repo
   2. Install its environment
3. VHAP
   1. Clone the repo
   2. Install its environment
4. PainDiffusion
   1. Clone the repo
   2. Install its environment
   
copy following files to its corresponding folders `controll_gui_with_gaussian_avatars.py` -> paindiffusion, `localviewer.py` -> gaussianavatars.

## Use

1. Change flame model of CHAP and GaussianAvatars at (gaussianAvatars/flame_model/flame.py) and (vhap/model/flame.py) to flame2020, should be `generic_model.pkl` from https://flame.is.tue.mpg.de/login.php.

2. Use CHAP to track a video or multiview video use the monocular steps and merge the output with the final step in nersemble. (check chap/doc)

3. After build the point cloud and flame sequence, use local_viewer.py in GaussianAvatars to view the results with the parth to the output of the previous step.

4. Run paindiffusion to drive the avatar according to pain stimuli.
