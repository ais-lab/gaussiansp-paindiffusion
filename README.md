# Driving GaussianAvatars to Express Pain

It's fast, high-quality rendering, with teeth...

Based on:
- PainDiffusion for pain expression
- GaussianAvatars
    - VHAP for face model tracking
    - GaussianAvatars for animation

---

## Install

To keep the installation simple, each component will be installed in a different environment.

1. Clone this repo.
2. GaussianAvatars (https://github.com/ShenhanQian/GaussianAvatars/tree/main)
     1. Clone the repo.
     2. Install its environment.
3. VHAP (https://github.com/ShenhanQian/VHAP)
     1. Clone the repo.
     2. Install its environment.
4. PainDiffusion (https://github.com/ais-lab/paindiffusion)
     1. Clone the repo.
     2. Install its environment.
     
Copy the following files to their corresponding folders: `controll_gui_with_gaussian_avatars.py` -> paindiffusion, `localviewer.py` -> gaussianavatars.

Update the path to config file in each file.

## Use

1. Change the flame model of CHAP and GaussianAvatars at (gaussianAvatars/flame_model/flame.py) and (vhap/model/flame.py) to flame2020, which should be `generic_model.pkl` from https://flame.is.tue.mpg.de/login.php.

2. Use VHAP to track a video or multiview video using the monocular steps and merge the output with the final step in nersemble (check vhap/doc).

3. After building the point cloud and flame sequence, use local_viewer.py in GaussianAvatars to view the results with the path to the output of the previous step.

4. Run PainDiffusion by `python controll_gui_with_gaussian_avatars.py` to drive the avatar according to pain stimuli.
