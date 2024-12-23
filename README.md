# Driving GaussianAvatars to Express Pain

It's fast, high-quality rendering, with teeth...

[gaussianavatars_with_paindiffusion.webm](https://github.com/user-attachments/assets/c255054a-a31b-47dd-b7d3-8e1970086962)

Based on:
- PainDiffusion for pain expression
- GaussianAvatars
    - VHAP for face model tracking
    - GaussianAvatars for animation

---

## Install

To keep the installation simple, each component will be installed in a different environment.

Clone this repository use recursive flag to pull all submodules at once.
`git clone --recursive https://github.com/ais-lab/gaussiansp-paindiffusion`

For each component, follow their own readme to install their environment.
     
Copy the following files to their corresponding folders: `controll_gui_with_gaussian_avatars.py` -> paindiffusion, `localviewer.py` -> gaussianavatars. This two files communicate through reading and writing to a share-memory file.

## Use

1. Change the flame model of CHAP and GaussianAvatars at (gaussianAvatars/flame_model/flame.py) and (vhap/model/flame.py) to flame2020, which should be `generic_model.pkl` from https://flame.is.tue.mpg.de/login.php. Because PainDiffusion use flame2020.

2. Use VHAP to track a video or multiview video using the monocular steps and merge the output with the final step in nersemble (check vhap/doc). 

3. After building the point cloud and flame sequence in step 2, train a gaussian splating of the point cloud, please follow the step of GaussianAvatars. 

4. In GaussianAvatars dir and using its env, use local_viewer.py of in GaussianAvatars to view the results with the path to the output of the previous step.
```bash
python local_viewer.py --point_path '/path/to/point_cloud.ply' --driving_mode
```
6. Run PainDiffusion by `python controll_gui_with_gaussian_avatars.py` to drive the avatar according to pain stimuli.
