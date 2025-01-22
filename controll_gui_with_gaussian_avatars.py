import threading
import time
from lightning import Trainer
import numpy as np
from collections import deque

import torch
import yaml
from diffusion.elucidated_for_video import ElucidatedDiffusion

from diffusion.module.utils.biovid import BioVidDM, bilateral_filter
# from inferno_package.render_from_exp import decode_latent_to_image

import serial
from dataclasses import dataclass
import tyro

emotion_map = {
    "Anger": 0,
    "Contempt": 1,
    "Disgust": 2,
    "Fear": 3,
    "Happiness": 4,
    "Neutral": 5,
    "Sadness": 6,
    "Surprise": 7
}

def load_model(conf_file) -> ElucidatedDiffusion:

    with open(conf_file, "r") as f:
        conf = yaml.safe_load(f)

    best_checkpoint = conf["BEST_CKPT"]
    
    model = ElucidatedDiffusion.from_conf(conf_file)

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        fast_dev_run=1,
        logger=False,
    )

    biovid = BioVidDM.from_conf(conf_file)
    trainer.test(model, datamodule=biovid, ckpt_path=best_checkpoint)

    model = model.eval()
    model = model.cuda()
    
    return model

def stimuli_sampling_loop():
    
    global stop_threads_flag
    global external_mode
    global is_arduino_avai
    
    while not stop_threads_flag:
        
        # read config value from UI
        try:
            config = torch.load(ramdisk_path_from_ui_to_paindiff)
        except Exception as e:
            print('read error')
            continue
        emotion_status = emotion_map[config['emotion_status']]
        pain_configuration = config['pain_configuration']
        external_mode = config['external_mode']
        
        # print(f"External mode: {external_mode}")
        # print(f"Emotion status: {emotion_status}")
        # print(f"Pain configuration: {pain_configuration}")
        # print(f"Arduino available: {is_arduino_avai}")
        
        # read value from arduino
        if is_arduino_avai and external_mode:
            arduino_val = arduino.readline().decode('utf-8').strip()
            if not arduino_val:
                continue        
            heat_stimuli = float(arduino_val)
        else:
            heat_stimuli = config['pain_stimuli']
            
        # print(f"Heat stimuli: {heat_stimuli}")
            
        stimuli_sample = {
            'pain_stimuli': heat_stimuli,
            'pain_configuration': pain_configuration,
            'emotion_status': emotion_status,
            'scripted_pain_stimuli': None
        }
        
        # Append to stimuli_queue
        stimuli_queue.append(stimuli_sample)
        # Sleep for sampling interval
        time.sleep(1.0 / sr)

def model_loop():
    global stop_threads_flag
        
    target_interval = 1.0 / generate_fps
    
    while not stop_threads_flag:
        stimuli_values = list(stimuli_queue)

        start = time.time()
        frames = generate_frames(stimuli_values)
        
        if frames is None:
            continue
        
        smooth_window.append(frames.cpu().numpy())
        
        smoothed_frames = np.concatenate(smooth_window, axis=0)
        
        # print(f"Smoothed frames shape: {smoothed_frames.shape}")
        
        # smoothed_frames = savitzky_golay(smoothed_frames, 5, 2)
        
        smoothed_frames = bilateral_filter(smoothed_frames)
        
        smoothed_frames = torch.tensor(smoothed_frames).float()
        
        # print(f"Smoothed frames shape after savitzky_golay: {smoothed_frames.shape}")
        
        prediction_time = time.time() - start 
        
        frame_interval = prediction_time / len(frames)
        
        # cap the prediction rate to 30fps by sleeping
        if frame_interval < target_interval:
            time.sleep((target_interval - frame_interval)*len(frames))
            frame_interval = target_interval
            
        print(f"Prediction time: {prediction_time}, Frame interval: {frame_interval}")
        
        torch.save((smoothed_frames[frames.shape[0]:], time.time(), target_interval), ramdisk_path_from_paindiff_to_ui)
        # torch.save((frames, time.time(), target_interval), ramdisk_path)

def generate_frames(stimuli_values):
    
    # construct ctrl tensor
    
    if len(stimuli_values) < window_size:
        return None
    
    emotion_list = [stimuli['emotion_status'] for stimuli in stimuli_values]
    
    pain_config = [stimuli['pain_configuration'] for stimuli in stimuli_values]
    
    pain_stimuli_list = [stimuli['pain_stimuli'] for stimuli in stimuli_values]
    
    [pain_stimuli_list, pain_config, emotion_list] = [torch.tensor(x).float().unsqueeze(0) for x in [pain_stimuli_list, pain_config, emotion_list]]
    
    [pain_stimuli_list, pain_config, emotion_list] = [x.cuda() for x in [pain_stimuli_list, pain_config, emotion_list]]
    
    ctrl = [pain_stimuli_list, pain_config, emotion_list]
        
    # define guide
    
    guide = [0.25, 0.5, 1.0]
    
    global past_frames
    
    prediction_tensor = model.sample_a_chunk(ctrl, guide, past_frames)
    
    past_frames = prediction_tensor.detach().clone()
    
    prediction_tensor = prediction_tensor.squeeze(0)
    return prediction_tensor
                  

@dataclass
class Config:
    conf_file: str = "/home/tien/paindiffusion_gaussian_avatars/paindiffusion/configure/scale_jawpose_window_32.yml"
    arduino_port: str = "/dev/ttyACM0"
    arduino_baudrate: int = 9600
    ramdisk_path_from_paindiff_to_ui: str = "/dev/shm/frames_paindiffusion.pt"
    ramdisk_path_from_ui_to_paindiff: str = "/dev/shm/config_paindiffusion.pt"
    sr: int = 25  # Sampling rate in Hz
    generate_fps: int = 25  # Frame generation rate in Hz
    window_size: int = 32
    smooth_window_size: int = 2

if __name__ == "__main__":
    config = tyro.cli(Config)

    model = load_model(config.conf_file)
    default_face = 'default_face/'
    
    external_mode = True
    is_arduino_avai = True
    try:
        arduino = serial.Serial(config.arduino_port, config.arduino_baudrate)
    except Exception as e:
        print(f"Error: {e}")
        is_arduino_avai = False
    
    ramdisk_path_from_paindiff_to_ui = config.ramdisk_path_from_paindiff_to_ui
    ramdisk_path_from_ui_to_paindiff = config.ramdisk_path_from_ui_to_paindiff

    frame_queue = deque()
    stimuli_queue = deque(maxlen=config.window_size)  # Fixed size queue

    past_frames = None
    stop_threads_flag = False

    sr = config.sr
    generate_fps = config.generate_fps
    window_size = config.window_size
    smooth_window = deque(maxlen=config.smooth_window_size)

    # Start threads
    stimuli_thread = threading.Thread(target=stimuli_sampling_loop)
    model_thread = threading.Thread(target=model_loop)
    stimuli_thread.start()
    model_thread.start()
