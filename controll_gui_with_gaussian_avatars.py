import io
import socket
import gradio as gr
import threading
import time
import asyncio
from lightning import Trainer
import numpy as np
from collections import deque

import torch
import yaml
from diffusion.elucidated_for_video import ElucidatedDiffusion

from diffusion.module.utils.biovid import BioVidDM, savitzky_golay, bilateral_filter
# from inferno_package.render_from_exp import decode_latent_to_image

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

model = load_model("configure/scale_jawpose_window_32.yml")
default_face = 'default_face/'

# Initialize shared variables
current_stimuli = {
    'pain_stimuli': 30,
    'pain_configuration': 5,
    'emotion_status': 5,
    'scripted_pain_stimuli': None
}


frame_queue = deque()
stimuli_queue = deque(maxlen=32)  # Fixed size queue

past_frames = None
current_frame = None
stop_threads_flag = False

scheduling_matrix = None

sr = 25  # Sampling rate in Hz
generate_fps = 25  # Frame generation rate in Hz
window_size = 32
smooth_window = deque(maxlen=2)


host = "localhost"
port = 5000

def stimuli_sampling_loop():
    global stop_threads_flag
    while not stop_threads_flag:
        # Sample current stimuli values
        stimuli_sample = current_stimuli.copy()
        # print(f"Stimuli sampling loop: {stimuli_sample['pain_stimuli']}, {stimuli_sample['pain_configuration']}, {stimuli_sample['emotion_status']}")
        # Append to stimuli_queue
        stimuli_queue.append(stimuli_sample)
        # Sleep for sampling interval
        time.sleep(1.0 / sr)

def model_loop():
    global stop_threads_flag
    
    ramdisk_path = "/dev/shm/frames_paindiffusion.pt"
    
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
        
        torch.save((smoothed_frames[frames.shape[0]:], time.time(), target_interval), ramdisk_path)
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
                  

# Global variable to keep track of the decay thread
decay_thread = None

def update_pain_stimuli(pain_stimuli):
    current_stimuli['pain_stimuli'] = pain_stimuli

async def decay_pain_stimuli():
    start_value = current_stimuli['pain_stimuli']
    original_value = 30
    duration = 5  # Duration in seconds
    steps = 50
    step_delay = duration / steps
    step_value = (start_value - original_value) / steps

    for _ in range(steps):
        await asyncio.sleep(step_delay)
        start_value -= step_value
        if start_value < original_value:
            start_value = original_value
        current_stimuli['pain_stimuli'] = start_value
        yield start_value  # Update the slider in the UI

def update_other_stimuli(pain_configuration, emotion_status):
    current_stimuli['pain_configuration'] = pain_configuration
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
    current_stimuli['emotion_status'] = emotion_map[emotion_status]

# Start threads
stimuli_thread = threading.Thread(target=stimuli_sampling_loop)
model_thread = threading.Thread(target=model_loop)
stimuli_thread.start()
model_thread.start()

with gr.Blocks() as demo:
    gr.HTML('''
    <h1 class="title is-1 publication-title">PainDiffusion: Can robot express pain?</h1>
    ''')

    with gr.Row():
        pain_stimuli_slider = gr.Slider(30, 60, value=30, label="Heat Stimuli", step=0.1)
        pain_configuration_slider = gr.Slider(5, 11, value=5, label="Pain Configuration", step=0.1)
        emotion_status_radio = gr.Radio(
            choices=[
                "Anger", "Contempt", "Disgust", "Fear",
                "Happiness", "Neutral", "Sadness", "Surprise"
            ],
            value="Neutral",
            label="Emotion Status"
        )

    # Update pain_stimuli in real-time as the slider moves
    pain_stimuli_slider.input(
        fn=update_pain_stimuli,
        inputs=pain_stimuli_slider,
        outputs=None
    )

    # Start decay when the slider is released
    pain_stimuli_slider.release(
        fn=decay_pain_stimuli,
        inputs=None,
        outputs=pain_stimuli_slider  # Update the slider value in the UI
    )

    # Update other stimuli when their sliders change
    pain_configuration_slider.change(
        fn=update_other_stimuli,
        inputs=[pain_configuration_slider, emotion_status_radio],
        outputs=None
    )
    emotion_status_radio.change(
        fn=update_other_stimuli,
        inputs=[pain_configuration_slider, emotion_status_radio],
        outputs=None
    )

demo.launch()
