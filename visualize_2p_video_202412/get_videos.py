# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = './ffmpeg/bin/ffmpeg.exe'
import tifffile

# generate video for dff traces.
def save_dff_video(args):
    window_size = 500
    colors = ['royalblue', 'crimson']
    labels = [str(x) for x in args.labels.strip('[]').split(',')]
    # initialize animation.
    def init():
        for line in lines:
            line.set_data([], [])
        vertical_line.set_xdata(0)
        time_text.set_text('')
        return lines + [vertical_line, time_text]
    # animation function.
    def update(frame):
        start_idx = max(0, frame - window_size // 2)
        end_idx = min(len(time), start_idx + window_size)
        for i, line in enumerate(lines):
            x_window = time[start_idx:end_idx] - time[frame]
            y_window = dff[i, start_idx:end_idx]
            line.set_data(x_window, y_window)
        vertical_line.set_xdata(0)
        time_text.set_text(f't = {time[frame]:.2f} s')
        return lines + [vertical_line, time_text]
    # prepare data.
    dff = np.load(os.path.join('./', 'results', 'temp_data', 'dff.npy'), allow_pickle=True)
    time = np.load(os.path.join('./', 'results', 'temp_data', 'time_img.npy'), allow_pickle=True)/1000
    roi_labels = np.load(os.path.join('./', 'results', 'temp_data', 'roi_labels.npy'), allow_pickle=True)
    for i in range(dff.shape[0]):
        dff[i,:] = (dff[i,:] - np.nanmin(dff[i,:])) / (np.nanmax(dff[i,:]) - np.nanmin(dff[i,:]) + 1e-5)
        dff[i,:] += i
    # set canvas.
    fig, ax = plt.subplots(1, 1, figsize=(4, 10), layout='tight')
    fig.patch.set_facecolor('black')
    lines = [ax.plot([], [], lw=2, color=colors[roi_labels[i]])[0] for i in range(dff.shape[0])]
    vertical_line = ax.axvline(x=0, color='white', linestyle='--')
    time_text = ax.text(0.8, 0.9, '', transform=ax.transAxes, color='white', fontsize=12)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.tick_params(axis='y', tick1On=False)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xlabel('time (ms)', color='white')
    handles = [
        ax.plot([], lw=0, color=colors[i], label=labels[i])[0] 
        for i in range(len(labels))
        if labels[i] != None]
    ax.legend(
        loc='upper right',
        handles=handles,
        labelcolor='linecolor',
        frameon=False,
        framealpha=0)
    # set anime.
    ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True)
    ani.save(os.path.join('./results', 'traces.mp4'), fps=30, dpi=300)

# automatical adjustment of contrast.
def adjust_video_contrast(video, lower_percentile=25, upper_percentile=99):
    video = video.astype('float32')
    lower = np.percentile(video, lower_percentile)
    upper = np.percentile(video, upper_percentile)
    video = np.clip((video - lower) * 255 / (upper - lower), 0, 255)
    video = video.astype('int32')
    return video

# output denoised fov video.
def save_fov_video():
    fps = 30
    # get video file list.
    filename = [f for f in os.listdir(os.path.join('./results', 'temp_denoised')) if 'tif' in f]
    filename.sort()
    videos = [tifffile.imread(os.path.join('./results', 'temp_denoised', f)) for f in filename]
    videos = [adjust_video_contrast(v) for v in videos]
    video_writer = cv2.VideoWriter(
        os.path.join('./results', 'fov.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), fps,
        (videos[0].shape[1], videos[0].shape[2]), isColor=True)
    for i in range(videos[0].shape[0]):
        if len(videos) == 1:
            frame = cv2.normalize(videos[0][i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            frame = cv2.merge((frame*0, frame, frame*0))
        if len(videos) == 2:
            frame_r = cv2.normalize(videos[0][i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            frame_g = cv2.normalize(videos[1][i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            frame = cv2.merge((frame_g*0, frame_g, frame_r))
        video_writer.write(frame)
    video_writer.release()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='If Farzaneh wants to kill you ask Yicong for a hug!')
    parser.add_argument('--labels', required=True, type=str, help='The frame around the output period.')
    args = parser.parse_args()

    save_fov_video()
    save_dff_video(args)