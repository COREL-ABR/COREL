#!/usr/bin/env python3
import os
from os import path
from utils.utils import run_shell_cmd

###### config #########
# set it to False if you want to execute the script (may take some time)
PRINT_COMMANDS = False
# change these parameters as you like
VIDEO_NAME = 'BMPCC4K_v4min'
INPUT_VIDEO_PATH = path.join('input-videos', f'{VIDEO_NAME}.mp4')
OUTPUT_FOLDER = path.join('output-videos')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_DASH_PKG_FOLDER = path.join(OUTPUT_FOLDER, 'bmpcc4k')
segment_size = 2  # chunk length (in secs)
target_fps = 30
 # [width, height, target bitrate, id]
dash_configs = [
    [7680, 4320, 152400000, '8k'],
    [3840, 2160, 32800000, '4k'],
    [2560, 1440, 11700000, '2k'],
    [1920, 1080, 8200000, '1080p'],
    [1280, 720, 4900000, '720p'],
    [640, 360, 1500000, '360p']
]

video_quality_renditions = [f"{OUTPUT_FOLDER}/{VIDEO_NAME}_intermed_{config[2]}.mp4#video:id={config[3]}"
                            for config in dash_configs]

## create directories if not there
os.makedirs(OUTPUT_DASH_PKG_FOLDER, exist_ok=True)

mp4box_bin = "MP4Box "
ffmpeg_bin = "ffmpeg "
overwrite_output_files = "-y "

## get intermediate representations
for config in dash_configs:
    width = config[0]
    height = config[1]
    target_bitrate = config[2]

    input_file = f"-i {INPUT_VIDEO_PATH} "
    fps = f"-r {target_fps} "
    h264_encoding_options = f"-x264opts 'keyint={target_fps * segment_size}:" \
                            f"min-keyint={target_fps * segment_size}:no-scenecut' "
    resolution = f"-vf scale={width}:{height} "
    bitrate = f"-b:v {target_bitrate} "
    buffer_maxrate = f"-maxrate {target_bitrate * 2} -bufsize {target_bitrate * 2} "
    enable_faststart = f"-movflags faststart "
    h264_profile = f"-profile:v main "
    encoder_preset = f"-preset fast "
    skip_audio = f"-an "
    inter_output_file = f"{OUTPUT_FOLDER}/{VIDEO_NAME}_intermed_{target_bitrate}.mp4"

    command = ffmpeg_bin + overwrite_output_files + input_file + fps + h264_encoding_options + \
              resolution + bitrate + buffer_maxrate + enable_faststart + h264_profile + \
              encoder_preset + skip_audio + inter_output_file
    if PRINT_COMMANDS:
        print(command)
    else:
        run_shell_cmd(command)

## dash packaging
segment = f"-dash {segment_size * 1000} "
fragment = f"-frag {segment_size * 1000} "
random_access_points = "-rap "
segment_name = f"-segment-name 'segment_$RepresentationID$_' "
dash_fps = f"-fps {target_fps} "
video_renditions = f"{' '.join(video_quality_renditions)} "
output_mpd_file = f"-out {OUTPUT_DASH_PKG_FOLDER}/{VIDEO_NAME}_playlist.mpd"

dash_command = mp4box_bin + segment + fragment + random_access_points + \
               segment_name + dash_fps + video_renditions + output_mpd_file
if PRINT_COMMANDS:
    print(dash_command)
else:
    run_shell_cmd(dash_command)

print('Complete./')
