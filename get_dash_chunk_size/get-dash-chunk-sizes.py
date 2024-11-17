#!/usr/bin/env python3

import os
import glob
from os import path

# [width, height, target bitrate, id]
dash_configs = [ # should be same as the one in prepare-dash-video script
    [7680, 4320, 152400000, '8k'],
    [3840, 2160, 32800000, '4k'],
    [2560, 1440, 11700000, '2k'],
    [1920, 1080, 8200000, '1080p'],
    [1280, 720, 4900000, '720p'],
    [640, 360, 1500000, '360p']
]

# assume videos are inside $VIDEO_PATH and format is segment_[id]_[chunk].m4s
BITRATE_LEVELS = dash_configs.__len__()
# change these value according to the video
TOTAL_VIDEO_CHUNkS = 121  
VIDEO_PATH = path.join('output-videos', 'bmpcc4k')

chunk_size_list = []
for idx, config in enumerate(dash_configs):
    files = glob.glob(path.join(VIDEO_PATH, f"segment_{config[3]}_*.m4s"))
    assert files.__len__() == TOTAL_VIDEO_CHUNkS
    chunk_sizes = []
    for chunk_id in range(1, TOTAL_VIDEO_CHUNkS + 1):
        file_path = path.join(VIDEO_PATH, f"segment_{config[3]}_{chunk_id}.m4s")
        chunk_size = os.path.getsize(file_path)
        chunk_sizes.append(chunk_size)
    print(f"[{config[3]}] = ", chunk_sizes)
    chunk_size_list.append(chunk_sizes)
