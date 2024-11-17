# Compress video Chunks in MPEG Dash format

## Install Video Encoding Tools
Video encoding tools: `ffmpeg` and `MP4Box` to compress a raw video in MPEG Dash format. 

To install the `ffmpeg` on Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

To install `MP4Box` on Ubuntu:

```bash
sudo apt-get install build-essential pkg-config git zlib1g-dev
```

```bash
git clone https://github.com/gpac/gpac.git gpac_public && cd gpac_public
```

```bash
./configure --static-bin
```

```bash
make && sudo make install
```

## Compress video chunks

1. Place the input video in [input-videos](input-videos) folder. Make sure the input video quality is better than the highest Dash bitrate to achieve target bitrate.

2. Edit the configurations in [prepare-dash-video.py](prepare-dash-video.py) script:

    a) Edit input video file name

    b) Edit chunk length if required

    c) Edit video encodings for Dash bitrate levels

3. Run the command to process generate Dash MPD file and encodings ``python3 prepare-dash-video.py``. The generated MPD file and videom encodings are placed in [output-videos](output-videos) folder.

## Output chunk size for each bitrate level

1. Change configurations inside [get-dash-chunk-sizes.py](get-dash-chunk-sizes.py) to match the processed video
2. Run the command ``python3 get-dash-chunk-sizes.py`` to get chunk size for all chunks at each bitrate level
