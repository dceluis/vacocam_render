# Vision-Assisted Camera Orientation / VacoCam

## Installation
```
git clone https://github.com/dceluis/vacocam_render
cd vacocam_render
pip install -r requirements.txt
```

## Usage
Rendering a supervised video is a multi-step process.

First, we detect the balls on each frame:
```
python detect.py --model="./path/to/yolov8.pt" "./path/to/video.mp4"
```

We can now run the two-step supervision strategies.

1. Removing static clusters:
```
python track.py --tracking="declustered" "./path/to/video.mp4"
```

2. Supervise the video focus using gpt4-vision:
```
python track.py --tracking="vacocam" "./path/to/video.mp4"
```

Finally, we can render the video:
```
python render.py --max-zoom=1.9 --min-zoom=1.2 --max-area=400 --min-area=50 --vacocam "./path/to/video.mp4"
```

## Technical Overview
![VacocamOverviewDark](https://github.com/dceluis/vacocam_render/assets/5464881/90f64cd2-76be-4338-aa31-c119d11486ad)
