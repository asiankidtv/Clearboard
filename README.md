# Clearboard: Visual Keystroke Inference Software

### Background
In the digital world, most methods for
collecting digital personal information
go through the internet. However,
physical methods of accumulating
data are still available, and may even
be more effective than going digitally,
where cryptography and cybersecurity
hinder data collection efforts. Many
people see the digital and physical
realm as two separate fields, unaware
of the dangers that can come from
coming both. 

Clearboard is a computer-vision prototype that tracks fingertip motion over a physical keyboard and attempts to infer typed text. It uses a camera feed, keyboard calibration, MediaPipe hand landmarks, and a YOLO keyboard detector.

## Setup

1. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add the following model files:

```text
(optional) src/best.pt
src/hand_landmarker.task
```

`best.pt` is the YOLO keyboard detection model and can either be replaced with an Oriented Bounding Box model found online or self-trained. `hand_landmarker.task` is the MediaPipe hand landmark model and can be found from Google.

4. Run the app:

```bash
python src/main.py
```

## Basic Use

- Choose the desired keyboardLayout from `keyboardLayouts.py`
- Click the four keyboard corners in this order: top-left, top-right, bottom-right, bottom-left.
- Press `c` to lock YOLO-detected keyboard corners if available.
- Press `q` to quit.
- A second window shows the typed text while the app is running.
    - On quit, it shows the corrected final message. Press any key to fully quit the program after.

## How It Works

The app first calibrates the keyboard by mapping four image-space corners to a normalized keyboard coordinate system in a process called homography. This homography then lets the app convert fingertip pixel positions into keyboard positions from `0` to `1`.

MediaPipe tracks hand landmarks and extracts fingertip points. The app smooths fingertip depth and keyboard position to reduce jitter using a Simple Exponential Smoothing formula. It then watches recent fingertip movement that meets the following conditions: 
- Direction reversal
- Large enough velocity change 
- Large enough travel distance
- Cooldown timing.

Keyboard layouts live in `src/keyboardLayouts.py`. The active layout is selected in `src/config.py` with `DETECTED_KEYBOARD`.

## Tuning

Change values found in  `src/config.py` for settings:

- `THRESHOLD`: Required velocity change for a press
- `MIN_PRESS_TRAVEL`: Required fingertip travel distance
- `PRESS_COOLDOWN_MS`: Per-finger cooldown after a press
- `SMOOTHING_ALPHA`: Smoothing for fingertip depth
- `KEYBOARD_POINT_SMOOTHING_ALPHA`: Smoothing for mapped keyboard position
- `KEY_HISTORY_SIZE`: How many recent key guesses are kept
