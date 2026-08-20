# Hand Motion Engine

<b>Real-time hand tracking and visual effects engine.</b>

## Requirements

- macOS or Windows 10/11 (64-bit)
- Python 3.11
- Webcam

## Installation

```bash
git clone https://github.com/deep-sengupta/hand_motion_engine.git
cd hand_motion_engine
```

```bash
python3.11 -m venv venv
source venv/bin/activate
```

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python motion_engine.py
```

## Controls

| Key | Action |
|---|---|
| `Space` / `E` | Next effect |
| `M` | Change quality mode |
| `L` | Toggle landmarks |
| `D` | Toggle debug |
| `R` | Start/stop recording |
| `Q` | Quit |

## Gestures

- ✌️ Peace — Next effect
- 👍 Thumbs Up — Previous effect
- ✋ Open Palm — Pause/resume effect
Effects activate only when both hands are visible and automatically change every 8 seconds.

## Performance
The engine supports up to 120 FPS, depending on camera hardware, resolution, and system performance.

For the best performance, use a 60/120 FPS camera and Performance mode.
