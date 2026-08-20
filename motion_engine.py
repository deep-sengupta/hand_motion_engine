import os
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except (ImportError, AttributeError) as exc:
    raise SystemExit(
        "MediaPipe Hands could not be loaded. "
        "Create a fresh virtual environment and install requirements.txt."
    ) from exc


PROJECT_NAME = "HAND MOTION ENGINE"
WINDOW_NAME = "Hand Motion Engine"

CAMERA_INDEX = 0
RESOLUTIONS = [
    (1280, 720),
    (960, 540),
    (640, 360),
]

FPS_TARGETS = [120, 90, 60, 30]

MODES = {
    "PERFORMANCE": {
        "detect_scale": 0.45,
        "effect_size": 0,
        "model_complexity": 0,
    },
    "BALANCED": {
        "detect_scale": 0.55,
        "effect_size": 0,
        "model_complexity": 0,
    },
    "QUALITY": {
        "detect_scale": 0.75,
        "effect_size": 0,
        "model_complexity": 1,
    },
}


def find_font(name):
    if not ImageFont:
        return None

    paths = [
        f"/System/Library/Fonts/Supplemental/{name}",
        f"/Library/Fonts/{name}",
        os.path.expanduser(f"~/Library/Fonts/{name}"),
    ]

    for path in paths:
        if os.path.exists(path):
            return path

    return None


OPEN_SANS = find_font("OpenSans-Regular.ttf")


def draw_text(frame, text, position, size=22, scale=1.0):
    if ImageFont and OPEN_SANS:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(OPEN_SANS, int(size * scale))
        draw.text(position, text, font=font, fill=(235, 245, 240))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65 * scale,
        (235, 245, 240),
        1,
        cv2.LINE_AA,
    )
    return frame


class FrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.running = True

    def put(self, frame):
        with self.lock:
            self.frame = frame
            self.sequence += 1

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, self.sequence
            return self.frame.copy(), self.sequence

    def stop(self):
        with self.lock:
            self.running = False


class CameraCapture(threading.Thread):
    def __init__(self, camera_index, width, height, requested_fps, buffer):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.requested_fps = requested_fps
        self.buffer = buffer
        self.cap = None
        self.actual_camera_fps = 0.0
        self.frames = 0
        self.start_time = 0.0

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.buffer.stop()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.start_time = time.perf_counter()

        while self.buffer.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.002)
                continue

            self.buffer.put(frame)
            self.frames += 1

            elapsed = time.perf_counter() - self.start_time
            if elapsed >= 1.0:
                self.actual_camera_fps = self.frames / elapsed
                self.frames = 0
                self.start_time = time.perf_counter()

        self.cap.release()

    def stop(self):
        self.buffer.stop()


def probe_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        return None

    results = []

    for width, height in RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        results.append((actual_w, actual_h))

    cap.release()

    if not results:
        return None

    preferred = sorted(
        results,
        key=lambda item: item[0] * item[1],
        reverse=True,
    )

    return preferred[0]


def make_gradient_lut(stops):
    stops = sorted(stops, key=lambda s: s[0])
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    for i in range(256):
        t = i / 255.0

        if t <= stops[0][0]:
            lut[i, 0] = stops[0][1]
            continue

        if t >= stops[-1][0]:
            lut[i, 0] = stops[-1][1]
            continue

        for j in range(len(stops) - 1):
            p0, c0 = stops[j]
            p1, c1 = stops[j + 1]

            if p0 <= t <= p1:
                local = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
                lut[i, 0] = [
                    c0[k] + (c1[k] - c0[k]) * local
                    for k in range(3)
                ]
                break

    return lut


COMIC_LUT = make_gradient_lut([
    (0.00, (20, 0, 10)),
    (0.30, (30, 20, 215)),
    (0.60, (30, 140, 255)),
    (0.80, (70, 235, 255)),
    (1.00, (240, 250, 255)),
])


def resize_work(patch, max_dim):
    h, w = patch.shape[:2]
    scale = min(1.0, max_dim / max(h, w))

    if scale >= 1.0:
        return patch, 1.0

    return cv2.resize(
        patch,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    ), scale


def restore_work(patch, target_shape):
    h, w = target_shape[:2]
    return cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)


def fx_grid(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=8)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = out.shape[:2]
    step = max(14, w // 18)

    for k, x in enumerate(range(0, w, step)):
        cv2.line(
            out,
            (x, 0),
            (x, h),
            (230, 225, 215) if k % 4 == 0 else (150, 140, 130),
            1,
            cv2.LINE_AA,
        )

    for k, y in enumerate(range(0, h, step)):
        cv2.line(
            out,
            (0, y),
            (w, y),
            (230, 225, 215) if k % 4 == 0 else (150, 140, 130),
            1,
            cv2.LINE_AA,
        )

    return out


def fx_comic(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = gray & 0xC0
    return cv2.applyColorMap(gray, COMIC_LUT)


def fx_glass(patch):
    blurred = cv2.GaussianBlur(patch, (25, 25), 0)
    glass = cv2.addWeighted(
        blurred,
        0.55,
        np.full_like(blurred, 255),
        0.45,
        0,
    )

    hsv = cv2.cvtColor(glass, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def fx_duotone(patch):
    gray = cv2.equalizeHist(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))

    stops = np.array([
        [70, 20, 10],
        [140, 40, 120],
        [90, 60, 235],
        [225, 215, 255],
    ], dtype=np.float32)

    lut = np.zeros((256, 3), dtype=np.uint8)
    segments = len(stops) - 1

    for i in range(256):
        t = i / 255.0
        segment = min(int(t * segments), segments - 1)
        local = t * segments - segment
        lut[i] = (
            stops[segment] * (1 - local)
            + stops[segment + 1] * local
        ).astype(np.uint8)

    return lut[gray]


def fx_paper(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        5,
    )

    detail = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    detail = cv2.threshold(detail, 32, 255, cv2.THRESH_BINARY)[1]

    paper = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    detail_bgr = cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)

    paper = cv2.addWeighted(paper, 0.82, detail_bgr, 0.28, 0)
    return cv2.convertScaleAbs(paper, alpha=1.12, beta=-8)


def fx_neon(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 45, 120)
    edge_color = np.zeros_like(patch)
    edge_color[edges > 0] = (255, 90, 210)

    glow_wide = cv2.GaussianBlur(edge_color, (0, 0), 7)
    glow_core = cv2.GaussianBlur(edge_color, (0, 0), 2)

    base = np.full_like(patch, (12, 8, 18))
    output = cv2.addWeighted(base, 1.0, glow_wide, 0.75, 0)
    output = cv2.addWeighted(output, 1.0, glow_core, 1.15, 0)
    return cv2.add(output, edge_color)


def fx_holographic(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)

    angle = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)

    hsv = cv2.merge([
        (angle * 179).astype(np.uint8),
        np.full_like(gray, 200, dtype=np.uint8),
        np.clip(gray * 1.2, 0, 255).astype(np.uint8),
    ])

    holo = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return cv2.addWeighted(holo, 0.75, patch, 0.25, 0)


EFFECTS = [
    ("COMIC", fx_comic),
    ("GLASS", fx_glass),
    ("DUOTONE", fx_duotone),
    ("PAPER", fx_paper),
    ("GRID", fx_grid),
    ("HOLOGRAPHIC", fx_holographic),
    ("NEON BLOOM", fx_neon),
]


def apply_effect(effect, patch, max_dim=0):
    return effect(patch)


def distance(a, b):
    ax, ay = float(a.x), float(a.y)
    bx, by = float(b.x), float(b.y)
    return float(np.hypot(ax - bx, ay - by))


def hand_block_points(lm, frame_width, frame_height):
    thumb = np.array(
        [float(lm[4].x) * frame_width, float(lm[4].y) * frame_height],
        dtype=np.float32,
    )
    index = np.array(
        [float(lm[8].x) * frame_width, float(lm[8].y) * frame_height],
        dtype=np.float32,
    )
    return index, thumb


def finger_extended(lm, tip, pip):
    return float(lm[tip].y) < float(lm[pip].y)


def gesture_from_landmarks(lm):
    index = finger_extended(lm, 8, 6)
    middle = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pinky = finger_extended(lm, 20, 18)

    thumb = distance(lm[4], lm[5]) > distance(lm[3], lm[5]) * 1.05

    count = int(index) + int(middle) + int(ring) + int(pinky)

    if thumb and count == 4:
        return "OPEN"

    if index and middle and not ring and not pinky:
        return "PEACE"

    if thumb and count == 0:
        return "THUMBS UP"

    if count == 0 and not thumb:
        return "FIST"

    return "NONE"


class GestureController:
    def __init__(self):
        self.last_gesture = "NONE"
        self.stable_count = 0
        self.cooldown = 0

    def update(self, gesture):
        if self.cooldown > 0:
            self.cooldown -= 1

        if gesture == self.last_gesture:
            self.stable_count += 1
        else:
            self.last_gesture = gesture
            self.stable_count = 1

        if self.stable_count < 8 or self.cooldown > 0:
            return None

        if gesture in {"PEACE", "THUMBS UP", "OPEN"}:
            self.cooldown = 30
            return gesture

        return None


def select_camera_mode():
    print()
    print(PROJECT_NAME)
    print("-" * len(PROJECT_NAME))
    print("Detecting camera...")

    detected = probe_camera()

    if detected:
        print(f"Camera resolution: {detected[0]}x{detected[1]}")
    else:
        print("Camera resolution could not be probed.")

    print("Starting adaptive 120/90/60/30 FPS capture.")
    return detected or (960, 540)


def draw_hud(
    frame,
    fps,
    camera_fps,
    effect_name,
    mode,
    resolution,
    recording,
    debug,
    hand_count,
    effect_paused=False,
):
    h, w = frame.shape[:2]

    frame = draw_text(
        frame,
        f"FPS {fps:3.0f}",
        (w - 150, 18),
        22,
    )

    frame = draw_text(
        frame,
        f"CAM {camera_fps:3.0f}",
        (w - 150, 45),
        17,
    )

    frame = draw_text(
        frame,
        f"EFFECT  {effect_name}",
        (18, h - 78),
        18,
    )

    frame = draw_text(
        frame,
        f"MODE  {mode}",
        (18, h - 53),
        17,
    )

    if hand_count != 2:
        effect_state = "WAIT"
    elif effect_paused:
        effect_state = "PAUSED"
    else:
        effect_state = "ON"
    frame = draw_text(
        frame,
        f"{resolution[0]}x{resolution[1]}  HANDS {hand_count}  EFFECT {effect_state}",
        (18, h - 29),
        16,
    )

    if recording:
        cv2.circle(frame, (20, 28), 8, (0, 0, 255), -1)
        frame = draw_text(frame, "REC", (36, 18), 18)

    if debug:
        frame = draw_text(
            frame,
            "DEBUG ON",
            (18, 18),
            17,
        )

    return frame


def main():
    resolution = select_camera_mode()
    width, height = resolution

    mode_names = list(MODES)
    mode_index = 0
    mode = mode_names[mode_index]

    frame_buffer = FrameBuffer()

    camera = CameraCapture(
        CAMERA_INDEX,
        width,
        height,
        120,
        frame_buffer,
    )
    camera.start()

    time.sleep(1.0)

    if camera.cap is None or not camera.cap.isOpened():
        raise SystemExit(
            "Could not open the webcam. Check macOS Camera permissions."
        )

    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=MODES[mode]["model_complexity"],
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    effect_index = 0
    gesture_controller = GestureController()

    show_landmarks = True
    debug = False
    effect_paused = False
    auto_effect_seconds = 8.0
    last_effect_change = time.perf_counter()

    block_corners = None
    block_smoothing = 0.35
    block_miss_frames = 0
    block_hold_frames = 4

    recording = False
    writer = None
    recording_path = None

    frame_times = deque(maxlen=30)
    last_sequence = -1
    last_processed_time = time.perf_counter()

    processing_fps = 0.0
    camera_fps = 0.0

    last_effect_time = 0.0
    effect_cooldown = 0.7

    while frame_buffer.running:
        frame, sequence = frame_buffer.get()

        if frame is None or sequence == last_sequence:
            time.sleep(0.001)
            continue

        last_sequence = sequence

        start = time.perf_counter()

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        settings = MODES[mode]

        detect_size = max(
            1,
            int(frame_width * settings["detect_scale"]),
        )

        detect_frame = cv2.resize(
            frame,
            (detect_size, int(frame_height * settings["detect_scale"])),
            interpolation=cv2.INTER_AREA,
        )

        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_count = 0
        gesture = "NONE"
        hand_points = []
        block_targets = []

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)

            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                if hand_count == 1:
                    gesture = gesture_from_landmarks(lm)

                points = np.array([
                    [
                        int(p.x * frame_width),
                        int(p.y * frame_height),
                    ]
                    for p in lm
                ])

                hand_points.append(points)

                if hand_count >= 2:
                    index_point, thumb_point = hand_block_points(
                        lm,
                        frame_width,
                        frame_height,
                    )
                    block_targets.append((index_point, thumb_point))

                if show_landmarks:
                    lm_points = [
                        0, 1, 2, 3, 4,
                        5, 6, 8,
                        9, 10, 12,
                        13, 14, 16,
                        17, 18, 20,
                    ]

                    for idx in lm_points:
                        px = int(lm[idx].x * frame_width)
                        py = int(lm[idx].y * frame_height)
                        cv2.circle(
                            frame,
                            (px, py),
                            4,
                            (120, 255, 180),
                            -1,
                            cv2.LINE_AA,
                        )

                    palm = np.array([
                        [
                            int(lm[i].x * frame_width),
                            int(lm[i].y * frame_height),
                        ]
                        for i in [0, 5, 9, 13, 17]
                    ])

                    palm_hull = cv2.convexHull(palm)
                    cv2.polylines(
                        frame,
                        [palm_hull],
                        True,
                        (120, 255, 180),
                        1,
                        cv2.LINE_AA,
                    )

        action = gesture_controller.update(gesture)
        now = time.perf_counter()

        if action and now - last_effect_time > effect_cooldown:
            if action == "PEACE":
                effect_index = (effect_index + 1) % len(EFFECTS)
                last_effect_change = now
                last_effect_time = now

            elif action == "THUMBS UP":
                effect_index = (effect_index - 1) % len(EFFECTS)
                last_effect_change = now
                last_effect_time = now

            elif action == "OPEN":
                effect_paused = not effect_paused
                last_effect_time = now
                last_effect_change = now

        if len(hand_points) == 2 and not effect_paused:
            if now - last_effect_change >= auto_effect_seconds:
                effect_index = (effect_index + 1) % len(EFFECTS)
                last_effect_change = now

        if len(block_targets) == 2:
            block_targets.sort(key=lambda pair: float(pair[0][0]))

            target_corners = np.array(
                [
                    block_targets[0][0],
                    block_targets[0][1],
                    block_targets[1][1],
                    block_targets[1][0],
                ],
                dtype=np.float32,
            )

            if block_corners is None:
                block_corners = target_corners.copy()
            else:
                block_corners += (
                    target_corners - block_corners
                ) * block_smoothing

            block_miss_frames = 0
        else:
            block_miss_frames += 1
            if block_miss_frames > block_hold_frames:
                block_corners = None

        if (
            block_corners is not None
            and len(block_targets) == 2
            and not effect_paused
        ):
            block_poly = cv2.convexHull(
                np.round(block_corners).astype(np.int32)
            )

            x, y, bw, bh = cv2.boundingRect(block_poly)

            padding = max(4, int(max(bw, bh) * 0.025))

            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(frame_width, x + bw + padding)
            y1 = min(frame_height, y + bh + padding)

            if x1 - x0 > 30 and y1 - y0 > 30:
                patch = frame[y0:y1, x0:x1]

                processed = apply_effect(
                    EFFECTS[effect_index][1],
                    patch,
                    settings["effect_size"],
                )

                mask = np.zeros(
                    (y1 - y0, x1 - x0),
                    dtype=np.uint8,
                )

                local_poly = block_poly.reshape(-1, 2) - np.array(
                    [x0, y0],
                    dtype=np.int32,
                )

                cv2.fillConvexPoly(mask, local_poly, 255)

                kernel_size = max(
                    3,
                    int(min(mask.shape[:2]) * 0.008),
                )
                if kernel_size % 2 == 0:
                    kernel_size += 1

                mask = cv2.GaussianBlur(
                    mask,
                    (kernel_size, kernel_size),
                    0,
                )

                alpha = (mask.astype(np.float32) / 255.0)[..., None]

                patch[:] = (
                    patch.astype(np.float32) * (1.0 - alpha)
                    + processed.astype(np.float32) * alpha
                ).astype(np.uint8)


        elapsed = time.perf_counter() - start

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            processing_fps = (
                instant_fps
                if processing_fps == 0
                else processing_fps * 0.85 + instant_fps * 0.15
            )

        camera_fps = camera.actual_camera_fps

        if recording and writer is not None:
            writer.write(frame)

        frame = draw_hud(
            frame,
            processing_fps,
            camera_fps,
            EFFECTS[effect_index][0],
            mode,
            (frame_width, frame_height),
            recording,
            debug,
            hand_count,
            effect_paused,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            effect_index = (effect_index + 1) % len(EFFECTS)

        elif key == ord("l"):
            show_landmarks = not show_landmarks

        elif key == ord("d"):
            debug = not debug

        elif key == ord("m"):
            mode_index = (mode_index + 1) % len(mode_names)
            mode = mode_names[mode_index]

            hands.close()
            hands = mp_hands.Hands(
                max_num_hands=2,
                model_complexity=MODES[mode]["model_complexity"],
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        elif key == ord("r"):
            if not recording:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                recording_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    f"motion_{stamp}.mp4",
                )

                record_fps = max(
                    30.0,
                    min(
                        120.0,
                        camera_fps if camera_fps > 0 else 60.0,
                    ),
                )

                writer = cv2.VideoWriter(
                    recording_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    record_fps,
                    (frame_width, frame_height),
                )

                if writer.isOpened():
                    recording = True
                    print(
                        f"Recording: {recording_path} "
                        f"at {record_fps:.0f} FPS"
                    )
                else:
                    writer.release()
                    writer = None
                    print("Could not start recording.")

            else:
                recording = False

                if writer is not None:
                    writer.release()
                    writer = None

                print(f"Saved recording: {recording_path}")

        elif key == ord("e"):
            effect_index = (effect_index + 1) % len(EFFECTS)

    if writer is not None:
        writer.release()

    hands.close()
    camera.stop()

    if camera.is_alive():
        camera.join(timeout=1.0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
