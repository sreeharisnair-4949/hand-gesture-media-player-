"""
AI-Based Hand Gesture Controlled Media Player - ALL-IN-ONE FILE (v3)
========================================================================
NEW IN THIS VERSION (v3 - hardening pass):
  1. CROSS-PLATFORM VOLUME CONTROL: pycaw (Windows) with automatic fallback
     to pyautogui media keys on macOS/Linux or if pycaw isn't installed.
  2. PREDICTION SMOOTHING: real-time predictions are voted over a small
     rolling window instead of trusting a single noisy frame.
  3. DROPPED-FRAME TOLERANCE: a hand briefly leaving frame / an occluded
     finger no longer instantly resets your gesture-hold progress.
  4. SAFER TRAINING: falls back gracefully if a gesture has too few
     samples to stratify, and warns you instead of crashing.
  5. DATA MANAGEMENT: new menu option to inspect / delete a gesture's
     collected samples without hand-editing the CSV.
  6. FRIENDLY ERRORS: missing optional dependencies (pycaw, pyautogui) and
     an unreachable webcam now produce clear guidance instead of tracebacks.
  7. SEPARATE COUNTDOWN CONSTANT: pre-capture "get ready" countdown is no
     longer tied to the gesture-hold duration used during playback.

Run stages IN ORDER the first time:
  1) Collect Data                      -> gesture_data.csv
  2) Train Model                       -> gesture_model.pkl, label_encoder.pkl
  3) Customize Gesture-Action Mapping  -> gesture_config.json
  4) Run Media Player                  -> real-time gesture control
  5) Manage Collected Data             -> inspect / delete gestures

Install dependencies:
  pip install opencv-python mediapipe scikit-learn pandas numpy matplotlib seaborn pyautogui
  # Windows only, for direct system volume control:
  pip install pycaw comtypes

Run:
  python gesture_app.py
"""

import os
import sys
import csv
import json
import time
import pickle
import platform
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp

# ============================================================
# CONFIG
# ============================================================
CSV_FILE = "gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
CONFIG_FILE = "gesture_config.json"

SAMPLES_PER_GESTURE = 300
COUNTDOWN_SECONDS = 2.0     # "get ready" pause before capture starts
HOLD_SECONDS = 2.0          # how long a gesture must be held to trigger an action
CONFIDENCE_THRESHOLD = 0.75
MIN_SAMPLES_TO_TRAIN = 20   # per gesture, below this training is refused

# Real-time smoothing: predictions are voted over this many recent frames,
# and a hand can vanish for up to MAX_MISSED_FRAMES consecutive frames
# without resetting the current gesture-hold progress.
SMOOTHING_WINDOW = 7
MAX_MISSED_FRAMES = 5

# How many synthetic rotated copies to generate per REAL training sample.
# Higher = more angle robustness, but slower training.
AUGMENTATIONS_PER_SAMPLE = 8
MAX_ROTATION_DEGREES = 30  # simulate hand tilted up to +/-30 degrees on any axis

# --- Air Mouse (cursor control) tuning ---
# This mode reads raw hand landmarks directly every frame (no trained model
# needed) since cursor movement must be continuous, not a held/triggered
# gesture like the rest of the app.
MOUSE_FRAME_REDUCTION = 100   # pixels of camera-frame margin excluded from the
                               # move region, so you don't have to reach the
                               # physical edges of the frame to hit screen edges
MOUSE_SMOOTHENING = 7          # higher = smoother but laggier cursor movement
MOUSE_PINCH_CLICK_RATIO = 0.35 # thumb-index (or index-middle) distance, as a
                               # fraction of hand size, below which counts as a pinch/click
MOUSE_CLICK_COOLDOWN = 0.5     # seconds between accepted clicks, to avoid multi-firing
MOUSE_SCROLL_SENSITIVITY = 2   # scroll "clicks" per pixel of vertical hand movement
MOUSE_SCROLL_DEAD_ZONE = 6     # pixels of vertical movement ignored, to avoid jitter

MEDIA_ACTIONS = ["play_pause", "next_track", "prev_track", "volume_up", "volume_down", "mute", "none"]

# Presentation (PowerPoint / Google Slides / most viewers) actions, driven by
# the same standard keyboard shortcuts these apps already listen for -
# no PowerPoint-specific API needed, just simulated key presses.
PPT_ACTIONS = [
    "ppt_next_slide", "ppt_prev_slide", "ppt_start_slideshow",
    "ppt_end_slideshow", "ppt_blank_screen", "none",
]

# Combined list of every action the model's ACTION_HANDLERS dict knows about.
ACTION_CHOICES = sorted(set(MEDIA_ACTIONS) | set(PPT_ACTIONS))

# Default mapping, used only if no gesture_config.json exists yet.
DEFAULT_GESTURE_ACTIONS = {
    "open_palm": "play_pause",
    "fist": "prev_track",
    "peace": "next_track",
    "thumbs_up": "volume_up",
    "thumbs_down": "volume_down",
}

# Default mapping for Presentation Mode - deliberately reuses the SAME
# gestures as media mode (open_palm/fist/peace/thumbs_up/thumbs_down), just
# pointed at slide-control actions instead. Media and Presentation mappings
# are stored and customized independently, so switching modes in Stage 4
# switches what each gesture does without retraining anything.
DEFAULT_PRESENTATION_GESTURE_ACTIONS = {
    "peace": "ppt_next_slide",
    "fist": "ppt_prev_slide",
    "open_palm": "ppt_start_slideshow",
    "thumbs_down": "ppt_end_slideshow",
    "thumbs_up": "ppt_blank_screen",
}

GESTURE_MODES = {
    "media": {"label": "Media Player", "actions": MEDIA_ACTIONS, "default": DEFAULT_GESTURE_ACTIONS},
    "presentation": {"label": "Presentation Control", "actions": PPT_ACTIONS,
                      "default": DEFAULT_PRESENTATION_GESTURE_ACTIONS},
}

# Preset gesture library offered during data collection, so users can pick
# from known, easy-to-perform-consistently gestures instead of having to
# invent their own from scratch. Custom gestures are still always allowed
# alongside these.
PRESET_GESTURES = {
    "open_palm": "Open hand, fingers spread, palm facing the camera",
    "fist": "Closed fist, all fingers curled in",
    "peace": "Index and middle finger up in a V, other fingers curled",
    "thumbs_up": "Fist with thumb pointing up",
    "thumbs_down": "Fist with thumb pointing down",
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# All gestures are canonicalized to look like this hand before being stored
# or classified. A hand of the OTHER handedness gets its x-axis mirrored so
# both hands map to the same geometry for the same gesture - this means one
# model works for both hands without needing separate training data per hand.
CANONICAL_HANDEDNESS = "Right"


def get_handedness_label(results, hand_index):
    """Returns 'Left' or 'Right' for the given detected hand, or None if
    MediaPipe didn't report handedness (rare, but handled defensively)."""
    try:
        return results.multi_handedness[hand_index].classification[0].label
    except (AttributeError, IndexError, TypeError):
        return None


# ============================================================
# LANDMARK PREPROCESSING (angle/scale/position invariance)
# ============================================================
def landmarks_to_array(hand_landmarks, mirror=False):
    """
    Convert MediaPipe landmarks to a (21,3) numpy array, translated so the
    wrist (landmark 0) sits at the origin. This removes POSITION dependence
    (gesture recognized the same regardless of where the hand is in frame).

    If mirror=True, the x-axis is flipped after centering, which converts a
    left-hand pose into its right-hand mirror image (or vice versa). Used to
    canonicalize both hands to the same shape so one model handles either.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float64)
    pts -= pts[0]
    if mirror:
        pts[:, 0] *= -1
    return pts


def preprocess_points(pts):
    """
    Takes wrist-centered (21,3) points and makes them:
      - SCALE invariant: normalized by wrist->middle-finger-MCP distance,
        so it works regardless of hand size or distance from camera.
      - IN-PLANE ROTATION invariant: rotated so the wrist->middle-finger-MCP
        vector always points "up", so it works if the hand is tilted
        sideways in the camera view.
    Returns a flat 63-length feature vector fed into the classifier.
    """
    pts = pts.copy()
    scale = np.linalg.norm(pts[9]) + 1e-8
    pts /= scale
    ref_x, ref_y = pts[9][0], pts[9][1]
    angle = np.arctan2(ref_x, -ref_y)
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts[:, :2] = pts[:, :2] @ rot.T
    return pts.flatten()


def random_rotation_matrix(max_angle_deg=MAX_ROTATION_DEGREES):
    """A random small 3D rotation, used to simulate viewing the hand from a
    different angle (tilted toward/away from camera, turned left/right)."""
    ax, ay, az = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg, size=3))
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def augment_points(pts, n_augments=AUGMENTATIONS_PER_SAMPLE, max_angle_deg=MAX_ROTATION_DEGREES):
    """Generate n_augments synthetic rotated copies of a wrist-centered point set."""
    return [pts @ random_rotation_matrix(max_angle_deg).T for _ in range(n_augments)]


# ============================================================
# SHARED HELPERS
# ============================================================
def open_camera(index=0):
    """Open a webcam with a friendly error instead of a silent black window."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"\n[ERROR] Could not open webcam (index {index}).")
        print("  - Check that no other application is using the camera.")
        print("  - On Linux, verify permissions (e.g. user in 'video' group).")
        print("  - Try a different camera index if you have more than one.")
        return None
    return cap


def make_hands():
    return mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.7,
    )


def sanitize_label(label):
    """Prevent labels that would corrupt the CSV or collide with config keys."""
    label = label.strip()
    if "," in label or '"' in label:
        return None
    if label.lower() in ("none", "exit", ""):
        return None
    return label


# ============================================================
# STAGE 1: DATA COLLECTION
# ============================================================
def get_landmark_row(hand_landmarks, label, mirror=False):
    pts = landmarks_to_array(hand_landmarks, mirror=mirror)
    row = pts.flatten().tolist()
    row.append(label)
    return row


def ensure_csv_header():
    if not os.path.exists(CSV_FILE):
        header = []
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        header.append("label")
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(header)


def get_existing_counts():
    """Sample counts per gesture already saved in the CSV, used so the picker
    can show the user what they've already collected."""
    if not os.path.exists(CSV_FILE):
        return {}
    import pandas as pd
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            return {}
        return df["label"].value_counts().to_dict()
    except Exception:
        return {}


def choose_gesture_to_collect():
    """Show a preset gesture library (with descriptions + progress) plus a
    custom-name option. Returns the chosen label, or None to exit collection."""
    counts = get_existing_counts()
    presets = list(PRESET_GESTURES.items())

    while True:
        print("\nChoose a gesture to record:")
        for i, (name, desc) in enumerate(presets, start=1):
            have = counts.get(name, 0)
            done = " (done)" if have >= SAMPLES_PER_GESTURE else ""
            print(f"  {i}) {name:12s} - {desc}  [{have}/{SAMPLES_PER_GESTURE} samples{done}]")
        print("  C) Custom gesture (type your own name)")
        print("  X) Exit data collection")

        choice = input("Your choice: ").strip()
        if choice.lower() == "x":
            return None
        if choice.lower() == "c":
            raw_label = input("Enter your custom gesture name: ").strip()
            label = sanitize_label(raw_label)
            if label is None:
                print("  Invalid label: avoid commas/quotes, and don't use 'none' or 'exit'.")
                continue
            return label
        try:
            idx = int(choice)
            if 1 <= idx <= len(presets):
                return presets[idx - 1][0]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def collect_data():
    ensure_csv_header()
    cap = open_camera()
    if cap is None:
        return
    hands = make_hands()

    while True:
        label = choose_gesture_to_collect()
        if label is None:
            break

        print(f"\nGet ready to show '{label}'. Press 's' in the camera window "
              f"to start a {COUNTDOWN_SECONDS:.0f}s countdown, then capturing "
              f"{SAMPLES_PER_GESTURE} samples begins. Press 'q' to stop early.")

        collecting = False
        countdown_start = None
        count = 0
        rows_buffer = []

        while True:
            success, frame = cap.read()
            if not success:
                print("[WARNING] Failed to read a frame from the webcam; retrying...")
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            in_countdown = countdown_start is not None and not collecting
            remaining = 0.0
            if in_countdown:
                elapsed = time.time() - countdown_start
                remaining = COUNTDOWN_SECONDS - elapsed
                if remaining <= 0:
                    collecting = True
                    countdown_start = None

            hand_label_text = "none"
            if result.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    handedness = get_handedness_label(result, idx)
                    hand_label_text = handedness or "unknown"
                    mirror = handedness is not None and handedness != CANONICAL_HANDEDNESS
                    if collecting and count < SAMPLES_PER_GESTURE:
                        rows_buffer.append(get_landmark_row(hand_landmarks, label, mirror=mirror))
                        count += 1

            cv2.putText(frame, f"Gesture: {label} | Hand: {hand_label_text} | Captured: {count}/{SAMPLES_PER_GESTURE}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if in_countdown:
                status, color = f"Get ready... starting in {remaining:.1f}s", (0, 255, 255)
            elif collecting:
                status, color = "COLLECTING... (tilt hand slightly for variety)", (0, 0, 255)
            else:
                status, color = "Press 's' to start", (0, 0, 255)
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Data Collection - press s to start, q to stop", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and not collecting and countdown_start is None:
                countdown_start = time.time()
            elif key == ord("q"):
                break
            if count >= SAMPLES_PER_GESTURE:
                print(f"Done capturing {SAMPLES_PER_GESTURE} samples for '{label}'.")
                break

        if rows_buffer:
            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerows(rows_buffer)
            print(f"Saved {len(rows_buffer)} rows for gesture '{label}' to {CSV_FILE}")
        else:
            print(f"No samples captured for '{label}' (nothing saved).")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Your dataset is saved in {CSV_FILE}")


# ============================================================
# STAGE 2: TRAIN YOUR OWN MODEL (with rotation augmentation)
# ============================================================
def train_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib
    matplotlib.use("Agg")  # avoid backend issues on headless / non-GUI runs
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not os.path.exists(CSV_FILE):
        print(f"No {CSV_FILE} found. Run Stage 1 (Collect Data) first.")
        return

    df = pd.read_csv(CSV_FILE)
    if df.empty:
        print(f"{CSV_FILE} is empty. Run Stage 1 (Collect Data) first.")
        return

    counts = df["label"].value_counts()
    print(f"Loaded {len(df)} raw samples across {df['label'].nunique()} gestures:")
    print(counts)

    if df["label"].nunique() < 2:
        print("\n[ERROR] Need at least 2 different gestures to train a classifier.")
        return

    too_few = counts[counts < MIN_SAMPLES_TO_TRAIN]
    if not too_few.empty:
        print(f"\n[ERROR] These gestures have fewer than {MIN_SAMPLES_TO_TRAIN} samples "
              f"and would produce an unreliable model:")
        print(too_few)
        print("Collect more samples for them (Stage 1) before training.")
        return

    X_raw = df.drop("label", axis=1).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # Stratified split needs at least 2 samples per class in the split;
    # fall back gracefully instead of crashing on small/imbalanced datasets.
    min_class_count = counts.min()
    can_stratify = min_class_count >= 2
    split_kwargs = dict(test_size=0.2, random_state=42)
    if can_stratify:
        split_kwargs["stratify"] = y
    else:
        print(f"\n[WARNING] Smallest gesture class has only {min_class_count} sample(s); "
              f"skipping stratified split.")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, **split_kwargs)

    print(f"\nAugmenting training data with random 3D rotations "
          f"(+/-{MAX_ROTATION_DEGREES} deg) for angle robustness...")
    X_train_list, y_train_list = [], []
    for i in range(len(X_train_raw)):
        pts = X_train_raw[i].reshape(21, 3)
        X_train_list.append(preprocess_points(pts))
        y_train_list.append(y_train[i])
        for rotated_pts in augment_points(pts):
            X_train_list.append(preprocess_points(rotated_pts))
            y_train_list.append(y_train[i])

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array([preprocess_points(X_test_raw[i].reshape(21, 3)) for i in range(len(X_test_raw))])

    print(f"Training set expanded from {len(X_train_raw)} to {len(X_train)} samples "
          f"({AUGMENTATIONS_PER_SAMPLE} rotated copies per original sample).")

    model = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy (on ORIGINAL, non-augmented held-out data): {acc * 100:.2f}%")

    # classification_report needs labels actually present in y_test/y_pred
    present_labels = sorted(set(y_test) | set(y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=encoder.inverse_transform(present_labels),
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.inverse_transform(present_labels),
                yticklabels=encoder.inverse_transform(present_labels))
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix - Gesture Classifier (rotation-augmented training)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nSaved confusion_matrix.png (include this in your report)")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(encoder, f)
    print(f"\nSaved trained model -> {MODEL_FILE}")
    print(f"Saved label encoder -> {ENCODER_FILE}")


# ============================================================
# STAGE 3: CUSTOMIZE GESTURE -> ACTION MAPPING
# ============================================================
def _load_raw_config():
    """Load gesture_config.json, migrating the old flat format (a single
    gesture->action mapping, from before Presentation Mode existed) into the
    new {"media": {...}, "presentation": {...}} shape automatically."""
    if not os.path.exists(CONFIG_FILE):
        return {"media": dict(DEFAULT_GESTURE_ACTIONS),
                "presentation": dict(DEFAULT_PRESENTATION_GESTURE_ACTIONS)}
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARNING] Could not read {CONFIG_FILE} ({e}); using defaults.")
        return {"media": dict(DEFAULT_GESTURE_ACTIONS),
                "presentation": dict(DEFAULT_PRESENTATION_GESTURE_ACTIONS)}

    if "media" in data or "presentation" in data:
        return {
            "media": data.get("media", dict(DEFAULT_GESTURE_ACTIONS)),
            "presentation": data.get("presentation", dict(DEFAULT_PRESENTATION_GESTURE_ACTIONS)),
        }
    # Old flat-format config (pre-Presentation-Mode) - treat it as the media mapping.
    print(f"[INFO] Migrating older {CONFIG_FILE} format to support multiple modes.")
    return {"media": data, "presentation": dict(DEFAULT_PRESENTATION_GESTURE_ACTIONS)}


def load_gesture_actions(mode, valid_gestures=None):
    """Load the mapping for a given mode ('media' or 'presentation'), silently
    dropping entries pointing at gestures the current model doesn't know or
    actions that no longer exist for that mode."""
    allowed_actions = GESTURE_MODES[mode]["actions"]
    mapping = _load_raw_config().get(mode, {})

    cleaned = {}
    for g, a in mapping.items():
        if a not in allowed_actions or a == "none":
            continue
        if valid_gestures is not None and g not in valid_gestures:
            continue
        cleaned[g] = a
    return cleaned


def save_gesture_actions(mode, mapping):
    """Save the mapping for one mode without disturbing the other mode's config."""
    data = _load_raw_config()
    data[mode] = mapping
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def choose_mode():
    """Ask which mode to work with. Returns 'media' or 'presentation', or None to cancel."""
    print("\nWhich mode?")
    print("  1) Media Player          (play/pause, next/prev track, volume, mute)")
    print("  2) Presentation Control  (next/prev slide, start/end slideshow, blank screen)")
    print("  X) Cancel")
    while True:
        choice = input("Your choice: ").strip().lower()
        if choice == "1":
            return "media"
        if choice == "2":
            return "presentation"
        if choice == "x":
            return None
        print("  Invalid choice, try again.")


def customize_gestures():
    if not os.path.exists(ENCODER_FILE):
        print("No trained model found yet. Run Stage 2 (Train Model) first.")
        return

    mode = choose_mode()
    if mode is None:
        print("Cancelled.")
        return
    mode_info = GESTURE_MODES[mode]
    mode_actions = mode_info["actions"]

    with open(ENCODER_FILE, "rb") as f:
        encoder = pickle.load(f)
    gestures = list(encoder.classes_)
    current = load_gesture_actions(mode, valid_gestures=gestures)

    print(f"\nCustomizing '{mode_info['label']}' mode.")
    print("Your trained gestures:", gestures)
    print("Available actions:", mode_actions)
    print("(Press Enter to keep the current/no mapping for a gesture)\n")

    mapping = {}
    for g in gestures:
        existing = current.get(g, "none")
        while True:
            choice = input(f"  Action for '{g}' [current: {existing}]: ").strip()
            if choice == "":
                if existing != "none":
                    mapping[g] = existing
                break
            if choice in mode_actions:
                if choice != "none":
                    mapping[g] = choice
                break
            print(f"  Invalid choice. Pick from: {mode_actions}")

    save_gesture_actions(mode, mapping)
    print(f"\nSaved your '{mode_info['label']}' gesture mapping to {CONFIG_FILE}:")
    print(json.dumps(mapping, indent=2))



# ============================================================
# STAGE 4: RUN THE REAL-TIME MEDIA PLAYER
# ============================================================
def _volume_via_pycaw(delta):
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    current = volume.GetMasterVolumeLevelScalar()
    new_level = min(1.0, max(0.0, current + delta))
    volume.SetMasterVolumeLevelScalar(new_level, None)
    print(f"[VOLUME] {current*100:.0f}% -> {new_level*100:.0f}%")


def _volume_via_pyautogui(delta):
    import pyautogui
    # Fallback: simulate the physical volume key the right number of times.
    # Each key press is roughly a 2% step on most systems.
    key = "volumeup" if delta > 0 else "volumedown"
    presses = max(1, round(abs(delta) / 0.02))
    for _ in range(presses):
        pyautogui.press(key)
    print(f"[VOLUME] pressed '{key}' x{presses} (pycaw unavailable, using key simulation)")


def set_volume_step(delta):
    """Change system volume. Uses pycaw for precise control on Windows;
    falls back to simulated media keys elsewhere or if pycaw isn't installed."""
    if platform.system() == "Windows":
        try:
            _volume_via_pycaw(delta)
            return
        except Exception as e:
            print(f"[WARNING] pycaw volume control failed ({e}); falling back to key simulation.")
    try:
        _volume_via_pyautogui(delta)
    except Exception as e:
        print(f"[ERROR] Could not adjust volume: {e}")


ACTION_HANDLERS = {
    "play_pause": lambda: _press_media_key("playpause"),
    "next_track": lambda: _press_media_key("nexttrack"),
    "prev_track": lambda: _press_media_key("prevtrack"),
    "volume_up": lambda: set_volume_step(0.05),
    "volume_down": lambda: set_volume_step(-0.05),
    "mute": lambda: _press_media_key("volumemute"),

    # Presentation control - these are the same keyboard shortcuts PowerPoint,
    # Google Slides, and most other presentation apps already respond to, so
    # this requires no PowerPoint-specific API - just make sure the
    # presentation window is focused/active on screen when you gesture.
    "ppt_next_slide": lambda: _press_media_key("right"),
    "ppt_prev_slide": lambda: _press_media_key("left"),
    "ppt_start_slideshow": lambda: _press_media_key("f5"),
    "ppt_end_slideshow": lambda: _press_media_key("esc"),
    "ppt_blank_screen": lambda: _press_media_key("b"),
}


def _press_media_key(key):
    try:
        import pyautogui
        pyautogui.press(key)
    except Exception as e:
        print(f"[ERROR] Could not send media key '{key}': {e}")


def perform_action(action):
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        print(f"[WARNING] Unknown action '{action}' - skipping.")
        return
    handler()
    print(f"[ACTION TRIGGERED] {action}")


def run_media_player():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE)):
        print("No trained model found. Run Stage 2 (Train Model) first.")
        return

    mode = choose_mode()
    if mode is None:
        print("Cancelled.")
        return
    mode_label = GESTURE_MODES[mode]["label"]

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_FILE, "rb") as f:
        encoder = pickle.load(f)

    gesture_actions = load_gesture_actions(mode, valid_gestures=list(encoder.classes_))
    print(f"\n{mode_label} mode active. Gesture -> action mapping:", gesture_actions)
    print("(Run Stage 3 from the menu any time to change this.)")
    if mode == "presentation":
        print("Make sure your presentation window/app is focused on screen "
              "so the simulated key presses (arrow keys, F5, Esc, B) reach it.")

    hands = make_hands()
    cap = open_camera()
    if cap is None:
        return

    current_gesture = None
    gesture_start_time = None
    action_done_for_current_hold = False
    last_triggered_action = ""
    missed_frames = 0
    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)

    window_title = f"AI Hand Gesture Control - {mode_label} (ESC to quit)"
    print(f"Running {mode_label}. Press ESC in the camera window to quit.")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARNING] Failed to read a frame from the webcam; retrying...")
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            raw_label = None
            confidence = 0.0

            hand_label_text = "none"
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                handedness = get_handedness_label(result, 0)
                hand_label_text = handedness or "unknown"
                mirror = handedness is not None and handedness != CANONICAL_HANDEDNESS
                pts = landmarks_to_array(hand_landmarks, mirror=mirror)
                features = preprocess_points(pts).reshape(1, -1)
                probs = model.predict_proba(features)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                if confidence >= CONFIDENCE_THRESHOLD:
                    raw_label = encoder.inverse_transform([pred_idx])[0]

            # --- smoothing: vote over recent frames instead of trusting one frame ---
            if raw_label is not None:
                recent_predictions.append(raw_label)
                missed_frames = 0
            else:
                missed_frames += 1
                if missed_frames > MAX_MISSED_FRAMES:
                    recent_predictions.clear()

            predicted_label = None
            if recent_predictions:
                vote_label, vote_count = Counter(recent_predictions).most_common(1)[0]
                # require a real majority within the window, not just a plurality of 1
                if vote_count >= max(2, len(recent_predictions) // 2 + 1):
                    predicted_label = vote_label

            # --- gesture-hold logic (tolerant of brief hand loss) ---
            if predicted_label is not None:
                if predicted_label != current_gesture:
                    current_gesture = predicted_label
                    gesture_start_time = time.time()
                    action_done_for_current_hold = False
                else:
                    held_time = time.time() - gesture_start_time
                    if held_time >= HOLD_SECONDS and not action_done_for_current_hold:
                        action = gesture_actions.get(current_gesture)
                        if action:
                            perform_action(action)
                            last_triggered_action = f"{current_gesture} -> {action}"
                        action_done_for_current_hold = True
            elif missed_frames > MAX_MISSED_FRAMES:
                current_gesture = None
                gesture_start_time = None
                action_done_for_current_hold = False
            # else: brief gap, keep current_gesture / timer alive

            y = 30
            cv2.putText(frame, f"Prediction: {predicted_label} ({confidence*100:.0f}%) | Hand: {hand_label_text}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            if current_gesture and gesture_start_time:
                held = min(time.time() - gesture_start_time, HOLD_SECONDS)
                cv2.putText(frame, f"Hold: {held:.1f}/{HOLD_SECONDS}s",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                y += 30
            cv2.putText(frame, f"Last action: {last_triggered_action}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# STAGE 5: MANAGE COLLECTED DATA
# ============================================================
def manage_data():
    if not os.path.exists(CSV_FILE):
        print(f"No {CSV_FILE} found yet. Nothing to manage.")
        return
    import pandas as pd
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        print(f"{CSV_FILE} is empty.")
        return

    print("\nCollected gestures:")
    print(df["label"].value_counts())

    choice = input(
        "\nEnter a gesture name to delete all its rows, or press Enter to cancel: "
    ).strip()
    if not choice:
        print("Cancelled.")
        return
    if choice not in df["label"].values:
        print(f"'{choice}' not found in {CSV_FILE}.")
        return

    confirm = input(f"Delete all {sum(df['label'] == choice)} rows for '{choice}'? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return

    df = df[df["label"] != choice]
    df.to_csv(CSV_FILE, index=False)
    print(f"Deleted gesture '{choice}'. Remember to re-run Stage 2 (Train Model) "
          f"to update the model, since it still reflects the old data.")


# ============================================================
# STAGE 6: AIR MOUSE (CURSOR CONTROL)
# ============================================================
# This mode does NOT use the trained classifier at all - it reads raw
# fingertip positions every single frame to move the cursor continuously,
# which the trained "hold a gesture for N seconds" system isn't designed
# for. Finger-up/down state and pinch distances are computed directly from
# MediaPipe landmarks using simple geometry.
_MOUSE_TIP_IDS = [8, 12, 16, 20]   # index, middle, ring, pinky fingertips
_MOUSE_PIP_IDS = [6, 10, 14, 18]   # corresponding lower knuckles


def _mouse_fingers_up(landmarks):
    """Returns [index_up, middle_up, ring_up, pinky_up] booleans. A finger
    counts as 'up' if its tip is above (smaller y than) its own lower
    knuckle - orientation-independent, works for either hand."""
    return [landmarks[tip].y < landmarks[pip].y
            for tip, pip in zip(_MOUSE_TIP_IDS, _MOUSE_PIP_IDS)]


def _mouse_pixel_dist(landmarks, i, j, frame_w, frame_h):
    x1, y1 = landmarks[i].x * frame_w, landmarks[i].y * frame_h
    x2, y2 = landmarks[j].x * frame_w, landmarks[j].y * frame_h
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def run_mouse_control():
    try:
        import pyautogui
    except ImportError:
        print("[ERROR] pyautogui is required for Air Mouse control. "
              "Install it with: pip install pyautogui")
        return

    screen_w, screen_h = pyautogui.size()
    cap = open_camera()
    if cap is None:
        return
    hands = make_hands()

    prev_x, prev_y = screen_w // 2, screen_h // 2
    prev_scroll_y = None
    last_click_time = 0.0
    last_double_click_time = 0.0
    last_right_click_time = 0.0

    print("\nAir Mouse Control active:")
    print("  - Index finger only up   -> move cursor")
    print("  - Index up + thumb pinch -> left click")
    print("  - Index + middle up      -> scroll mode (move hand up/down)")
    print("  - Index + middle pinch   -> double click")
    print("  - Index + middle + ring  -> right click")
    print("Press ESC in the camera window to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARNING] Failed to read a frame from the webcam; retrying...")
                continue
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            mode_text = "No hand detected"

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                index_up, middle_up, ring_up, pinky_up = _mouse_fingers_up(lm)

                # Hand size (wrist -> middle-finger-MCP) used to scale pinch
                # thresholds so they work regardless of hand distance from camera.
                hand_size = _mouse_pixel_dist(lm, 0, 9, frame_w, frame_h) + 1e-6
                thumb_index_dist = _mouse_pixel_dist(lm, 4, 8, frame_w, frame_h) / hand_size
                index_middle_dist = _mouse_pixel_dist(lm, 8, 12, frame_w, frame_h) / hand_size

                ix, iy = lm[8].x * frame_w, lm[8].y * frame_h  # index fingertip, pixel coords
                now = time.time()

                if index_up and middle_up and ring_up and not pinky_up:
                    # --- RIGHT CLICK ---
                    mode_text = "Right Click"
                    prev_scroll_y = None
                    if now - last_right_click_time > MOUSE_CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        last_right_click_time = now

                elif index_up and middle_up and not ring_up and not pinky_up:
                    # --- SCROLL MODE (+ double click via pinch) ---
                    mode_text = "Scroll"
                    if prev_scroll_y is not None:
                        dy = prev_scroll_y - iy  # moving hand up -> positive -> scroll up
                        if abs(dy) > MOUSE_SCROLL_DEAD_ZONE:
                            pyautogui.scroll(int(dy * MOUSE_SCROLL_SENSITIVITY))
                    prev_scroll_y = iy

                    if index_middle_dist < MOUSE_PINCH_CLICK_RATIO:
                        mode_text = "Double Click"
                        if now - last_double_click_time > MOUSE_CLICK_COOLDOWN:
                            pyautogui.doubleClick()
                            last_double_click_time = now

                elif index_up and not middle_up and not ring_up and not pinky_up:
                    # --- MOVE MODE (+ left click via pinch) ---
                    mode_text = "Move"
                    prev_scroll_y = None
                    x1 = np.interp(ix, (MOUSE_FRAME_REDUCTION, frame_w - MOUSE_FRAME_REDUCTION), (0, screen_w))
                    y1 = np.interp(iy, (MOUSE_FRAME_REDUCTION, frame_h - MOUSE_FRAME_REDUCTION), (0, screen_h))
                    curr_x = prev_x + (x1 - prev_x) / MOUSE_SMOOTHENING
                    curr_y = prev_y + (y1 - prev_y) / MOUSE_SMOOTHENING
                    curr_x = float(min(max(curr_x, 1), screen_w - 2))
                    curr_y = float(min(max(curr_y, 1), screen_h - 2))
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                    if thumb_index_dist < MOUSE_PINCH_CLICK_RATIO:
                        mode_text = "Left Click"
                        if now - last_click_time > MOUSE_CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = now
                else:
                    prev_scroll_y = None
                    mode_text = "Hand detected (no active mode)"
            else:
                prev_scroll_y = None

            cv2.putText(frame, f"Mode: {mode_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Air Mouse Control (ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# MENU
# ============================================================
def check_optional_dependencies():
    """Warn up front about anything that will be needed later, so failures
    happen at menu-time with a clear message instead of mid-gesture."""
    warnings = []
    try:
        import pyautogui  # noqa: F401
    except ImportError:
        warnings.append("pyautogui not installed - media key / volume actions won't work "
                         "(pip install pyautogui)")
    if platform.system() == "Windows":
        try:
            import pycaw  # noqa: F401
        except ImportError:
            warnings.append("pycaw not installed - volume will fall back to key simulation "
                             "(pip install pycaw comtypes for precise control)")
    if warnings:
        print("\n[Startup check]")
        for w in warnings:
            print(f"  - {w}")


def main():
    check_optional_dependencies()
    while True:
        print("\n" + "=" * 60)
        print("AI HAND GESTURE CONTROLLED MEDIA PLAYER (v3)")
        print("=" * 60)
        print("1) Collect Data                     (build your own dataset)")
        print("2) Train Model                      (angle-robust AI training)")
        print("3) Customize Gesture-Action Mapping (choose Media or Presentation mode)")
        print("4) Run Gesture Control               (choose Media or Presentation mode)")
        print("5) Manage Collected Data             (inspect / delete a gesture)")
        print("6) Air Mouse Control                 (control cursor with your hand)")
        print("7) Exit")
        choice = input("Choose an option (1-7): ").strip()
        if choice == "1":
            collect_data()
        elif choice == "2":
            train_model()
        elif choice == "3":
            customize_gestures()
        elif choice == "4":
            run_media_player()
        elif choice == "5":
            manage_data()
        elif choice == "6":
            run_mouse_control()
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Cleaning up...")
        cv2.destroyAllWindows()
        sys.exit(0)