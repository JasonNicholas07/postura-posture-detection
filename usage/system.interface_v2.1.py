# Postura - Inference with Notification + Feedback
#
# Notification: triggers after bad posture holds for BAD_POSTURE_ALERT_SECONDS.
#               Cooldown prevents re-alerting until posture recovers and worsens again.
# Feedback:     Press F during live view to say "I'm actually sitting fine right now".
#               Feedback is logged to posture_feedback.csv for future retraining.
# Notification falls back to a console beep if plyer is unavailable.

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import joblib
import urllib.request
import os
import time
import csv
from collections import deque
from plyer import notification as plyer_notification

try:
    from plyer import notification as plyer_notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    print("plyer not found -- install with: pip install plyer")
    print("Falling back to console alert.\n")


# CONFIG
BAD_POSTURE_ALERT_SECONDS = 10   # alert fires after this many continuous seconds of bad posture
ALERT_COOLDOWN_SECONDS    = 60   # minimum gap between repeated alerts
FEEDBACK_LOG_PATH         = 'data/posture_feedback.csv'


# TEMPORAL SMOOTHER
class TemporalSmoother:
    def __init__(self, window: int = 10):
        self.window  = window
        self.history = []

    def update(self, prediction: int) -> int:
        self.history.append(prediction)
        if len(self.history) > self.window:
            self.history.pop(0)
        counts = np.bincount(self.history)
        return int(np.argmax(counts))

    def reset(self):
        self.history = []


# NOTIFICATION
class PostureNotifier:
    def __init__(self, alert_after_seconds: float, cooldown_seconds: float):
        self.alert_after   = alert_after_seconds
        self.cooldown      = cooldown_seconds
        self._bad_since    = None   # timestamp when current bad streak started
        self._last_alert   = 0.0   # timestamp of last fired alert
        self._alerted_this_streak = False

    def update(self, is_bad: bool, current_class: str) -> bool:
        """
        Call every frame with whether the current smoothed prediction is bad.
        Returns True the moment an alert should fire (once per streak).
        """
        now = time.time()

        if not is_bad:
            self._bad_since = None
            self._alerted_this_streak = False
            return False

        if self._bad_since is None:
            self._bad_since = now

        elapsed        = now - self._bad_since
        cooldown_clear = (now - self._last_alert) >= self.cooldown

        if elapsed >= self.alert_after and cooldown_clear and not self._alerted_this_streak:
            self._last_alert          = now
            self._alerted_this_streak = True
            self._fire(current_class)
            return True

        return False

    def seconds_in_bad(self) -> float:
        if self._bad_since is None:
            return 0.0
        return time.time() - self._bad_since

    def _fire(self, current_class: str):
        msg = f"Fix it up you shrimp!"
        if PLYER_AVAILABLE:
            plyer_notification.notify(
                title='Postura',
                message=msg,
                app_name='Postura',
                timeout=5,
            )
        else:
            print(f"\n[ALERT] {msg}")
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass


# FEEDBACK LOGGER
class FeedbackLogger:
    def __init__(self, path: str):
        self.path    = path
        self._header_written = os.path.exists(path) and os.path.getsize(path) > 0

    def log(self, model_class: str, user_class: str, feature_row: pd.Series):
        row = {
            'timestamp':   time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_said':  model_class,
            'user_said':   user_class,
        }
        row.update(feature_row.to_dict())

        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

        print(f"[Feedback] Logged: model='{model_class}' -> user='{user_class}'")


# FEATURE ENGINEERING  (exact copy of training build_features)
LANDMARK_COUNT = 13

raw_features = []
for i in range(1, LANDMARK_COUNT + 1):
    raw_features.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])

SELECTED_FEATURES = [
    'y2', 'y5', 'z11', 'z12',
    'y2_y5_ratio', 'z11_z12_diff', 'z11_z12_ratio',
    'y2_diff', 'y5_diff', 'z11_diff', 'z12_diff',
    'y2_moving_avg', 'y5_moving_avg', 'z11_moving_avg', 'z12_moving_avg',
    'y2_normalized', 'z11_normalized',
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    feat['y2']  = df['y2']
    feat['y5']  = df['y5']
    feat['z11'] = df['z11']
    feat['z12'] = df['z12']

    shoulder_mid_x = (df['x12'] + df['x13']) / 2
    shoulder_mid_y = (df['y12'] + df['y13']) / 2

    feat['nose_to_shoulder_mid_x']    = df['x1'] - shoulder_mid_x
    feat['nose_to_shoulder_mid_y']    = df['y1'] - shoulder_mid_y
    feat['nose_to_shoulder_mid_dist'] = np.sqrt(
        feat['nose_to_shoulder_mid_x'] ** 2 + feat['nose_to_shoulder_mid_y'] ** 2
    )
    feat['shoulder_width'] = np.sqrt(
        (df['x12'] - df['x13']) ** 2 + (df['y12'] - df['y13']) ** 2
    )
    feat['left_ear_shoulder_y_diff']  = df['y8']  - df['y12']
    feat['right_ear_shoulder_y_diff'] = df['y9']  - df['y13']
    feat['ear_shoulder_y_asymmetry']  = (
        feat['left_ear_shoulder_y_diff'] - feat['right_ear_shoulder_y_diff']
    )
    feat['nose_left_ear_x_diff']  = df['x1'] - df['x8']
    feat['nose_right_ear_x_diff'] = df['x1'] - df['x9']
    feat['eye_level_diff']        = df['y3'] - df['y6']

    feat['relative_nose_z']    = df['z1'] - ((df['z12'] + df['z13']) / 2)
    feat['normalized_nose_z']  = feat['relative_nose_z'] / (feat['shoulder_width'] + 0.0001)
    feat['neck_forward_angle'] = np.degrees(
        np.arctan2(
            np.abs(feat['relative_nose_z']),
            np.abs(feat['nose_to_shoulder_mid_y']) + 0.0001
        )
    )

    v_cols = [f'v{i}' for i in range(1, LANDMARK_COUNT + 1)]
    if all(col in df.columns for col in v_cols):
        feat['mean_visibility'] = df[v_cols].mean(axis=1)
        feat['min_visibility']  = df[v_cols].min(axis=1)

    feat['y2_y5_ratio']   = df['y2'] / (df['y5'] + 1e-6)
    feat['z11_z12_diff']  = df['z11'] - df['z12']
    feat['z11_z12_ratio'] = df['z11'] / (df['z12'] + 1e-6)

    feat['y2_diff']  = df['y2'].diff().fillna(0)
    feat['y5_diff']  = df['y5'].diff().fillna(0)
    feat['z11_diff'] = df['z11'].diff().fillna(0)
    feat['z12_diff'] = df['z12'].diff().fillna(0)

    feat['y2_moving_avg']  = df['y2'].rolling(window=3).mean().bfill()
    feat['y5_moving_avg']  = df['y5'].rolling(window=3).mean().bfill()
    feat['z11_moving_avg'] = df['z11'].rolling(window=3).mean().bfill()
    feat['z12_moving_avg'] = df['z12'].rolling(window=3).mean().bfill()

    feat['y2_normalized']  = df['y2'] / (df['y5'] + 1e-6)
    feat['z11_normalized'] = df['z11'] / (df['z12'] + 1e-6)

    return feat[SELECTED_FEATURES]


# 1. MEDIAPIPE
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Mengambil model MediaPipe dari Google APIs...")
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
    urllib.request.urlretrieve(url, model_path)
    print("Model berhasil diunduh!\n")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)


# 2. LOAD PKL
print("XGBoost loading...")
model_data   = joblib.load('posture_xgboost_v1.3.pkl')
model        = model_data['model']
le           = model_data['encoder']
normal_idx   = model_data['normal_idx']
threshold    = model_data['normal_threshold']
back_idx     = int(le.transform(['Back'])[0]) if 'Back' in le.classes_ else None
BACK_THRESHOLD = 0.95

n_classes    = len(le.classes_)
smoother     = TemporalSmoother(window=15)
frame_buffer = deque(maxlen=3)
notifier     = PostureNotifier(BAD_POSTURE_ALERT_SECONDS, ALERT_COOLDOWN_SECONDS)
feedback_log = FeedbackLogger(FEEDBACK_LOG_PATH)

model_features = list(model.feature_names_in_)
if model_features != SELECTED_FEATURES:
    print("WARNING: feature mismatch between script and model.")
    print(f"  Model expects : {model_features}")
    print(f"  Script defines: {SELECTED_FEATURES}")

print(f"Classes: {list(le.classes_)}  |  Normal threshold: {threshold}")
print(f"Alert after: {BAD_POSTURE_ALERT_SECONDS}s of bad posture")
print("Controls: F = feedback (I'm actually fine)  |  Q = quit\n")


# 3. THRESHOLD PREDICTION
def predict_with_threshold(proba_1d: np.ndarray) -> int:
    if back_idx is not None and proba_1d[back_idx] >= BACK_THRESHOLD:
        return back_idx
    if proba_1d[normal_idx] >= threshold:
        return normal_idx
    mask = np.ones(n_classes, dtype=bool)
    mask[normal_idx] = False
    if back_idx is not None:
        mask[back_idx] = False
    return int(np.argmax(proba_1d * mask))


# 4. UI HELPERS
UPPER_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3),
    (4, 5), (5, 6), (7, 8), (9, 10), (11, 12),
]
CLASS_COLORS = {
    'Normal':  (34,  197,  94),
    'Forward': (239,  68,  68),
    'Back':    (234, 179,   8),
}
DEFAULT_COLOR = (156, 163, 175)


def get_color(class_name: str):
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def draw_skeleton(image, landmarks):
    h, w, _ = image.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks[:LANDMARK_COUNT]]
    for (a, b) in UPPER_CONNECTIONS:
        if a < LANDMARK_COUNT and b < LANDMARK_COUNT:
            cv2.line(image, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
    for i, (cx, cy) in enumerate(pts):
        vis   = landmarks[i].visibility
        alpha = max(0.3, min(1.0, vis))
        color = tuple(int(c * alpha) for c in (255, 230, 80))
        cv2.circle(image, (cx, cy), 6, color, -1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 6, (80, 80, 80), 1, cv2.LINE_AA)


def draw_status_panel(image, pred_class: str, proba: np.ndarray,
                      raw_class: str, smoothed_class: str,
                      bad_seconds: float):
    margin  = 16
    panel_w = 320
    panel_h = 185

    overlay = image.copy()
    cv2.rectangle(overlay, (margin, margin),
                  (margin + panel_w, margin + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    color = get_color(pred_class)
    cv2.rectangle(image, (margin, margin),
                  (margin + panel_w, margin + 44), color, -1)
    cv2.putText(image, f"STATUS: {pred_class}",
                (margin + 10, margin + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    bar_x       = margin + 10
    bar_y_start = margin + 58
    bar_max_w   = panel_w - 20
    bar_h       = 16

    for i, cls in enumerate(le.classes_):
        prob   = proba[i]
        bcolor = get_color(cls)
        y      = bar_y_start + i * 34

        cv2.putText(image, cls, (bar_x, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(image, (bar_x + 75, y),
                      (bar_x + 75 + bar_max_w - 75, y + bar_h), (60, 60, 60), -1)
        filled = int((bar_max_w - 75) * prob)
        cv2.rectangle(image, (bar_x + 75, y),
                      (bar_x + 75 + filled, y + bar_h), bcolor, -1)
        cv2.putText(image, f"{prob * 100:.1f}%",
                    (bar_x + 75 + (bar_max_w - 75) + 4, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Bad posture timer bar
    if bad_seconds > 0:
        timer_y = margin + panel_h - 28
        ratio   = min(bad_seconds / BAD_POSTURE_ALERT_SECONDS, 1.0)
        bar_w   = panel_w - 20
        filled_w = int(bar_w * ratio)
        timer_color = (0, 165, 255) if ratio < 1.0 else (0, 0, 220)

        cv2.rectangle(image, (bar_x, timer_y), (bar_x + bar_w, timer_y + 10), (60, 60, 60), -1)
        cv2.rectangle(image, (bar_x, timer_y), (bar_x + filled_w, timer_y + 10), timer_color, -1)
        cv2.putText(image, f"Bad posture: {bad_seconds:.0f}s",
                    (bar_x, timer_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    footer_y = margin + panel_h - 6
    cv2.putText(image,
                f"raw:{raw_class}  smooth:{smoothed_class}  thr:{threshold}  [F]=feedback",
                (margin + 6, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)


def draw_feedback_prompt(image, model_class: str):
    h, w = image.shape[:2]
    box_x, box_y = w // 2 - 230, h // 2 - 50
    cv2.rectangle(image, (box_x, box_y), (box_x + 460, box_y + 100), (20, 20, 20), -1)
    cv2.rectangle(image, (box_x, box_y), (box_x + 460, box_y + 100), (255, 255, 255), 1)
    cv2.putText(image, f"Model thinks: {model_class}",
                (box_x + 14, box_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(image, "Press Y = I'm actually fine (Normal)",
                (box_x + 14, box_y + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (34, 197, 94), 1, cv2.LINE_AA)
    cv2.putText(image, "Press N = Cancel",
                (box_x + 14, box_y + 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (156, 163, 175), 1, cv2.LINE_AA)


def draw_no_pose(image):
    cv2.putText(image, "No pose detected", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2, cv2.LINE_AA)


# 5. CAMERA LOOP
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Can't open camera")

print("Press Q to quit.")

# State
feedback_mode     = False   # True while the Y/N prompt is on screen
last_smooth_class = 'Normal'
last_proba        = None
last_feature_row  = None   # pd.Series of the last frame's features, for logging

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed reading frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)
    annotated = frame.copy()

    if result.pose_landmarks:
        pose_landmarks = result.pose_landmarks[0]

        row = []
        for i in range(LANDMARK_COUNT):
            lm = pose_landmarks[i]
            row.extend([lm.x, lm.y, lm.z, lm.visibility])

        frame_buffer.append(row)

        if len(frame_buffer) == 3:
            landmark_df    = pd.DataFrame(list(frame_buffer), columns=raw_features)
            X_live         = build_features(landmark_df).tail(1).reset_index(drop=True)
            last_feature_row = X_live.iloc[0]

            proba          = model.predict_proba(X_live)[0]
            raw_pred       = predict_with_threshold(proba)
            smooth_pred    = smoother.update(raw_pred)

            raw_class      = le.inverse_transform([raw_pred])[0]
            smooth_class   = le.inverse_transform([smooth_pred])[0]
            last_smooth_class = smooth_class
            last_proba        = proba

            is_bad = smooth_class != 'Normal'
            notifier.update(is_bad, smooth_class)
            bad_secs = notifier.seconds_in_bad()

            draw_skeleton(annotated, pose_landmarks)
            draw_status_panel(annotated, smooth_class, proba,
                              raw_class, smooth_class, bad_secs)

            if feedback_mode:
                draw_feedback_prompt(annotated, smooth_class)

        else:
            cv2.putText(annotated, "Warming up...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            draw_skeleton(annotated, pose_landmarks)

    else:
        smoother.reset()
        frame_buffer.clear()
        notifier.update(False, 'Normal')
        draw_no_pose(annotated)

    cv2.imshow('Postura', annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('f') and not feedback_mode:
        # Open feedback prompt
        feedback_mode = True

    elif feedback_mode:
        if key == ord('y') and last_feature_row is not None:
            # User says they are actually in Normal posture
            feedback_log.log(
                model_class=last_smooth_class,
                user_class='Normal',
                feature_row=last_feature_row,
            )
            # Override smoother so display immediately reflects correction
            smoother.reset()
            for _ in range(smoother.window):
                smoother.update(normal_idx)
            feedback_mode = False

        elif key == ord('n') or key == 27:   # N or Escape cancels
            feedback_mode = False

cap.release()
cv2.destroyAllWindows()
print("Done")