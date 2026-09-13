import sys
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
from PIL import Image
import customtkinter as ctk

try:
    from plyer import notification as plyer_notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    print("plyer not found -- falling back to console alert.\n")


# 1. CONFIG & CONSTANTS
BAD_POSTURE_ALERT_SECONDS = 10
ALERT_COOLDOWN_SECONDS    = 60
FEEDBACK_LOG_PATH         = 'data/posture_feedback.csv'
LANDMARK_COUNT            = 13

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


# 2. CORE CLASSES
class TemporalSmoother:
    def __init__(self, window: int = 15):
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


class PostureNotifier:
    def __init__(self, alert_after_seconds: float, cooldown_seconds: float):
        self.alert_after   = alert_after_seconds
        self.cooldown      = cooldown_seconds
        self._bad_since    = None
        self._last_alert   = 0.0
        self._alerted_this_streak = False

    def update(self, is_bad: bool, current_class: str) -> bool:
        now = time.time()
        if not is_bad:
            self._bad_since = None
            self._alerted_this_streak = False
            return False

        if self._bad_since is None:
            self._bad_since = now

        elapsed = now - self._bad_since
        cooldown_clear = (now - self._last_alert) >= self.cooldown

        if elapsed >= self.alert_after and cooldown_clear and not self._alerted_this_streak:
            self._last_alert = now
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
            plyer_notification.notify(title='Postura', message=msg, app_name='Postura', timeout=5)
        else:
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass


class FeedbackLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
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
    feat['nose_to_shoulder_mid_dist'] = np.sqrt(feat['nose_to_shoulder_mid_x'] ** 2 + feat['nose_to_shoulder_mid_y'] ** 2)
    feat['shoulder_width'] = np.sqrt((df['x12'] - df['x13']) ** 2 + (df['y12'] - df['y13']) ** 2)

    feat['left_ear_shoulder_y_diff']  = df['y8']  - df['y12']
    feat['right_ear_shoulder_y_diff'] = df['y9']  - df['y13']
    feat['ear_shoulder_y_asymmetry']  = feat['left_ear_shoulder_y_diff'] - feat['right_ear_shoulder_y_diff']

    feat['nose_left_ear_x_diff']  = df['x1'] - df['x8']
    feat['nose_right_ear_x_diff'] = df['x1'] - df['x9']
    feat['eye_level_diff']        = df['y3'] - df['y6']

    feat['relative_nose_z']    = df['z1'] - ((df['z12'] + df['z13']) / 2)
    feat['normalized_nose_z']  = feat['relative_nose_z'] / (feat['shoulder_width'] + 0.0001)
    feat['neck_forward_angle'] = np.degrees(np.arctan2(np.abs(feat['relative_nose_z']), np.abs(feat['nose_to_shoulder_mid_y']) + 0.0001))

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


# 3. DESKTOP PET OVERLAY (the only visible UI)
class DesktopPet(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes("-topmost", True)

        # Transparent background for Windows
        transparent_color = "#000001"
        self.configure(fg_color=transparent_color)
        try:
            self.wm_attributes("-transparentcolor", transparent_color)
        except Exception:
            pass

        self.geometry("+150+150")

        # Dragging variables
        self._offset_x = 0
        self._offset_y = 0
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)

        # Chat Bubble
        self.bubble = ctk.CTkLabel(
            self, text="", text_color="#dc2626", fg_color="white",
            corner_radius=12, font=ctk.CTkFont(size=14, weight="bold")
        )
        self.bubble.pack(pady=(0, 5))
        self.bubble.pack_forget()

        # Dynamic State Images
        self.images = {}
        image_files = {
            'Normal': 'assets/udang_normal.png',
            'Forward': 'assets/udang_forward.png',
            'Back': 'assets/udang_backward.png'
        }
        for state, path in image_files.items():
            if os.path.exists(path):
                img = Image.open(path)
                wpercent = (180 / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((180, hsize), Image.LANCZOS)
                self.images[state] = ctk.CTkImage(light_image=img, size=(180, hsize))

        self.image_label = ctk.CTkLabel(self, text="")
        if 'Normal' in self.images:
            self.image_label.configure(image=self.images['Normal'])
        else:
            self.image_label.configure(text="[Pet Image Missing]", text_color="white")
        self.image_label.pack()

    def on_press(self, event):
        self._offset_x = event.x
        self._offset_y = event.y

    def on_drag(self, event):
        x = self.winfo_pointerx() - self._offset_x
        y = self.winfo_pointery() - self._offset_y
        self.geometry(f"+{x}+{y}")

    def update_state(self, posture_class: str):
        if posture_class in self.images:
            self.image_label.configure(image=self.images[posture_class])

        if posture_class != "Normal":
            self.bubble.configure(text=f"⚠️ Fix Posture!\n({posture_class})")
            self.bubble.pack(pady=(0, 5), before=self.image_label)
        else:
            self.bubble.pack_forget()


# 4. MAIN APP (headless camera loop + pet only, no dashboard window)

class PostureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.withdraw()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialization logic
        self.init_models()
        self.smoother     = TemporalSmoother(window=15)
        self.frame_buffer = deque(maxlen=3)
        self.notifier     = PostureNotifier(BAD_POSTURE_ALERT_SECONDS, ALERT_COOLDOWN_SECONDS)
        self.feedback_log = FeedbackLogger(FEEDBACK_LOG_PATH)

        self.feedback_mode     = False
        self.last_smooth_class = 'Normal'
        self.last_feature_row  = None

        self.pet = DesktopPet(self)

    
        self.pet.bind("<f>", self.trigger_feedback)
        self.pet.bind("<y>", self.confirm_feedback)
        self.pet.bind("<n>", self.cancel_feedback)
        self.pet.bind("<Escape>", self.cancel_feedback)
        self.pet.focus_set()

    
        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def init_models(self):
        # MediaPipe
        model_path = 'pose_landmarker_lite.task'
        if not os.path.exists(model_path):
            print("Downloading MediaPipe model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False, num_poses=1)
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # XGBoost (v1.3 file holding the feature-selected logic)
        model_data = joblib.load('model/posture_xgboost_v1.3.pkl')
        self.model = model_data['model']
        self.le    = model_data['encoder']
        self.normal_idx = model_data['normal_idx']
        self.threshold  = model_data['normal_threshold']
        self.back_idx   = int(self.le.transform(['Back'])[0]) if 'Back' in self.le.classes_ else None
        self.BACK_THRESHOLD = 0.95
        self.n_classes  = len(self.le.classes_)

    def predict_with_threshold(self, proba_1d: np.ndarray) -> int:
        if self.back_idx is not None and proba_1d[self.back_idx] >= self.BACK_THRESHOLD:
            return self.back_idx
        if proba_1d[self.normal_idx] >= self.threshold:
            return self.normal_idx
        mask = np.ones(self.n_classes, dtype=bool)
        mask[self.normal_idx] = False
        if self.back_idx is not None:
            mask[self.back_idx] = False
        return int(np.argmax(proba_1d * mask))

    # Keybind handlers
    def trigger_feedback(self, event=None):
        self.feedback_mode = True

    def confirm_feedback(self, event=None):
        if self.feedback_mode and self.last_feature_row is not None:
            self.feedback_log.log(model_class=self.last_smooth_class, user_class='Normal', feature_row=self.last_feature_row)
            self.smoother.reset()
            for _ in range(self.smoother.window):
                self.smoother.update(self.normal_idx)
            self.feedback_mode = False

    def cancel_feedback(self, event=None):
        self.feedback_mode = False

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)

            if result.pose_landmarks:
                pose_landmarks = result.pose_landmarks[0]

                row = []
                for i in range(LANDMARK_COUNT):
                    lm = pose_landmarks[i]
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                self.frame_buffer.append(row)

                if len(self.frame_buffer) == 3:
                    landmark_df = pd.DataFrame(list(self.frame_buffer), columns=raw_features)
                    X_live = build_features(landmark_df).tail(1).reset_index(drop=True)
                    self.last_feature_row = X_live.iloc[0]

                    proba = self.model.predict_proba(X_live)[0]
                    raw_pred = self.predict_with_threshold(proba)
                    smooth_pred = self.smoother.update(raw_pred)

                    smooth_class = self.le.inverse_transform([smooth_pred])[0]
                    self.last_smooth_class = smooth_class

                    is_bad = smooth_class != 'Normal'
                    self.notifier.update(is_bad, smooth_class)

                    self.pet.update_state(smooth_class)

            else:
                self.smoother.reset()
                self.frame_buffer.clear()
                self.notifier.update(False, 'Normal')
                self.pet.update_state("Normal")

        self.after(16, self.update_camera)

    def on_closing(self):
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = PostureApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()