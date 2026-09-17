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

UI_CLASS_COLORS = {
    'Normal':  "#22c55e", 
    'Forward': "#ef4444", 
    'Back':    "#eab308", 
}

UPPER_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 8), (9, 10), (11, 12),
]


# 2. CORE CLASSES (v2 Logic)
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

    # V2 Advanced Temporal Features
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


# 3. DESKTOP PET OVERLAY (Tkinter Port)
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


# 4. MAIN APP GUI (v4 Dashboard + v2 Logic)
class PostureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Postura Pro (v4 UI + v2 Core + Desktop Pet)")
        self.geometry("1050x600")
        self.minsize(900, 500)
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

        self.setup_ui()
        self.pet = DesktopPet(self)

        # Key bindings for feedback
        self.bind("<f>", self.trigger_feedback)
        self.bind("<y>", self.confirm_feedback)
        self.bind("<n>", self.cancel_feedback)
        self.bind("<Escape>", self.cancel_feedback)

        # Start Camera
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

        # XGBoost (Ensure this is the v1.3 file holding v2 logic!)
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
        if self.back_idx is not None: mask[self.back_idx] = False
        return int(np.argmax(proba_1d * mask))

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)
        
        self.video_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=15)
        self.sidebar.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.sidebar.grid_propagate(False) 
        
        ctk.CTkLabel(self.sidebar, text="Posture Dashboard", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(25, 20), padx=20, anchor="w")
        
        self.status_card = ctk.CTkFrame(self.sidebar, fg_color="#1e1e1e", corner_radius=10)
        self.status_card.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkLabel(self.status_card, text="CURRENT POSTURE", text_color="gray", font=ctk.CTkFont(size=12)).pack(pady=(15, 0))
        self.status_value = ctk.CTkLabel(self.status_card, text="Waiting...", font=ctk.CTkFont(size=32, weight="bold"))
        self.status_value.pack(pady=(0, 15))

        self.prob_card = ctk.CTkFrame(self.sidebar, fg_color="#1e1e1e", corner_radius=10)
        self.prob_card.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkLabel(self.prob_card, text="PROBABILITIES", text_color="gray", font=ctk.CTkFont(size=12)).pack(pady=(15, 10), anchor="w", padx=15)
        
        self.progress_bars = {}
        self.prob_labels = {}
        for cls in self.le.classes_:
            row_frame = ctk.CTkFrame(self.prob_card, fg_color="transparent")
            row_frame.pack(fill="x", padx=15, pady=5)
            ctk.CTkLabel(row_frame, text=f"{cls}", width=60, anchor="w").pack(side="left")
            
            pb = ctk.CTkProgressBar(row_frame, progress_color=UI_CLASS_COLORS.get(cls, "gray"), height=10)
            pb.set(0)
            pb.pack(side="left", fill="x", expand=True, padx=(10, 10))
            pct_lbl = ctk.CTkLabel(row_frame, text="0%", width=35, anchor="e")
            pct_lbl.pack(side="right")
            
            self.progress_bars[cls] = pb
            self.prob_labels[cls] = pct_lbl
        
        ctk.CTkFrame(self.prob_card, height=10, fg_color="transparent").pack() 

        # Lower Sidebar statuses
        self.timer_label = ctk.CTkLabel(self.sidebar, text="Bad Posture: 0s", text_color="gray", font=ctk.CTkFont(size=14, weight="bold"))
        self.timer_label.pack(side="bottom", pady=5)
        
        self.debug_label = ctk.CTkLabel(self.sidebar, text="Raw: -- | Smoothed: --", text_color="gray", font=ctk.CTkFont(size=11))
        self.debug_label.pack(side="bottom", pady=5)

        self.instruction_label = ctk.CTkLabel(self.sidebar, text="[F]=Feedback Mode", text_color="#3b82f6", font=ctk.CTkFont(size=12))
        self.instruction_label.pack(side="bottom", pady=10)


    def draw_skeleton(self, image, landmarks):
        h, w, _ = image.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks[:LANDMARK_COUNT]]
        for (a, b) in UPPER_CONNECTIONS:
            if a < LANDMARK_COUNT and b < LANDMARK_COUNT:
                cv2.line(image, pts[a], pts[b], (150, 150, 150), 2, cv2.LINE_AA)
        for cx, cy in pts:
            cv2.circle(image, (cx, cy), 5, (255, 200, 50), -1, cv2.LINE_AA)

    def draw_feedback_prompt(self, image, model_class: str):
        h, w = image.shape[:2]
        box_x, box_y = w // 2 - 230, h // 2 - 50
        cv2.rectangle(image, (box_x, box_y), (box_x + 460, box_y + 100), (20, 20, 20), -1)
        cv2.rectangle(image, (box_x, box_y), (box_x + 460, box_y + 100), (255, 255, 255), 1)
        cv2.putText(image, f"Model thinks: {model_class}", (box_x + 14, box_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, "Press Y = I'm actually fine (Normal)", (box_x + 14, box_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (94, 197, 34), 1, cv2.LINE_AA)
        cv2.putText(image, "Press N or ESC = Cancel", (box_x + 14, box_y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 163, 156), 1, cv2.LINE_AA)

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
            
            annotated = rgb_frame.copy()

            if result.pose_landmarks:
                pose_landmarks = result.pose_landmarks[0]
                self.draw_skeleton(annotated, pose_landmarks)

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

                    raw_class = self.le.inverse_transform([raw_pred])[0]
                    smooth_class = self.le.inverse_transform([smooth_pred])[0]
                    self.last_smooth_class = smooth_class

                    # Notify Logic
                    is_bad = smooth_class != 'Normal'
                    self.notifier.update(is_bad, smooth_class)
                    bad_secs = self.notifier.seconds_in_bad()

                    # Dashboard Update
                    self.status_value.configure(text=smooth_class.upper(), text_color=UI_CLASS_COLORS.get(smooth_class, "white"))
                    self.debug_label.configure(text=f"Raw: {raw_class} | Smoothed: {smooth_class}")
                    
                    if bad_secs > 0:
                        self.timer_label.configure(text=f"Bad Posture: {bad_secs:.0f}s", text_color="#ef4444")
                    else:
                        self.timer_label.configure(text="Bad Posture: 0s", text_color="gray")

                    for i, cls in enumerate(self.le.classes_):
                        p = proba[i]
                        self.progress_bars[cls].set(p)
                        self.prob_labels[cls].configure(text=f"{int(p*100)}%")

                    # Pet Update
                    self.pet.update_state(smooth_class)
                    
                    if self.feedback_mode:
                        self.draw_feedback_prompt(annotated, smooth_class)

            else:
                self.smoother.reset()
                self.frame_buffer.clear()
                self.notifier.update(False, 'Normal')
                self.pet.update_state("Normal")
                
                self.status_value.configure(text="NO POSE", text_color="gray")
                for cls in self.le.classes_:
                    self.progress_bars[cls].set(0)
                    self.prob_labels[cls].configure(text="0%")

            # Render to CustomTkinter
            frame_h, frame_w = annotated.shape[:2]
            label_w, label_h = self.video_frame.winfo_width(), self.video_frame.winfo_height()
            
            if label_w > 10 and label_h > 10:
                scale = min(label_w/frame_w, label_h/frame_h)
                new_w, new_h = int(frame_w * scale), int(frame_h * scale)
                
                pil_image = Image.fromarray(annotated)
                ctk_img = ctk.CTkImage(light_image=pil_image, size=(new_w, new_h))
                self.video_label.configure(image=ctk_img)
                self.video_label.image = ctk_img

        self.after(16, self.update_camera)

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = PostureApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()