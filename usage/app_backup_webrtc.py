# Postura - Stable CPU version
# Architecture: WebRTC streams -> shared frame buffer -> pose thread -> result queue -> Streamlit

import os
import time
import csv
import queue
import json
import threading
from collections import deque
import av
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import joblib
import urllib.request
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="Postura", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stProgress > div > div { border-radius: 999px; }
    .metric-card { background: #1e293b; border-radius: 12px;
                   padding: 16px 20px; margin-bottom: 12px; }
    .section-label { color: #64748b; font-size: 0.75rem; font-weight: 600;
                     letter-spacing: 0.08em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
LANDMARK_COUNT   = 13
MODEL_PKL_PATH   = os.path.join('model', 'posture_xgboost_baseline.pkl')
MP_MODEL_PATH    = 'pose_landmarker_lite.task'
FEEDBACK_LOG     = os.path.join('data', 'posture_feedback.csv')
BAD_POSTURE_SECS = 10
ALERT_COOLDOWN   = 60

POSE_INTERVAL     = 0.33      # run pose ~3x/sec
INFERENCE_MIN_INT = 0.20
DASHBOARD_REFRESH = 0.25
VIDEO_FPS         = 15
MAX_W             = 480
MAX_H             = 360
MP_INPUT          = 256
SMOOTHER_WINDOW   = 5

BANNER_COLOR = (200, 120, 0)
BANNER_TEXT  = (255, 255, 255)

raw_features = []
for i in range(1, LANDMARK_COUNT + 1):
    raw_features.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])


# ============================================================
# SHARED STATE (module-level)
# ============================================================
# Only keep the LATEST frame — no backlog, bounded memory
_LATEST_FRAME = {"img": None, "lock": threading.Lock()}
_RESULT_QUEUE = queue.Queue(maxsize=1)
_POSE_THREAD  = {"t": None, "stop": threading.Event(), "started": False}
_POSE_READY   = threading.Event()   # set true once MediaPipe is loaded


# ============================================================
# FEATURE ENGINEERING
# ============================================================
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
        feat['nose_to_shoulder_mid_x'] ** 2 + feat['nose_to_shoulder_mid_y'] ** 2)
    feat['shoulder_width'] = np.sqrt(
        (df['x12'] - df['x13']) ** 2 + (df['y12'] - df['y13']) ** 2)
    feat['left_ear_shoulder_y_diff']  = df['y8'] - df['y12']
    feat['right_ear_shoulder_y_diff'] = df['y9'] - df['y13']
    feat['ear_shoulder_y_asymmetry']  = (
        feat['left_ear_shoulder_y_diff'] - feat['right_ear_shoulder_y_diff'])
    feat['nose_left_ear_x_diff']  = df['x1'] - df['x8']
    feat['nose_right_ear_x_diff'] = df['x1'] - df['x9']
    feat['eye_level_diff']        = df['y3'] - df['y6']

    feat['relative_nose_z']   = df['z1'] - ((df['z12'] + df['z13']) / 2)
    feat['normalized_nose_z'] = feat['relative_nose_z'] / (feat['shoulder_width'] + 1e-4)
    feat['neck_forward_angle'] = np.degrees(np.arctan2(
        np.abs(feat['relative_nose_z']),
        np.abs(feat['nose_to_shoulder_mid_y']) + 1e-4))

    v_cols = [f'v{i}' for i in range(1, LANDMARK_COUNT + 1)]
    if all(c in df.columns for c in v_cols):
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
    return feat


class TemporalSmoother:
    def __init__(self, window=5):
        self.window  = window
        self.history = deque(maxlen=window)

    def update(self, pred):
        self.history.append(pred)
        counts = np.bincount(list(self.history), minlength=10)
        return int(np.argmax(counts))

    def reset(self):
        self.history.clear()


def log_feedback(model_class, user_class, feature_row):
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    exists = os.path.exists(FEEDBACK_LOG) and os.path.getsize(FEEDBACK_LOG) > 0
    row = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'model_said': model_class, 'user_said': user_class}
    row.update(feature_row.to_dict())
    with open(FEEDBACK_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# MODEL LOADING (main thread — do this BEFORE starting workers)
# ============================================================
@st.cache_resource
def load_xgboost():
    data = joblib.load(MODEL_PKL_PATH)
    return (data['model'], data['encoder'], data['normal_idx'],
            data['normal_threshold'], data['features'])


@st.cache_resource
def load_mediapipe():
    if not os.path.exists(MP_MODEL_PATH):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        urllib.request.urlretrieve(url, MP_MODEL_PATH)
    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


xgb_model, le, normal_idx, threshold, model_features = load_xgboost()
n_classes = len(le.classes_)
back_idx  = int(le.transform(['Back'])[0]) if 'Back' in le.classes_ else None
BACK_THRESHOLD = 0.95


def predict_with_threshold(proba):
    if back_idx is not None and proba[back_idx] >= BACK_THRESHOLD:
        return back_idx
    if proba[normal_idx] >= threshold:
        return normal_idx
    mask = np.ones(n_classes, dtype=bool)
    mask[normal_idx] = False
    if back_idx is not None:
        mask[back_idx] = False
    return int(np.argmax(proba * mask))


# ============================================================
# POSE WORKER THREAD
# ============================================================
def _pose_worker():
    """
    Runs in a dedicated thread. Loads MediaPipe INSIDE this thread,
    grabs the latest frame, runs pose + XGBoost, pushes result to _RESULT_QUEUE.
    """
    try:
        detector = load_mediapipe()
    except Exception as e:
        print(f"[pose_worker] MediaPipe load failed: {e}")
        return

    _POSE_READY.set()

    smoother     = TemporalSmoother(SMOOTHER_WINDOW)
    frame_buffer = deque(maxlen=3)
    last_infer   = 0.0
    last_result  = None

    while not _POSE_THREAD["stop"].is_set():
        t0 = time.time()

        # Grab latest frame
        with _LATEST_FRAME["lock"]:
            img = _LATEST_FRAME["img"]
            _LATEST_FRAME["img"] = None   # consume it

        if img is None:
            time.sleep(0.02)
            continue

        try:
            small = cv2.resize(img, (MP_INPUT, MP_INPUT),
                               interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                row = []
                for i in range(LANDMARK_COUNT):
                    p = lm[i]
                    row.extend([p.x, p.y, p.z, p.visibility])
                frame_buffer.append(row)

                if len(frame_buffer) == 3:
                    now = time.time()
                    if (now - last_infer) >= INFERENCE_MIN_INT:
                        last_infer = now
                        df = pd.DataFrame(list(frame_buffer), columns=raw_features)
                        feat = build_features(df).tail(1).reset_index(drop=True)
                        X = feat[model_features]
                        proba = xgb_model.predict_proba(X)[0]
                        raw_pred = predict_with_threshold(proba)
                        smooth_pred = smoother.update(raw_pred)
                        raw_class = le.inverse_transform([raw_pred])[0]
                        smooth_class = le.inverse_transform([smooth_pred])[0]

                        last_result = {
                            'smooth_class': smooth_class,
                            'raw_class': raw_class,
                            'feature_row': feat[model_features].iloc[0],
                        }
                        try:
                            _RESULT_QUEUE.put_nowait(last_result)
                        except queue.Full:
                            pass
            else:
                smoother.reset()
                frame_buffer.clear()
                last_result = None
                try:
                    _RESULT_QUEUE.put_nowait({
                        'smooth_class': None, 'raw_class': None,
                        'feature_row': None,
                    })
                except queue.Full:
                    pass

        except Exception as e:
            # Don't kill the thread on a single bad frame
            print(f"[pose_worker] frame error: {e}")

        # Throttle: aim for ~3 poses/sec
        elapsed = time.time() - t0
        time.sleep(max(0, POSE_INTERVAL - elapsed))


def ensure_pose_worker():
    if _POSE_THREAD["started"]:
        return
    _POSE_THREAD["stop"].clear()
    t = threading.Thread(target=_pose_worker, daemon=True)
    t.start()
    _POSE_THREAD["t"] = t
    _POSE_THREAD["started"] = True


# ============================================================
# WEBRTC PROCESSOR (LIGHT — just stores latest frame + draws banner)
# ============================================================
class LiteProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_class = 'Waiting'
        self.counter = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format='bgr24')

        # Store latest frame for the worker (non-blocking)
        with _LATEST_FRAME["lock"]:
            _LATEST_FRAME["img"] = img

        # Pull latest classification from worker (non-blocking)
        try:
            payload = _RESULT_QUEUE.get_nowait()
            if payload['smooth_class'] is not None:
                self.last_class = payload['smooth_class']
            else:
                self.last_class = 'No pose'
        except queue.Empty:
            pass

        # Draw banner
        out = img.copy()
        cv2.rectangle(out, (10, 10), (280, 52), BANNER_COLOR, -1)
        cv2.putText(out, f"STATUS: {self.last_class}",
                    (18, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, BANNER_TEXT, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(out, format='bgr24')


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    'smooth_class': 'Waiting',
    'raw_class': 'Waiting',
    'last_feature_row': None,
    'bad_since': None,
    'last_alert': 0.0,
    'alerted_streak': False,
    'bad_seconds': 0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# NOTIFICATIONS
# ============================================================
def update_notifier(smooth_class, alert_secs):
    now = time.time()
    is_bad = smooth_class not in (None, 'Normal', 'Waiting', 'No pose')
    if not is_bad:
        st.session_state.bad_since = None
        st.session_state.alerted_streak = False
        st.session_state.bad_seconds = 0.0
        return False
    if st.session_state.bad_since is None:
        st.session_state.bad_since = now
    elapsed = now - st.session_state.bad_since
    st.session_state.bad_seconds = elapsed
    cooldown_ok = (now - st.session_state.last_alert) >= ALERT_COOLDOWN
    if elapsed >= alert_secs and cooldown_ok and not st.session_state.alerted_streak:
        st.session_state.last_alert = now
        st.session_state.alerted_streak = True
        return True
    return False


def send_desktop_notification(title: str, body: str):
    js = f"""
    <script>
    (function() {{
        const title = {json.dumps(title)};
        const body  = {json.dumps(body)};
        function fire() {{
            if (Notification.permission === "granted") {{
                new Notification(title, {{ body: body,
                    tag: "postura-alert", requireInteraction: true }});
            }}
        }}
        if (!("Notification" in window)) return;
        if (Notification.permission === "granted") fire();
        else if (Notification.permission !== "denied") {{
            Notification.requestPermission().then(p => {{ if (p === "granted") fire(); }});
        }}
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Postura")
    st.divider()

    st.markdown("### Thresholds")
    st.slider("Normal threshold", 0.40, 0.90, float(threshold), 0.05)
    st.slider("Back threshold", 0.70, 1.00, BACK_THRESHOLD, 0.05)

    st.divider()
    st.markdown("### Alert settings")
    alert_secs = st.slider("Alert after (seconds)", 3, 60, BAD_POSTURE_SECS, 1)

    st.divider()
    st.markdown("### Notifications")
    if st.button("Enable Desktop Notifications", use_container_width=True):
        send_desktop_notification(
            "Postura Notifications Enabled",
            "You'll be alerted here when your posture needs fixing.")
        st.success("Permission requested.")
    if st.button("Test Notification", use_container_width=True):
        send_desktop_notification("Test Alert", "Posture alert preview.")
        st.info("Test notification sent.")

    st.divider()
    st.markdown("### Feedback")
    if st.button("I'm sitting fine right now", use_container_width=True):
        if (st.session_state.last_feature_row is not None
                and st.session_state.smooth_class not in (None, 'Waiting')):
            log_feedback(st.session_state.smooth_class, 'Normal',
                         st.session_state.last_feature_row)
            st.success("Feedback logged.")
        else:
            st.warning("No active prediction to log yet.")

    if os.path.exists(FEEDBACK_LOG):
        fb_df = pd.read_csv(FEEDBACK_LOG)
        st.caption(f"{len(fb_df)} feedback records logged")

    st.divider()
    st.caption(f"Classes: {', '.join(le.classes_)}")


# ============================================================
# MAIN
# ============================================================
st.markdown("## Real-Time Posture Detection")

col_video, col_dash = st.columns([3, 2], gap="large")

with col_video:
    # Start the pose worker ONCE (daemon thread, safe)
    ensure_pose_worker()

    ctx = webrtc_streamer(
        key="postura",
        video_processor_factory=LiteProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={
            "video": {
                "width":  {"max": MAX_W},
                "height": {"max": MAX_H},
                "frameRate": {"ideal": VIDEO_FPS, "max": VIDEO_FPS},
            },
            "audio": False,
        },
        async_processing=True,
    )


@st.fragment(run_every=DASHBOARD_REFRESH)
def dashboard():
    try:
        payload = _RESULT_QUEUE.get_nowait()
        if payload['smooth_class'] is not None:
            st.session_state.smooth_class = payload['smooth_class']
            st.session_state.raw_class = payload['raw_class']
            st.session_state.last_feature_row = payload['feature_row']
        else:
            st.session_state.smooth_class = 'No pose'
    except queue.Empty:
        pass

    if update_notifier(st.session_state.smooth_class, alert_secs):
        send_desktop_notification(
            "Postura Alert",
            f"Fix your posture! Detected: {st.session_state.smooth_class}")

    bad_secs = st.session_state.bad_seconds
    if bad_secs > 0:
        ratio = min(bad_secs / alert_secs, 1.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="section-label">Bad posture duration</div>
            <div style="color:#fb923c;font-size:1.4rem;font-weight:700">
                {bad_secs:.0f}s / {alert_secs}s
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(ratio)


with col_dash:
    dashboard()