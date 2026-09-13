# Postura - Real-Time Posture Detection (Browser-Side Inference)
# All pose detection AND classification happen in the browser.
# Python only handles the timer, notifications, and feedback logging.

import os
import time
import csv
import json
import streamlit as st
import streamlit.components.v1 as components
from postura_pose import postura_pose


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Postura",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stProgress > div > div { border-radius: 999px; }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .section-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

FEEDBACK_LOG     = os.path.join(_ROOT, "data", "posture_feedback.csv")
BAD_POSTURE_SECS = 10
ALERT_COOLDOWN   = 60

# Class labels from the LabelEncoder used during training
LE_CLASSES = ["Back", "Forward", "Normal"]


# ============================================================
# HELPERS
# ============================================================
def log_feedback(model_class, user_class, feature_row=None):
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    exists = os.path.exists(FEEDBACK_LOG) and os.path.getsize(FEEDBACK_LOG) > 0
    row = {
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_said": model_class,
        "user_said":  user_class,
    }
    if feature_row is not None:
        row.update(feature_row.to_dict())
    with open(FEEDBACK_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "camera_enabled":   False,
    "smooth_class":     "Waiting",
    "raw_class":        "Waiting",
    "last_feature_row": None,
    "bad_since":        None,
    "last_alert":       0.0,
    "alerted_streak":   False,
    "bad_seconds":      0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# NOTIFICATION LOGIC
# ============================================================
def update_notifier(smooth_class, alert_secs):
    now = time.time()
    is_bad = smooth_class not in (None, "Normal", "Waiting", "No pose")

    if not is_bad:
        st.session_state.bad_since      = None
        st.session_state.alerted_streak = False
        st.session_state.bad_seconds    = 0.0
        return False

    if st.session_state.bad_since is None:
        st.session_state.bad_since = now

    elapsed = now - st.session_state.bad_since
    st.session_state.bad_seconds = elapsed

    cooldown_ok = (now - st.session_state.last_alert) >= ALERT_COOLDOWN

    if (elapsed >= alert_secs and cooldown_ok
            and not st.session_state.alerted_streak):
        st.session_state.last_alert     = now
        st.session_state.alerted_streak = True
        return True

    return False


def send_desktop_notification(title: str, body: str):
    """Fire a browser desktop notification (works when tab is in the background)."""
    js = f"""
    <script>
    (function() {{
        const title = {json.dumps(title)};
        const body  = {json.dumps(body)};

        function fire() {{
            if (Notification.permission === "granted") {{
                new Notification(title, {{
                    body: body,
                    tag: "postura-alert",
                    requireInteraction: true
                }});
            }}
        }}

        if (!("Notification" in window)) return;

        if (Notification.permission === "granted") {{
            fire();
        }} else if (Notification.permission !== "denied") {{
            Notification.requestPermission().then(p => {{
                if (p === "granted") fire();
            }});
        }}
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


# ============================================================
# VIDEO FRAGMENT (isolates the component from full-page reruns)
# ============================================================
@st.fragment
def video_fragment():
    result = postura_pose(key="postura")

    if result and result.get("posture_class"):
        st.session_state.smooth_class = result["posture_class"]
        st.session_state.raw_class    = result["posture_class"]
    else:
        st.info("Waiting for pose data...")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Postura")
    st.divider()

    st.markdown("### Alert settings")
    alert_secs = st.slider("Alert after (seconds)", 3, 60, BAD_POSTURE_SECS, 1)

    st.divider()
    st.markdown("### Notifications")
    st.caption(
        "Enable desktop notifications to get alerts even when this tab is in the background."
    )
    if st.button("Enable Desktop Notifications", use_container_width=True):
        send_desktop_notification(
            "Postura Notifications Enabled",
            "You'll be alerted here when your posture needs fixing.",
        )
        st.success("Permission requested. Check your browser's notification prompt.")

    if st.button("Test Notification", use_container_width=True):
        send_desktop_notification(
            "Test Alert",
            "This is what a posture alert looks like.",
        )
        st.info("Test notification sent.")

    st.divider()
    st.markdown("### Feedback")
    if st.button("I'm sitting fine right now", use_container_width=True):
        if st.session_state.smooth_class not in (None, "Waiting"):
            log_feedback(
                model_class=st.session_state.smooth_class,
                user_class="Normal",
            )
            st.success("Feedback logged. Thank you!")
        else:
            st.warning("No active prediction to log yet.")

    if os.path.exists(FEEDBACK_LOG):
        try:
            fb_df = __import__("pandas").read_csv(FEEDBACK_LOG)
            st.caption(f"{len(fb_df)} feedback records logged")
        except Exception:
            pass

    st.divider()
    st.caption(f"Classes: {', '.join(LE_CLASSES)}")


# ============================================================
# MAIN LAYOUT
# ============================================================
st.markdown("## Real-Time Posture Detection")

col_video, col_dash = st.columns([3, 2], gap="large")

with col_video:
    if not st.session_state.camera_enabled:
        st.info("Click **Start Camera** to grant webcam access and begin detection.")
        if st.button("Start Camera", type="primary", use_container_width=True):
            st.session_state.camera_enabled = True
            st.rerun()
    else:
        video_fragment()


with col_dash:
    smooth = st.session_state.get("smooth_class", "Waiting")
    raw    = st.session_state.get("raw_class", "Waiting")

    # Fire desktop notification if threshold reached
    if update_notifier(smooth, alert_secs):
        send_desktop_notification(
            "Postura Alert",
            f"Fix your posture! Detected: {smooth}",
        )

    # === Current prediction ===
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="section-label">Current posture</div>
            <div style="font-size:1.6rem;font-weight:800;margin-top:4px">
                {smooth}
            </div>
            <div style="color:#64748b;font-size:0.8rem;margin-top:4px">
                Raw: {raw}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # === Bad posture timer ===
    bad_secs = st.session_state.get("bad_seconds", 0.0)
    if bad_secs > 0:
        ratio = min(bad_secs / alert_secs, 1.0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="section-label">Bad posture duration</div>
                <div style="color:#fb923c;font-size:1.4rem;font-weight:700">
                    {bad_secs:.0f}s / {alert_secs}s
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(ratio)