# Postura - Real-Time Posture Detection (Browser-Side Inference)
import os
import time
import csv
import json
import streamlit as st
import streamlit.components.v1 as components
from postura_pose import postura_pose


# PAGE CONFIG
st.set_page_config(
    page_title="Postura",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    * App background override */
    .stApp {
        background-color: #0a0a0a;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Kill default Streamlit dividers */
    hr { border-color: #262626; }

    /* Cards */
    .metric-card {
        background: #141414;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border: 1px solid #1f1f1f;
    }
    .metric-card .section-label {
        color: #9ca3af;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .status-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 4px;
        color: #facc15;
    }
    .status-waiting { color: #9ca3af !important; }
    .status-normal  { color: #facc15 !important; }
    .status-forward { color: #f97316 !important; }
    .status-back    { color: #f59e0b !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #1f1f1f;
    }
    section[data-testid="stSidebar"] h2 {
        color: #facc15;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background: #facc15 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        border-radius: 999px;
    }
    .stProgress > div > div > div > div {
        background-color: #facc15 !important;
    }

    /* Primary button */
    button[kind="primary"] {
        background-color: #facc15 !important;
        color: #0a0a0a !important;
        font-weight: 700 !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        background-color: #eab308 !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #f5f5f5;
    }

    /* Yellow horizontal divider utility */
    .y-divider {
        height: 2px;
        background: #facc15;
        margin: 10px 0 18px 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


# CONSTANTS
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

FEEDBACK_LOG     = os.path.join(_ROOT, "data", "posture_feedback.csv")
BAD_POSTURE_SECS = 10
ALERT_COOLDOWN   = 60

# Class labels from the LabelEncoder used during training
LE_CLASSES = ["Back", "Forward", "Normal"]


# HELPERS
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


# SESSION STATE
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


# NOTIFICATION LOGIC
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
    """Fire a browser desktop notification"""
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
                    renotify: true,
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


# VIDEO FRAGMENT
@st.fragment
def video_fragment():
    result = postura_pose(key="postura")

    if result and result.get("posture_class"):
        st.session_state.smooth_class = result["posture_class"]
        st.session_state.raw_class    = result["posture_class"]
    else:
        st.info("Waiting for pose data...")


# SIDEBAR
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


# MAIN LAYOUT
st.markdown("## Postura")
st.markdown("Real time Posture Reminder")

col_video, col_dash = st.columns([3, 2], gap="large")

with col_video:
    if not st.session_state.camera_enabled:
        st.info("Click **Start Camera** to begin detection.")
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
            f"Fix your posture!",
        )

    # === Current prediction ===
    # st.markdown(
    #     f"""
    #     <div class="metric-card">
    #         <div class="section-label">Current posture</div>
    #         <div style="font-size:1.6rem;font-weight:800;margin-top:4px">
    #             {smooth}
    #         </div>
    #         <div style="color:#64748b;font-size:0.8rem;margin-top:4px">
    #             Raw: {raw}
    #         </div>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # === Bad posture timer ===
    # bad_secs = st.session_state.get("bad_seconds", 0.0)
    # if bad_secs > 0:
    #     ratio = min(bad_secs / alert_secs, 1.0)
    #     st.markdown(
    #         f"""
    #         <div class="metric-card">
    #             <div class="section-label">Bad posture duration</div>
    #             <div style="color:#fb923c;font-size:1.4rem;font-weight:700">
    #                 {bad_secs:.0f}s / {alert_secs}s
    #             </div>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )
    #     st.progress(ratio)