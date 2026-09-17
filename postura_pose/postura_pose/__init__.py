import os
import streamlit.components.v2 as components

_HERE = os.path.dirname(__file__)
_BUILD = os.path.join(_HERE, "frontend", "build")

with open(os.path.join(_BUILD, "index.js"), "r", encoding="utf-8") as f:
    _JS_CONTENT = f.read()

_detector = components.component(
    "postura_pose.postura_pose",
    js=_JS_CONTENT,
)


def postura_pose(key: str = "postura_pose"):
    return _detector(
        key=key,
        default={"posture_class": None},
        on_posture_class_change=lambda: None,
    )