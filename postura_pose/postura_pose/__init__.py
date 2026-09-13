import streamlit as st
import streamlit.components.v2 as components

_postura_pose = components.component(
    "postura_pose.postura_pose",
    html="index.html",
    js="index.js",
)

def postura_pose(key: str = "postura_pose"):
    result = _postura_pose(
        key=key,
        default={"landmarks": None},
        on_landmarks_change=lambda: None,   
    )
    return result