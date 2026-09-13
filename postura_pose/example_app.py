import streamlit as st
from postura_pose import postura_pose

st.set_page_config(page_title="Postura Pose Test", layout="wide")
st.title("Postura Pose Component Test")

result = postura_pose(key="test_pose")

if result and result.get("landmarks"):
    landmarks = result["landmarks"]
    st.success(f"Received {len(landmarks)} landmark values")
    st.write("First 8 values:", landmarks[:8])
else:
    st.info("Waiting for pose data...")