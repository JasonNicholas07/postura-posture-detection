import base64
import os

ONNX_PATH = "model/posture_xgboost.onnx"
OUTPUT_TS = os.path.join(
    "postura_pose", "postura_pose", "frontend", "src", "model_b64.ts"
)

with open(ONNX_PATH, "rb") as f:
    data = f.read()

print(f"Read {len(data):,} bytes from {ONNX_PATH}")

b64 = base64.b64encode(data).decode("utf-8")

with open(OUTPUT_TS, "w") as f:
    f.write("// Auto-generated from posture_xgboost.onnx\n")
    f.write(f"export const ONNX_B64 = \"{b64}\";\n")

print(f"Base64 length: {len(b64):,} chars")
print(f"Written to: {OUTPUT_TS}")