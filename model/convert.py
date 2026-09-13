import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType


# Pickle stubs
def build_features(df):
    raise NotImplementedError("Stub only")


class TemporalSmoother:
    def __init__(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): return 0
    def reset(self, *args, **kwargs): pass


# Load model
data = joblib.load("model/xgboost_final.pkl")
final_model   = data["model"]
feature_names = data["features"]
le            = data["encoder"]
normal_idx    = data["normal_idx"]

n_features = len(feature_names)

print(f"Loaded model with {n_features} features")
print(f"Classes: {list(le.classes_)}")
print(f"Normal class index: {normal_idx}")


# Rename booster features to f0, f1, ...
booster = final_model.get_booster()
booster.feature_names = [f"f{i}" for i in range(n_features)]


# Convert to ONNX
initial_type = [("input", FloatTensorType([None, n_features]))]
onnx_model = onnxmltools.convert_xgboost(
    final_model,
    initial_types=initial_type,
)

with open("model/posture_xgboost.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\nSaved to model/posture_xgboost.onnx")


# Verify ONNX == XGBoost
import onnxruntime as ort
sess = ort.InferenceSession("model/posture_xgboost.onnx")
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]
print(f"ONNX inputs:  {[i.name for i in sess.get_inputs()]}")
print(f"ONNX outputs: {output_names}")

X_sample = np.random.randn(5, n_features).astype(np.float32)
onnx_out   = sess.run(None, {input_name: X_sample})
onnx_proba = onnx_out[1] if len(onnx_out) > 1 else onnx_out[0]
onnx_pred  = np.argmax(onnx_proba, axis=1)

sk_proba = final_model.predict_proba(X_sample)
sk_pred  = np.argmax(sk_proba, axis=1)

print("\nVerification:")
print(f"  ONNX preds:    {onnx_pred}")
print(f"  XGBoost preds: {sk_pred}")
print(f"  Max proba diff: {np.max(np.abs(onnx_proba - sk_proba)):.6f}")
print(f"  Match: {np.array_equal(onnx_pred, sk_pred)}")