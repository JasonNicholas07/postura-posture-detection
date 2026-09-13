# Posture Classification

# library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Posture Classification Training (baseline, no hyperparameter search)")

# 1. define (setting)
DATASET_PATH        = 'data/dataset_postur_more.csv'
MODEL_OUTPUT_PATH   = 'model/xgboost_final.pkl'
TEST_SIZE           = 0.2
RANDOM_STATE        = 67
NORMAL_THRESHOLD    = 0.55  
BACK_THRESHOLD      = 0.95

# Plain / normal hyperparameters (no Optuna search)
N_ESTIMATORS   = 300
LEARNING_RATE  = 0.1
MAX_DEPTH      = 6

# Epoch range to display on the learning curve plot
PLOT_EPOCH_MIN = 0
PLOT_EPOCH_MAX = 100

# data loading
print("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Rows: {len(df):}   |   Columns: {len(df.columns)}")
print(f"Class distribution:\n{df['class'].value_counts().to_string()}")

# FEATURES
LANDMARK_COUNT = 13   # upper body landmarks 1–13
raw_features = []
for i in range(1, LANDMARK_COUNT + 1):
    raw_features.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])


def compute_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.einsum('ij,ij->i', v1, v2) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    feat['y2'] = df['y2']
    feat['y5'] = df['y5']
    feat['z11'] = df['z11']
    feat['z12'] = df['z12']

    # 1. REFERENSI TITIK TENGAH (Pusat Tubuh)
    shoulder_mid_x = (df['x12'] + df['x13']) / 2
    shoulder_mid_y = (df['y12'] + df['y13']) / 2

    # Shoulder midpoint vs nose (forward/back lean indicator)
    feat['nose_to_shoulder_mid_x'] = df['x1'] - shoulder_mid_x
    feat['nose_to_shoulder_mid_y'] = df['y1'] - shoulder_mid_y
    feat['nose_to_shoulder_mid_dist'] = np.sqrt(
        feat['nose_to_shoulder_mid_x']**2 + feat['nose_to_shoulder_mid_y']**2
    )

    # Shoulder width (lateral lean / slouch)
    feat['shoulder_width'] = np.sqrt(
        (df['x12'] - df['x13'])**2 + (df['y12'] - df['y13'])**2
    )

    # Ear–shoulder alignment (head tilt/forward head)
    feat['left_ear_shoulder_y_diff']  = df['y8']  - df['y12']
    feat['right_ear_shoulder_y_diff'] = df['y9']  - df['y13']
    feat['ear_shoulder_y_asymmetry']  = (
        feat['left_ear_shoulder_y_diff'] - feat['right_ear_shoulder_y_diff']
    )

    # Nose–ear horizontal offset (head forward lean)
    feat['nose_left_ear_x_diff']  = df['x1'] - df['x8']
    feat['nose_right_ear_x_diff'] = df['x1'] - df['x9']

    # Eye level asymmetry (head tilt left/right)
    feat['eye_level_diff'] = df['y3'] - df['y6']

    # 3. depth feature (Sumbu Z) & SUDUT
    feat['relative_nose_z'] = df['z1'] - ((df['z12'] + df['z13']) / 2)

    # Normalisasi
    feat['normalized_nose_z'] = feat['relative_nose_z'] / (feat['shoulder_width'] + 0.0001)

    # sudut derajat leher
    feat['neck_forward_angle'] = np.degrees(
        np.arctan2(
            np.abs(feat['relative_nose_z']),
            np.abs(feat['nose_to_shoulder_mid_y']) + 0.0001
        )
    )
    LANDMARK_COUNT = 13
    v_cols = [f'v{i}' for i in range(1, LANDMARK_COUNT + 1)]

    if all(col in df.columns for col in v_cols):
        feat['mean_visibility'] = df[v_cols].mean(axis=1)
        feat['min_visibility']  = df[v_cols].min(axis=1)

    # ratio
    feat['y2_y5_ratio']   = df['y2'] / (df['y5'] + 1e-6)
    feat['z11_z12_diff']  = df['z11'] - df['z12']
    feat['z11_z12_ratio'] = df['z11'] / (df['z12'] + 1e-6)

    # temporal features
    feat['y2_diff']  = df['y2'].diff().fillna(0)
    feat['y5_diff']  = df['y5'].diff().fillna(0)
    feat['z11_diff'] = df['z11'].diff().fillna(0)
    feat['z12_diff'] = df['z12'].diff().fillna(0)

    # Moving averages over a window of 3 frames
    feat['y2_moving_avg']  = df['y2'].rolling(window=3).mean().bfill()
    feat['y5_moving_avg']  = df['y5'].rolling(window=3).mean().bfill()
    feat['z11_moving_avg'] = df['z11'].rolling(window=3).mean().bfill()
    feat['z12_moving_avg'] = df['z12'].rolling(window=3).mean().bfill()

    feat['y2_normalized']  = df['y2'] / (df['y5'] + 1e-6)
    feat['z11_normalized'] = df['z11'] / (df['z12'] + 1e-6)

    return feat

print("\nEngineering features...")
X_full = build_features(df)

# Define the specific feature set
features = [
    # --- raw / temporal (moving-avg) group you were already using ---
    'y2', 'y5', 'z11', 'z12',
    'y2_y5_ratio', 'z11_z12_diff', 'z11_z12_ratio',
    'y2_diff', 'y5_diff', 'z11_diff', 'z12_diff',
    'y2_moving_avg', 'y5_moving_avg', 'z11_moving_avg', 'z12_moving_avg',
    'y2_normalized', 'z11_normalized',

    # --- scale-invariant geometric group (was being computed but discarded) ---
    'nose_to_shoulder_mid_x', 'nose_to_shoulder_mid_y', 'nose_to_shoulder_mid_dist',
    'shoulder_width',
    'left_ear_shoulder_y_diff', 'right_ear_shoulder_y_diff', 'ear_shoulder_y_asymmetry',
    'nose_left_ear_x_diff', 'nose_right_ear_x_diff',
    'eye_level_diff',
    'relative_nose_z', 'normalized_nose_z', 'neck_forward_angle',
    'mean_visibility', 'min_visibility',
]

X_full = X_full[features]

print(f"Total features used: {X_full.shape[1]}")
feature_names = list(X_full.columns)

# 4. LABEL ENCODING + SPLIT
print("\nEncoding labels and splitting data...")
le = LabelEncoder()
y_encoded = le.fit_transform(df['class'])
print(f"    Classes: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)
print(f"    Train: {len(X_train):,}   |   Test: {len(X_test):,}")

normal_idx = int(le.transform(['Normal'])[0])
back_idx = le.transform(['Back'])[0]
print(f"    'Normal' class index: {normal_idx}")
print(f"    'Back' class index: {back_idx}")


# 5. CLASS WEIGHTS
def make_sample_weights(y, normal_idx, normal_multiplier=1.0):
    classes  = np.unique(y)
    cw       = compute_class_weight('balanced', classes=classes, y=y)
    wdict    = dict(zip(classes, cw))
    wdict[normal_idx] *= normal_multiplier
    return np.array([wdict[c] for c in y])


sw_train = make_sample_weights(y_train, normal_idx, normal_multiplier=1.0)


print("\nTraining baseline XGBoost model (fixed hyperparameters)...")

final_model = XGBClassifier(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    max_depth=MAX_DEPTH,
    subsample = 0.6108295,
    colsample_bytree = 0.926256,
    gamma = 0.22345556,
    min_child_weight = 6,
    reg_alpha = 0.00080326,
    reg_lambda = 0.5334286,
    normal_multiplier = 0.94148274,
    random_state=RANDOM_STATE,
    eval_metric=['mlogloss', 'merror'],   # merror -> accuracy = 1 - merror
)

eval_set = [(X_train, y_train), (X_test, y_test)]

final_model.fit(
    X_train, y_train,
    sample_weight=sw_train,
    eval_set=eval_set,
    verbose=False
)

results = final_model.evals_result()
epochs  = len(results['validation_0']['mlogloss'])
x_axis  = range(epochs)

train_loss = np.array(results['validation_0']['mlogloss'])
val_loss   = np.array(results['validation_1']['mlogloss'])
train_acc  = 1 - np.array(results['validation_0']['merror'])
val_acc    = 1 - np.array(results['validation_1']['merror'])
lr_fixed   = np.full(epochs, LEARNING_RATE)  # constant per round (no schedule)

print(f"\nFinal epoch -> loss: {train_loss[-1]:.4f} | val_loss: {val_loss[-1]:.4f} "
      f"| accuracy: {train_acc[-1]*100:.2f}% | val_accuracy: {val_acc[-1]*100:.2f}% "
      f"| lr (fixed): {LEARNING_RATE}")

# -----------------------------------------------------
# combined learning curve
# -----------------------------------------------------
plot_max = min(PLOT_EPOCH_MAX, epochs)  # clip in case fewer rounds were trained
if PLOT_EPOCH_MIN >= epochs:
    raise ValueError(
        f"PLOT_EPOCH_MIN={PLOT_EPOCH_MIN} but only {epochs} boosting rounds were trained "
        f"(increase N_ESTIMATORS or lower PLOT_EPOCH_MIN)."
    )

window = slice(PLOT_EPOCH_MIN, plot_max)
x_window = list(x_axis)[window]

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x_window, train_loss[window], label='loss', color='#1f77b4', linewidth=1.3)
ax.plot(x_window, train_acc[window], label='accuracy', color='#ff7f0e', linewidth=1.3)
ax.plot(x_window, val_loss[window], label='val_loss', color='#2ca02c', linewidth=1.3)
ax.plot(x_window, val_acc[window], label='val_accuracy', color='#d62728', linewidth=1.3)
ax.plot(x_window, lr_fixed[window], label='lr', color='#9467bd', linewidth=1.3)

ax.set_xlabel('Epoch')
ax.set_ylim(0.0, 1.0)
ax.set_xlim(PLOT_EPOCH_MIN, plot_max)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.6)
ax.set_title(f'Learning curve (epoch {PLOT_EPOCH_MIN}-{plot_max})')

plt.tight_layout()
plt.savefig('learning_curve_combined.png', dpi=150)
plt.show()

# Evaluasi Final
y_pred = final_model.predict(X_test)


# 8. THRESHOLD-BASED PREDICTION FUNCTION
def predict_with_threshold(model, X, normal_idx, back_idx, normal_thresh=0.7, back_thresh=0.8):
    proba      = model.predict_proba(X)                   # (N, n_classes)
    n_classes  = proba.shape[1]
    predictions = []

    for p in proba:
        if p[back_idx] >= back_thresh:
            predictions.append(back_idx)
        elif p[normal_idx] >= normal_thresh:
            predictions.append(normal_idx)
        else:
            # Pick best class that is NEITHER Normal NOR Back
            mask = np.ones(n_classes, dtype=bool)
            mask[normal_idx] = False
            mask[back_idx] = False
            best_other = np.argmax(p * mask)
            predictions.append(int(best_other))

    return np.array(predictions)

# 9. EVALUATION
print("\nEvaluation on held-out test set...")

# Standard argmax prediction
y_pred_raw = final_model.predict(X_test)

# Threshold-based prediction
y_pred_thresh = predict_with_threshold(
    model=final_model,
    X=X_test,
    normal_idx=normal_idx,
    back_idx=back_idx,
    normal_thresh=NORMAL_THRESHOLD,
    back_thresh=BACK_THRESHOLD
)

acc_raw    = accuracy_score(y_test, y_pred_raw)
acc_thresh = accuracy_score(y_test, y_pred_thresh)

print(f"\n  Accuracy (argmax):              {acc_raw    * 100:.2f}%")
print(f"  Accuracy (threshold={NORMAL_THRESHOLD}):    {acc_thresh * 100:.2f}%")

print("\n Classification Report (argmax)")
print(classification_report(y_test, y_pred_raw, target_names=le.classes_))

print(f"   Classification Report (threshold={NORMAL_THRESHOLD}) ")
print(classification_report(y_test, y_pred_thresh, target_names=le.classes_))

print("  Confusion Matrix ")


# --- PLOT: FEATURE IMPORTANCE ---
importances = final_model.feature_importances_
fi_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': importances
}).sort_values('importance', ascending=True).tail(15)  # top 15, ascending for horizontal bar

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(fi_df['feature'], fi_df['importance'], color='#2ca02c')
ax.set_xlabel('Importance (gain-normalized)')
ax.set_title('Top 15 Feature Importances')
ax.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

print("\nTop 15 Feature Importances:")
fi_desc = fi_df.sort_values('importance', ascending=False)
for _, row in fi_desc.iterrows():
    bar = '█' * int(row['importance'] * 200)
    print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")

# --- PLOT: CONFUSION MATRIX (heatmap, threshold-based) ---
cm = confusion_matrix(y_test, y_pred_thresh)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

ax.set_xticks(range(len(le.classes_)))
ax.set_yticks(range(len(le.classes_)))
ax.set_xticklabels(le.classes_, rotation=45, ha='right')
ax.set_yticklabels(le.classes_)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (threshold={NORMAL_THRESHOLD})')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
        ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)',
                ha='center', va='center', color=text_color, fontsize=9)

fig.colorbar(im, ax=ax, label='Row-normalized proportion')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()


# 11. TEMPORAL SMOOTHER
class TemporalSmoother:
    def __init__(self, window: int = 10):
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

# 12. SAVE
export_data = {
    'model':              final_model,
    'encoder':            le,
    'features':           feature_names,
    'raw_features':       raw_features,
    'normal_idx':         normal_idx,
    'normal_threshold':   NORMAL_THRESHOLD,
    'build_features_fn':  build_features,   # save the fn reference for inference
    'TemporalSmoother':   TemporalSmoother,
}
joblib.dump(export_data, MODEL_OUTPUT_PATH)
print(f"\nModel saved → '{MODEL_OUTPUT_PATH}'")