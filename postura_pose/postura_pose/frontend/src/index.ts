import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import * as ort from "onnxruntime-web";
import { FrontendRendererArgs } from "@streamlit/component-v2-lib";
import { ONNX_B64 } from "./model_b64";

// Feature order
const FEATURES = [
  "y2", "y5", "z11", "z12",
  "y2_y5_ratio", "z11_z12_diff", "z11_z12_ratio",
  "y2_diff", "y5_diff", "z11_diff", "z12_diff",
  "y2_moving_avg", "y5_moving_avg", "z11_moving_avg", "z12_moving_avg",
  "y2_normalized", "z11_normalized",
  "nose_to_shoulder_mid_x", "nose_to_shoulder_mid_y", "nose_to_shoulder_mid_dist",
  "shoulder_width",
  "left_ear_shoulder_y_diff", "right_ear_shoulder_y_diff", "ear_shoulder_y_asymmetry",
  "nose_left_ear_x_diff", "nose_right_ear_x_diff",
  "eye_level_diff",
  "relative_nose_z", "normalized_nose_z", "neck_forward_angle",
  "mean_visibility", "min_visibility",
];

// LabelEncoder sorts classes alphabetically: Back, Forward, Normal
const CLASS_NAMES = ["Back", "Forward", "Normal"];

const SMOOTHER_WINDOW = 5;
const SEND_INTERVAL_MS = 500;

// --- Style theme ---
const THEME_BG      = "#0a0a0a";
const THEME_PANEL   = "#141414";
const THEME_ACCENT  = "#facc15";
const THEME_TEXT    = "#f5f5f5";
const THEME_MUTED   = "#9ca3af";
const THEME_DANGER  = "#ef4444";

const renderer = async (args: FrontendRendererArgs) => {
  const { parentElement, setStateValue } = args;

  const wrapper = document.createElement("div");
  wrapper.style.position = "relative";
  wrapper.style.background = THEME_PANEL;
  wrapper.style.borderRadius = "12px";
  wrapper.style.overflow = "hidden";
  wrapper.style.border = "1px solid #1f1f1f";
  parentElement.appendChild(wrapper);

  const video = document.createElement("video");
  video.autoplay = true;
  video.playsInline = true;
  video.muted = true;
  video.style.width = "100%";
  video.style.display = "block";
  video.style.borderRadius = "12px";
  wrapper.appendChild(video);

  const canvas = document.createElement("canvas");
  canvas.style.position = "absolute";
  canvas.style.top = "0";
  canvas.style.left = "0";
  canvas.style.pointerEvents = "none";
  wrapper.appendChild(canvas);

  try {
    // --- Load ONNX model ---
    const onnxBytes = Uint8Array.from(atob(ONNX_B64), (c) => c.charCodeAt(0));
    const session = await ort.InferenceSession.create(onnxBytes);

    // --- Load MediaPipe ---
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });

    // --- Webcam ---
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    video.srcObject = stream;

    const resizeCanvas = () => {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
    };
    video.addEventListener("loadedmetadata", resizeCanvas);
    video.addEventListener("play", resizeCanvas);

    // --- Yellow status banner ---
    const drawBanner = (className: string) => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const pillX = 10;
      const pillY = 10;
      const pillW = 300;
      const pillH = 52;
      const r = 10;

      // Rounded black pill with yellow border
      ctx.beginPath();
      ctx.moveTo(pillX + r, pillY);
      ctx.lineTo(pillX + pillW - r, pillY);
      ctx.quadraticCurveTo(pillX + pillW, pillY, pillX + pillW, pillY + r);
      ctx.lineTo(pillX + pillW, pillY + pillH - r);
      ctx.quadraticCurveTo(pillX + pillW, pillY + pillH, pillX + pillW - r, pillY + pillH);
      ctx.lineTo(pillX + r, pillY + pillH);
      ctx.quadraticCurveTo(pillX, pillY + pillH, pillX, pillY + pillH - r);
      ctx.lineTo(pillX, pillY + r);
      ctx.quadraticCurveTo(pillX, pillY, pillX + r, pillY);
      ctx.closePath();
      ctx.fillStyle = "rgba(10,10,10,0.85)";
      ctx.fill();
      ctx.strokeStyle = THEME_ACCENT;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Yellow text
      ctx.fillStyle = THEME_ACCENT;
      ctx.font = "bold 22px sans-serif";
      ctx.textBaseline = "middle";
      ctx.fillText(`STATUS: ${className}`, pillX + 14, pillY + pillH / 2);
    };

    // --- Buffers for temporal features ---
    const frameBuffer: number[][] = [];
    const classHistory: number[] = [];
    let lastSent = 0;

    const decode = (row: number[]): Record<string, number> => {
      const out: Record<string, number> = {};
      for (let i = 0; i < 13; i++) {
        out[`x${i + 1}`] = row[i * 4 + 0];
        out[`y${i + 1}`] = row[i * 4 + 1];
        out[`z${i + 1}`] = row[i * 4 + 2];
        out[`v${i + 1}`] = row[i * 4 + 3];
      }
      return out;
    };

    const computeFeatures = (buffer: number[][]): number[] => {
      const n = buffer.length;
      const cur = decode(buffer[n - 1]);
      const prv = decode(buffer[n - 2]);

      const shoulder_mid_x = (cur.x12 + cur.x13) / 2;
      const shoulder_mid_y = (cur.y12 + cur.y13) / 2;

      const nose_to_shoulder_mid_x = cur.x1 - shoulder_mid_x;
      const nose_to_shoulder_mid_y = cur.y1 - shoulder_mid_y;
      const nose_to_shoulder_mid_dist = Math.sqrt(
        nose_to_shoulder_mid_x ** 2 + nose_to_shoulder_mid_y ** 2
      );
      const shoulder_width = Math.sqrt(
        (cur.x12 - cur.x13) ** 2 + (cur.y12 - cur.y13) ** 2
      );
      const left_ear_shoulder_y_diff = cur.y8 - cur.y12;
      const right_ear_shoulder_y_diff = cur.y9 - cur.y13;
      const ear_shoulder_y_asymmetry =
        left_ear_shoulder_y_diff - right_ear_shoulder_y_diff;
      const nose_left_ear_x_diff = cur.x1 - cur.x8;
      const nose_right_ear_x_diff = cur.x1 - cur.x9;
      const eye_level_diff = cur.y3 - cur.y6;

      const relative_nose_z = cur.z1 - (cur.z12 + cur.z13) / 2;
      const normalized_nose_z = relative_nose_z / (shoulder_width + 0.0001);
      const neck_forward_angle =
        (Math.atan2(
          Math.abs(relative_nose_z),
          Math.abs(nose_to_shoulder_mid_y) + 0.0001
        ) *
          180) /
        Math.PI;

      const v_cols = [1,2,3,4,5,6,7,8,9,10,11,12,13].map((i) => cur[`v${i}`]);
      const mean_visibility = v_cols.reduce((a, b) => a + b, 0) / v_cols.length;
      const min_visibility = Math.min(...v_cols);

      const eps = 1e-6;
      const y2 = cur.y2, y5 = cur.y5, z11 = cur.z11, z12 = cur.z12;

      const y2_y5_ratio = y2 / (y5 + eps);
      const z11_z12_diff = z11 - z12;
      const z11_z12_ratio = z11 / (z12 + eps);
      const y2_diff = y2 - prv.y2;
      const y5_diff = y5 - prv.y5;
      const z11_diff = z11 - prv.z11;
      const z12_diff = z12 - prv.z12;

      const k = Math.min(3, n);
      let y2_sum = 0, y5_sum = 0, z11_sum = 0, z12_sum = 0;
      for (let i = n - k; i < n; i++) {
        const r = decode(buffer[i]);
        y2_sum += r.y2; y5_sum += r.y5; z11_sum += r.z11; z12_sum += r.z12;
      }
      const y2_moving_avg = y2_sum / k;
      const y5_moving_avg = y5_sum / k;
      const z11_moving_avg = z11_sum / k;
      const z12_moving_avg = z12_sum / k;

      const y2_normalized = y2 / (y5 + eps);
      const z11_normalized = z11 / (z12 + eps);

      const featMap: Record<string, number> = {
        y2, y5, z11, z12,
        y2_y5_ratio, z11_z12_diff, z11_z12_ratio,
        y2_diff, y5_diff, z11_diff, z12_diff,
        y2_moving_avg, y5_moving_avg, z11_moving_avg, z12_moving_avg,
        y2_normalized, z11_normalized,
        nose_to_shoulder_mid_x, nose_to_shoulder_mid_y, nose_to_shoulder_mid_dist,
        shoulder_width,
        left_ear_shoulder_y_diff, right_ear_shoulder_y_diff, ear_shoulder_y_asymmetry,
        nose_left_ear_x_diff, nose_right_ear_x_diff,
        eye_level_diff,
        relative_nose_z, normalized_nose_z, neck_forward_angle,
        mean_visibility, min_visibility,
      };

      return FEATURES.map((f) => featMap[f]);
    };

    const predictClass = async (features: number[]): Promise<number> => {
      const tensor = new ort.Tensor(
        "float32",
        Float32Array.from(features),
        [1, features.length]
      );
      const feeds: Record<string, ort.Tensor> = {};
      feeds[session.inputNames[0]] = tensor;
      const out = await session.run(feeds);
      const proba = out[session.outputNames[1]].data as Float32Array;
      let best = 0, bestVal = proba[0];
      for (let i = 1; i < proba.length; i++) {
        if (proba[i] > bestVal) { bestVal = proba[i]; best = i; }
      }
      return best;
    };

    const detectLoop = async () => {
      if (video.readyState >= 2) {
        const t = performance.now();
        const results = poseLandmarker.detectForVideo(video, t);

        if (results.landmarks && results.landmarks.length > 0) {
          const lm = results.landmarks[0].slice(0, 13);
          const flat = lm.flatMap((p) => [p.x, p.y, p.z, p.visibility ?? 0]);

          frameBuffer.push(flat);
          if (frameBuffer.length > 3) frameBuffer.shift();

          if (frameBuffer.length === 3) {
            try {
              const features = computeFeatures(frameBuffer);
              const idx = await predictClass(features);

              classHistory.push(idx);
              if (classHistory.length > SMOOTHER_WINDOW) classHistory.shift();

              const counts: Record<number, number> = {};
              for (const c of classHistory) counts[c] = (counts[c] || 0) + 1;
              const smoothedIdx = parseInt(
                Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]
              );
              const className = CLASS_NAMES[smoothedIdx] ?? "Unknown";

              drawBanner(className);

              if (Date.now() - lastSent > SEND_INTERVAL_MS) {
                setStateValue("posture_class", className);
                lastSent = Date.now();
              }
            } catch (e) {
              console.error("Inference error:", e);
            }
          }
        } else {
          drawBanner("No pose");
        }
      }
      requestAnimationFrame(detectLoop);
    };

    detectLoop();
  } catch (err) {
    console.error("Setup error:", err);
    parentElement.innerHTML = `<div style="color:${THEME_DANGER};background:${THEME_BG};padding:8px;font-family:sans-serif;border-radius:8px;">Error: ${err}</div>`;
  }
};

export default renderer;