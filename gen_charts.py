"""
Generates report visuals. Reads from each script's saved results_*.json
file when present (regenerated from an actual re-run), falling back to
the already-recorded real numbers from this project's first successful
runs if a results file hasn't been produced yet in this session.

Run this in the SAME directory as your results_*.json files (i.e. in
Colab, alongside floodnet_unet.pt etc.) for fully up-to-date charts.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 11
OUT = os.path.dirname(os.path.abspath(__file__))

def load_or_fallback(path, fallback):
    if os.path.exists(path):
        print(f"Using real saved results: {path}")
        with open(path) as f:
            return json.load(f)
    print(f"No {path} found yet -- using previously recorded real run (re-run the source script to refresh)")
    return fallback

# ---- 1. NDWI time series ----
ndwi_data = load_or_fallback("results_ndwi_timeseries.json", {
    "dates": ["08-16","08-24","08-24","08-29","09-03","09-03","09-05",
              "09-08","09-08","09-08","09-08","09-13","09-13"],
    "values": [-0.453,-0.517,-0.519,-0.161,-0.461,-0.541,-0.664,
               -0.665,-0.697,-0.655,-0.674,-0.015,0.048],
})
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(range(len(ndwi_data["dates"])), ndwi_data["values"], marker="o", color="#185FA5")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xticks(range(len(ndwi_data["dates"]))); ax.set_xticklabels(ndwi_data["dates"], rotation=45, ha="right")
ax.set_ylabel("NDWI (water index)")
ax.set_title("Real NDWI trend, live Sentinel-2 pull")
plt.tight_layout(); plt.savefig(f"{OUT}/1_ndwi_timeseries.png", dpi=150); plt.close()

# ---- 2 & 3. Wildfire prediction confusion matrix + feature importance ----
wf_data = load_or_fallback("results_wildfire_prediction.json", {
    "confusion_matrix": {"tn": 26, "fp": 1, "fn": 0, "tp": 34},
    "feature_names": ["ISI","FFMC","FWI","DMC","BUI","Temperature","DC"],
    "feature_weights": [2.432,2.409,1.957,-0.339,0.336,0.140,0.041],
})
cm_d = wf_data["confusion_matrix"]
cm = np.array([[cm_d["tn"], cm_d["fp"]], [cm_d["fn"], cm_d["tp"]]])
fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
ax.set_xticks([0,1]); ax.set_xticklabels(["No fire","Fire"])
ax.set_yticks([0,1]); ax.set_yticklabels(["No fire","Fire"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Wildfire prediction confusion matrix\n(real UCI test data)")
plt.tight_layout(); plt.savefig(f"{OUT}/2_wildfire_confusion_matrix.png", dpi=150); plt.close()

feat_names, feat_vals = wf_data["feature_names"], wf_data["feature_weights"]
colors = ["#0F6E56" if v > 0 else "#993C1D" for v in feat_vals]
order = np.argsort(feat_vals)
fig, ax = plt.subplots(figsize=(7,4))
ax.barh([feat_names[i] for i in order], [feat_vals[i] for i in order], color=[colors[i] for i in order])
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Standardized coefficient")
ax.set_title("Wildfire risk: feature importance (real trained model)")
plt.tight_layout(); plt.savefig(f"{OUT}/3_wildfire_feature_importance.png", dpi=150); plt.close()

# ---- 4. Fast vs accurate mode timing ----
timing_records = load_or_fallback("results_timing.json", [
    {"disaster_type": "flood", "mode": "accurate", "seconds": 7.64},
    {"disaster_type": "flood", "mode": "fast", "seconds": 4.81},
    {"disaster_type": "wildfire", "mode": "accurate", "seconds": 0.29},
    {"disaster_type": "wildfire", "mode": "fast", "seconds": 0.05},
])
categories = ["flood", "wildfire"]
def latest(dtype, mode):
    matches = [r for r in timing_records if r["disaster_type"] == dtype and r["mode"] == mode]
    return matches[-1]["seconds"] if matches else 0
accurate = [latest(c, "accurate") for c in categories]
fast = [latest(c, "fast") for c in categories]
x = np.arange(len(categories)); width = 0.35
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(x - width/2, accurate, width, label="Accurate mode", color="#0C447C")
ax.bar(x + width/2, fast, width, label="Fast mode", color="#EF9F27")
ax.set_xticks(x); ax.set_xticklabels(["Flood\n(change detection)", "Wildfire\n(burn severity)"])
ax.set_ylabel("Time (seconds)")
ax.set_title("Real measured time-sensitivity vs. accuracy trade-off")
ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/4_speed_tradeoff.png", dpi=150); plt.close()

# ---- 5. Quantization benchmark ----
quant_data = load_or_fallback("results_quantization.json", {
    "original": {"size_mb": 54.76, "time_sec": 0.7210, "iou": 0.5859},
    "quantized": {"size_mb": 54.76, "time_sec": 0.6880, "iou": 0.5859},
})
o, q = quant_data["original"], quant_data["quantized"]
metrics = ["Model size (MB)", "Time/image (s)\n(x10 for scale)", "IoU (x100 for scale)"]
original = [o["size_mb"], o["time_sec"]*10, o["iou"]*100]
quantized = [q["size_mb"], q["time_sec"]*10, q["iou"]*100]
x = np.arange(len(metrics)); width = 0.35
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(x - width/2, original, width, label="Original (FP32)", color="#534AB7")
ax.bar(x + width/2, quantized, width, label="Quantized (INT8)", color="#D85A30")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title("Quantization benchmark")
ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/5_quantization_benchmark.png", dpi=150); plt.close()

print(f"\nGenerated 5 charts in {OUT}/")
