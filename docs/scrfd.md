# SCRFD Model Detection Score:

- det_score < `0.4`  → very uncertain detection, likely false positive
- det_score `0.4-0.6` → partial face, occluded
- det_score > `0.6`  → clean detection ✅
- det_score > `0.85` → near perfect detection ✅✅