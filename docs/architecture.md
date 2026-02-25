# Architecture

- Tracking happens per frame inside batch.

## Camera

### Camera Worker

## Frame Queue ( Single Instance )

### Camera Config




Frame
 → Detection (SCRFD)
 → Filter
 → Crop
 → Padding
 → Alignment
 → Resize + Normalize
 → Embedding (AdaFace)
 → Tracking / Recognition