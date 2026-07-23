# False positives on 16 deer-absent dusk frames

Frozen inference at confidence 0.25, NMS IoU 0.70, and square 640-pixel input (batch 1). Only predictions assigned to the deer class are counted.

| model | deer detections | frames alerted | FP/frame | frame alert rate |
|---|---:|---:|---:|---:|
| all_condition_yolov8n | 4 | 3/16 | 0.250 | 0.188 |
| real_plus_synthetic_searched | 0 | 0/16 | 0.000 | 0.000 |

These 16 frames come from four temporally related self-collected clips. The counts are a nuisance-output diagnostic, not a deployment false-alarm rate.
