# Model Architecture and Selection

The study uses a custom lightweight Ultralytics YOLO-family
encoder--neck--head architecture. It is not stock YOLO26n. The exact selected
detector configuration is stored in
[`configs/detection_range_model.yaml`](../configs/detection_range_model.yaml).

## Architecture selection

The executable search varied:

- backbone and neck blocks: `C2fCIB` or `C3Ghost`;
- downsampling: `SCDown` or `ADown`;
- late attention: `PSA` or none;
- width multiplier: 0.20 or 0.25; and
- depth multiplier: 0.34 or 0.50.

`SPPF` and a 1,024-channel cap were fixed. A language-model strategy prior
recommended a reduced search region from the larger executable space. A
deterministic selector then measured candidates, enforced limits of fewer than
2.5 million parameters and at most 7.0 GFLOPs at 640-pixel input, and retained
eight candidates that covered the reduced design choices.

All eight candidates were trained from scratch for 100 epochs on the same fixed
15,000-image COCO training subset with seed 0 and were evaluated on COCO
val2017. The highest proxy mAP selected the configuration used by this study:

- `C2fCIB` backbone and neck;
- `SCDown` downsampling;
- `SPPF` context followed by terminal `PSA`;
- width multiplier 0.25, depth multiplier 0.34, and 1,024-channel cap; and
- P3, P4, and P5 inputs to the one-class deer-detection head.

After selection, the same architecture was trained for 300 epochs on full COCO
with P3--P5 feature distillation from a YOLO26s teacher. A learned 1-by-1
projector aligned teacher and student channels during training. The teacher and
projector are not part of inference. These distilled student weights initialize
the synthetic range and geometry training. Architecture selection and
distillation are implementation steps, not claimed contributions of the
deer-safety study.

## Task heads

The range and geometry variants are separate checkpoints with the same selected
encoder and neck.

### Range head

The P3 feature map is passed through:

1. a 3-by-3 convolution with 64 output channels and SiLU;
2. a 3-by-3 convolution with 32 output channels and SiLU; and
3. a 1-by-1 convolution with one output channel.

The output represents log distance. Smooth-L1 loss is evaluated at each
ground-truth deer-box center, and inference samples the resulting metric range
at each detected-box center.

### Geometry heads

Depth and surface normals use independent P3 decoders with the same
64-channel/32-channel convolutional layout. The depth decoder ends in one
Softplus-activated channel. The normal decoder ends in three channels that are
unit-normalized for the cosine loss. During real-night adaptation, the
detection network is frozen and only these two geometry heads are updated.

Road grade is not produced by a learned head. Depth is back-projected and a
road plane is fitted; the pitch-compensated longitudinal plane tilt supplies
the grade estimate.
