# Synthetic Deer Dataset — Scene Reference

**Last verified: 2026-07-23.** All frame/box counts and distance ranges in this document were read
directly from the rendered HDF5 and COCO files, not copied from notes.

**Totals: 23 rendered scenes, 10,889 frames, 13,610 deer boxes.** The near-range,
30--60 m range, and road-furniture subsets contain 1,314/1,314, 8,375/10,717,
and 1,200/1,579 frames/boxes, respectively.

Every frame, in every scene, contains:

| Modality | Stored as | Notes |
|---|---|---|
| RGB image | `colors` | 960×540 (or 1280×720 for crossing1–3) |
| Metric depth | `depth` | meters, per pixel |
| Surface normals | `normals` | camera frame |
| Semantic segmentation | `class_segmaps` | deer = 1, background = 0 |
| Instance segmentation | `instance_segmaps` | one id per animal |
| Instance attributes | `instance_attribute_maps` | maps instance id → category |
| Diffuse albedo | `diffuse` | lighting-free colour pass |
| Deer boxes | `coco_data/coco_annotations.json` | COCO format |
| **Per-deer distance** | derived | median `depth` inside each deer's instance mask |

That last row is the reason the dataset exists: **real dashcam footage cannot be labeled with
per-animal metric distance.** Rendering is what makes a distance-supervised range head possible.

---

## 1. The three base environments

Everything traces back to three Blender source files. This matters more than the scene count,
because scenes from one source share road mesh, terrain, sky, camera rig, and exposure.

| Environment | Source file | Looks like | Scenes |
|---|---|---|---|
| **Mountain road** | `mountain_road.blend` | Winding two-lane road cut into a hillside; rock/earth embankment rising on one side, concrete barrier or curb along it; deep night, headlights only | crossing1, 2, 3 |
| **Forest road** | `night_road_scene.blend` | Dense tree canopy closing overhead, **no sky visible**; double-yellow centre line; headlights fade into blackness; fog frequent | crossing4, 6, **7**, f1, f2, f3 |
| **Open rural road** | `american_rural_uphill_road_with_assets.blend` | Grassy shoulders, scattered pines and broadleaf trees, white delineator posts, distant low hills under a residual dusk sky | crossing8–15, cueA–D |

**Visual signature (measured):** the forest environment has the *least* visible sky (0.15 sky
fraction vs 0.27–0.47 for rural) and the *darkest* upper image third (22 vs 26–48), yet the
*highest* overall brightness (54 vs 21–36) because headlights blast the near foreground against
black. Its image statistics are effectively inverted relative to the rural family — which is why
models trained on rural scenes fail on it.

---

## 2. Near-range corpus (deer within ~35 m)

Rendered first, before the range experiments. Used to train and validate the early detection and
geometry models. **Single deer per frame** (the multi-deer generator bug was not yet fixed).

| Scene | Environment | Frames | Boxes | Distance | Resolution | Function |
|---|---|---:|---:|---:|---|---|
| crossing1 | Mountain | 199 | 199 | 3–22 m | 1280×720 | detection/geometry training |
| crossing2 | Mountain | 199 | 199 | 3–16 m | 1280×720 | detection/geometry training |
| crossing3 | Mountain | 199 | 199 | 4–16 m | 1280×720 | detection/geometry training |
| crossing4 | Forest | 198 | 198 | 9–33 m | 960×540 | detection/geometry training |
| crossing6 | Forest | 519 | 519 | 8–35 m | 960×540 | **geometry validation set** |

*Subtotal: 1,314 frames. crossing1–4 (795 frames) = training, crossing6 (519) = validation.*
*There is no crossing5 — it was renamed to crossing6 early on.*

---

## 3. Range corpus (deer 30–60 m) — the main experimental set

Purpose-built for the range and stopping-distance work. All are 960×540. The original
crossing7--crossing8 pair predates the multi-deer correction and is effectively single-deer;
the later range scenes contain 1--3 deer per frame with roughly one frame in eight deer-free.

### 3.1 The original pair

| Scene | Env | Alignment | Grade | Frames | Boxes | Function / what it looks like |
|---|---|---|---|---:|---:|---|
| **crossing7** | Forest | straight | level | 600 | 525 | **The out-of-domain probe.** Dark tunnel of canopy, double-yellow line, fog. Originally the single training road; now the one scene no rural-trained model handles. |
| **crossing8** | Rural | gentle curve | +6% crest | 600 | 536 | **The original zero-shot test road.** Open rural, climbs to a crest. Used to compare single-scene vs multi-scene training on an identical test set. |

These scenes are a bundled domain-transfer comparison, not a one-factor ablation. They use
different source `.blend` environments, road geometry, camera coordinate/aiming logic,
exposure, lateral spawn width, and deer scale ranges. They retain the same resolution,
30--60 m distance sampler, and default distributions for fog, headlights, oncoming glare,
camera height/lens, and deer-free frames. The archived scripts did not record a random
seed, so the released frames—not bitwise re-rendering—define the experiment. The complete
source hashes, settings comparison, and all-frame image statistics are in
`trb2027/output/scene_provenance/scene_provenance.{json,md}`.

### 3.2 Geometry variants — vary the *road*, hold the scenery fixed

Built by remapping the rural base: same trees, sky, textures; only the centreline changes.

| Scene | Alignment | Grade (30–60 m) | Frames | Boxes | Looks like |
|---|---|---|---:|---:|---|
| **crossing9a** | straight | −4% | 600 | 813 | Straight descending road; treeline left, open field right |
| **crossing9b** | right curve | level (0%) | 600 | 826 | Road bends right (−2.5 m lateral by 60 m); trees both sides |
| **crossing9c** | S-curve | +3% | 600 | 818 | Swings right then left (+1.7→−1.8 m); climbing |
| **crossing10** | left curve | crest, +5%→−3% | 600 | 802 | Bends left and rises to a **parabolic crest** (apex 40 m) then descends; far deer occluded by the crest |

### 3.3 Appearance variants — vary the *look*, same 3D assets

Same tree meshes as above, recoloured/thinned/relit.

| Scene | Environment look | Grade | Frames | Boxes | Distinguishing features |
|---|---|---|---:|---:|---|
| **crossing11** | Open farmland | −2% | 600 | 783 | Golden crop stubble, 121/136 vegetation hidden, **no lane markings, no posts**, bright sky |
| **crossing12** | Dense autumn forest | −5% | 600 | 819 | Orange/red foliage crowding the shoulder, vegetation ×1.35, **dark sky (×0.55)**, posts + markings |
| **crossing13** | Arid scrub | +4% | 600 | 799 | Red-ochre desert dirt, all tall trees removed (130/136 hidden), markings but no posts |

### 3.4 Asset variants — replace the vegetation *meshes*

All 137 original vegetation objects deleted and repopulated from a different asset library.

| Scene | Vegetation source | Alignment / grade | Frames | Boxes | Looks like |
|---|---|---|---:|---:|---|
| **crossing14** | glTF conifer (`road_with_trees.glb`) | left curve, −3% | 600 | 782 | 70 scattered detailed pines; posts + markings |
| **crossing15** | `real-bush` FBX pack | straight, +2% | 600 | 812 | 150 low shrubs on both shoulders, prairie feel, no posts |

### 3.5 Road-furniture ablation (cueA–D) — controlled 2×2

**Not training data.** Four scenes identical in geometry, vegetation, lighting, and random seed;
the *only* difference is road furniture. Evaluated with one checkpoint to isolate the cue.

| Scene | Lane markings | Delineator posts | Frames | Boxes | Result (range MAE 30–40 m) |
|---|:--:|:--:|---:|---:|---|
| **cueA** | yes | yes | 300 | 404 | 1.82 m (control) |
| **cueB** | yes | no | 300 | 403 | **1.41 m** (best) |
| **cueC** | no | yes | 300 | 384 | 3.18 m |
| **cueD** | no | no | 300 | 388 | **3.65 m** (worst) |

**Finding:** removing markings roughly doubles range error; removing posts changes nothing.
Detection recall is flat (0.99–1.00) across all four, so this is specifically a *ranging* effect.

### 3.6 Forest variants (f1–f3) — completed recovery set

Built from `night_road_scene.blend`, crossing7's own source, to give the forest domain siblings.
Straight and flat (road_z = 0.12 m) like their parent; the camera was moved back to X = −26.7 so the
full 30–60 m corridor sits on actual road, which crossing7 did not do.

| Scene | Canopy | Frames | Boxes | Function |
|---|---|---:|---:|---|
| **f1** | dense green summer, 90/90 vegetation, sky ×0.9 | 575 | 778 | test whether crossing7's failure is *domain coverage* |
| **f2** | thinned autumn, 43/90 vegetation, sky ×1.3 | 600 | 806 | " |
| **f3** | dark wet, 81/90 vegetation, sky ×0.5 | 600 | 818 | " |

---

## 4. Which scenes were used for what

| Experiment | Scenes used | Outcome |
|---|---|---|
| Early detection + geometry models | crossing1–4 (train), crossing6 (val) | baseline synthetic models |
| Historical single-scene range experiment | crossing7 → crossing8 | On current instance labels, no band passes either the recall or joint screen |
| Primary multi-scene cross-validation | crossing7, 8, 9a–c, 10, 11–13, 14, 15 | Aggregate rural joint screen passes through 60 m; 6/10 rural folds individually pass at 50–60 m |
| Out-of-domain probe | crossing7 | fails all bands; degrades as rural scenes are added |
| Road-furniture ablation | cueA–D | markings act as a monocular depth cue |
| Forest-domain recovery | crossing7 held out after adding f1–f3; f1 control fold | crossing7 recall becomes 0.503/0.595/0.627; held-out f1 recall is 0.934/0.879/0.738 |

---

## 5. Practical notes and caveats

- **Two generator bugs were fixed on 2026-07-20.** Animated `hide_render` was ignored because
  BlenderProc enables Cycles persistent data, and duplicated deer shared mesh data so the
  segmentation collapsed them into one instance. **Everything rendered before that date
  (crossing1–8) is effectively single-deer**, which is why those scenes show ~1.00 box per frame
  while later scenes show ~1.30–1.38.
- **crossing7 and crossing8 box counts (525, 536) differ from earlier notes (733, 724)** because
  the current converter counts one box per deer *instance*, whereas the older converter split
  occluded animals into connected components. The instance counts are the correct ones.
- **Ten of the eleven primary 30–60 m LOSO scenes come from one source environment.** The three
  forest siblings are reserved for the separate recovery analysis. Scene count is not the same
  as domain coverage — that distinction drives the generalization limits observed.
- **crossing10's crest was rebuilt** on 2026-07-21 as a proper parabolic vertical curve. The
  original version met +5% and −3% grades at a sharp kink at exactly 30 m, which is not a real road
  shape and sat exactly where measurement begins; it produced a spurious ~20 m range error.
- **Distance is always derived**, never authored: median depth inside the instance mask.

## 6. Where things live

| What | Path |
|---|---|
| Rendered scenes | `deer_dataset/synthetic_deer_dataset/<subset>/<scene>/` |
| Scene index and experiment splits | `deer_dataset/synthetic_deer_dataset/SCENES.csv`, `splits/` |
| YOLO conversions | `trb2027/data/loso/scenes/<tag>/` |
| Blender scene files | `blender/DeerDataset/<scene>.blend` |
| Road profiles | `blender/DeerDataset/<scene>_road_profile.json` |
| Scene builders | `build_scene_variant.py`, `build_appearance_variant.py`, `build_asset_variant.py`, `build_cue_variant.py`, `build_forest_variant.py` |
| Generator | `blender/DeerDataset/generate_deer_crossing_nightroad.py` |
