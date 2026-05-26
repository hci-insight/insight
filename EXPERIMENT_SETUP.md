# Shot Capture Experiment

This prototype is meant to run from the `cv` conda environment.

```bash
conda activate cv
```

If another teammate needs the same environment from scratch:

```bash
conda env create -f visionproject/environment.yml
conda activate cv
```

### Download detector models (run once)

The face detectors need small model files (~230 KB each) in `visionproject/models/`:

```bash
cd visionproject
bash download_models.sh
```

This fetches:

- `models/blaze_face_short_range.tflite` — MediaPipe (Google) Face Detection
- `models/face_detection_yunet_2023mar.onnx` — OpenCV YuNet face detector

If a model file is missing, the script also prints the exact `curl` command when it starts.

On macOS, `scrcpy` and `adb` can also be installed from the included Brewfile:

```bash
brew bundle --file visionproject/Brewfile
```

It supports several capture policies in one loop so you can compare them with the same scene.

## Detectors (`--detector`)

The detection backend is selectable. Faces are detected in a **single pass over
the whole frame** (one-shot), so every person is found at once instead of the
window-scanning, flickering behaviour of the old HOG path.

| `--detector`  | What it detects | Notes |
| ------------- | --------------- | ----- |
| `mediapipe` (default) | Faces (Google MediaPipe / BlazeFace) | Best on top-down aerial group shots. Needs `mediapipe` + the `.tflite` model. |
| `yunet`       | Faces (OpenCV YuNet ONNX) | No extra pip install beyond the `.onnx` model. Good fallback. |
| `hog`         | Full bodies (legacy OpenCV HOG) | Window-scan; weak on top-down views (~32.5% hit rate in our test). Kept for comparison. |

Example:

```bash
python visionproject/shot_capture_experiment.py --detector mediapipe --source webcam
python visionproject/shot_capture_experiment.py --detector yunet    --source webcam
```

## Step 1-3: head-count match (count mode)

The `count3` mode covers steps 1-3 of the program logic:

1. **Set the number of people** — `--target-persons N`, or adjust live with `-` / `=`.
2. **Detect the count from the camera** — one-shot face detection per frame.
3. **Verify detected == target** — the overlay shows `Count: n/N [MATCH|UNDER|OVER]`,
   and each detected person is **labeled** with a numbered box (`P1`, `P2`, …),
   ordered left-to-right within the frame. When the count matches and stays
   stable for `--stable-frames`, it auto-captures.

```bash
python visionproject/shot_capture_experiment.py \
  --detector mediapipe --start-mode count3 --target-persons 4
```

Modes:

- `manual`: user decides when to capture
- `person1`: auto-capture when one person is detected in a stable and usable framing
- `count3`: auto-capture when exactly 3 people are detected
- `ratio`: auto-capture when total person bbox area ratio is inside a target range

## What "person1" means

`person1` is the new default mode.

It auto-captures only when all conditions hold for several frames:

- exactly 1 person is detected
- the person bbox area ratio is inside a target range
- the person center is close to the frame center
- the person bbox is not touching the frame borders

This is the closest simple baseline for:

"If one person is properly inside the frame, capture automatically."

## Recommended setup

### Android mirror

```bash
scrcpy --max-size=1080 --max-fps=30 --window-title HCI_phone
```

Then run the script on the mirrored phone region:

```bash
conda activate cv
python visionproject/shot_capture_experiment.py \
  --source screen \
  --screen-region 100,80,420,900 \
  --crop 20,120,380,680
```

`--screen-region` should cover the whole mirrored phone window.

`--crop` should cover only the camera preview area inside the phone UI. This is important because status bars and buttons can disturb the detector.

### Webcam test

```bash
conda activate cv
python visionproject/shot_capture_experiment.py --source webcam --webcam-index 0
```

## Key controls

- `1`: switch to manual mode
- `2`: switch to `person1` mode
- `3`: switch to `count3` mode
- `4`: switch to `ratio` mode
- `-`: decrease target person count
- `=` (or `+`): increase target person count
- `c` or `space`: capture immediately
- `q`: quit

## Useful tuning for person1

Default values:

- `--one-person-ratio-min 0.10`
- `--one-person-ratio-max 0.35`
- `--center-tolerance 0.18`
- `--border-margin-ratio 0.05`
- `--stable-frames 10`
- `--cooldown-sec 2.0`

Meaning:

- `one-person-ratio-min/max`: how large the detected person should appear
- `center-tolerance`: how far the bbox center may drift from the frame center
- `border-margin-ratio`: how much free space must remain from each border
- `stable-frames`: how long the condition must hold before auto-capture
- `cooldown-sec`: prevents repeated captures

If auto-capture is too strict:

- reduce `--stable-frames`
- increase `--center-tolerance`
- widen `--one-person-ratio-min/max`

If it captures too easily:

- increase `--stable-frames`
- reduce `--center-tolerance`
- increase `--border-margin-ratio`

## Output

Each run creates a new folder under `visionproject/captures_compare/<timestamp>/`.

Inside it:

- `manual/`
- `person1/`
- `count3/`
- `ratio/`
- `captures.csv`

Each capture stores:

- raw frame
- overlay frame with boxes and status text
- one CSV row with mode, trigger type, person count, area ratio, and elapsed time

## Suggested comparison flow

If you want a quick first experiment, compare these three:

1. `manual`
2. `person1`
3. `ratio`

That gives you:

- full user control
- simple single-subject auto capture
- composition-size based auto capture

## Detector notes

The default detector is now **MediaPipe (Google) Face Detection**, with **YuNet**
as a no-extra-install fallback. Both are one-shot face detectors and handle the
top-down aerial composition far better than the legacy `--detector hog` path,
which our test showed only recognised a person ~32.5% of the time.

The `hog` backend is kept only for comparison. Face detection assumes faces are
visible (which matches the redesigned composition criterion in the slides:
"얼굴이 보이는 구도").
