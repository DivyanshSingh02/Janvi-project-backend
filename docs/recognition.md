# Face recognition: FaceNet CPU backend (Stage 6B)

> **A face match is a lead, not proof of identity. A human must review every candidate before any action is taken.**
> Similarity scores are not identity probabilities. Do not make automated decisions from them.

## Architecture and contract

The application uses `RecognitionService` from `app/recognition/base.py`. The real implementation is
`FaceNetRecognitionService` in `app/recognition/facenet.py`; `UnconfiguredRecognitionService` (answers 503) stays the
default (`RECOGNITION_BACKEND=none`). A fake service exists only in `tests/fake_recognition.py`.

The FaceNet backend uses MTCNN (`facenet-pytorch`) for detection and InceptionResnetV1 with VGGFace2 weights for
512-dimensional embeddings, on the CPU. What happens to one image:

1. Pillow decodes it, applies the EXIF orientation and converts it to RGB (transparency is flattened onto white).
   Bytes that are not a usable image (corrupt, truncated, decompression bomb) have **no face**: `extract_faces()` returns
   `[]` and the existing code treats the photo as "no face detected" instead of failing the whole case.
2. MTCNN looks for faces. Images whose long side exceeds 2000 px are shrunk **for detection only**; the minimum face
   size handed to MTCNN is scaled the same way, boxes are mapped back, and the gates, the reported boxes and the 160x160
   crops all use the original image.
3. Two quality gates (below) drop faces. **All faces that pass are returned**, biggest first. The one-face rule for
   enrolment and search stays in `extract_single_face()`, so a later video stage can analyse frames with several faces.
4. The crops go through the network in batches of 16; each vector is returned as `float32`, L2-normalised
   (`app/recognition/vectors.py`), together with its box and detection probability.

Only one image is processed at a time per process (a lock): bounded CPU and memory, and no races on the detector.
Every worker process of a multi-worker deployment holds its own copy of the models (roughly 0.5 GB of RAM each).

| Setting | Default | Meaning |
| --- | ---: | --- |
| `RECOGNITION_BACKEND` | `none` | `none` or `facenet` |
| `RECOGNITION_MODEL_DIR` | `models/recognition` | Folder holding `facenet-vggface2.pt` (relative to where the API is started; use an absolute path in production) |
| `RECOGNITION_MIN_FACE_SIZE` | `40` | Smallest accepted face box side, in pixels of the original image (12 to 1000) |
| `RECOGNITION_MIN_DETECTION_PROBABILITY` | `0.90` | Smallest accepted MTCNN detection probability (above 0, at most 1) |

These are image-quality gates, not identity thresholds. Invalid values are refused when the settings load.

## Install (Windows, Python 3.11, CPU)

`requirements.txt` stays light. The optional file `requirements-facenet.txt` (it includes `requirements.txt`) pins
`torch==2.6.0`, `torchvision==0.21.0`, `requests==2.32.3`, `tqdm==4.67.1` and `facenet-pytorch==2.6.0`.

**Do not run `pip install -r requirements-facenet.txt` directly.** As far as I know, `facenet-pytorch==2.6.0`'s
metadata still asks for `numpy<2`, `Pillow<10.3`, `torch<2.3` and `torchvision<0.18`, which conflict with this project's
pins (numpy 2.1.3, Pillow 11.3.0): pip would refuse or downgrade them. Nothing is downgraded; install in three steps.
Step 2 installs PyTorch **with** its own dependencies; step 3 skips only facenet-pytorch's stale bounds:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 requests==2.32.3 tqdm==4.67.1
.\venv\Scripts\python.exe -m pip install --no-deps facenet-pytorch==2.6.0
.\venv\Scripts\python.exe -c "import numpy, PIL, torch, torchvision, facenet_pytorch; print(numpy.__version__, PIL.__version__, torch.__version__, torchvision.__version__)"
```

Expected: `2.1.3 11.3.0 2.6.0 0.21.0` (numpy and Pillow unchanged). On Windows the PyPI wheels of torch and torchvision
are CPU builds, so no extra index is needed; all packages have Windows wheels for Python 3.11, and facenet-pytorch is a
pure-Python wheel. `pip check` will keep reporting facenet-pytorch's stale bounds: that is the documented, deliberate
exception. If `import torch` fails with a DLL error, install the current Microsoft Visual C++ Redistributable (x64).

MTCNN's own small detector weights ship inside the facenet-pytorch package (check:
`python -c "import facenet_pytorch, pathlib; print(sorted(p.name for p in (pathlib.Path(facenet_pytorch.__file__).parent/'data').glob('*.pt')))"`
should list `onet.pt pnet.pt rnet.pt`).

## Download the weights once, explicitly

```powershell
.\venv\Scripts\python.exe -m scripts.download_recognition_model --check
```

It downloads the `20180402-114759-vggface2.pt` release asset of the facenet-pytorch project on GitHub, saves it as
`facenet-vggface2.pt` in `RECOGNITION_MODEL_DIR` (or `--directory`), and prints the size and the SHA-256 of the file it
saved, computed from the file on disk. **No checksum is hard-coded** (none could be verified): write the printed one down
and compare it when you copy the file between machines. It never replaces an existing file without `--force`, and an
interrupted or too-small download leaves nothing behind. `--check` additionally loads the weights into the network
(needs the install above); `--check-only` does that for a file already in place, without downloading.
The API itself never downloads anything and loads the weights with `weights_only=True`, so the file cannot run code.

**Licence.** The pretrained weights derive from the **VGGFace2** dataset. The project's MIT licence covers its code, not
necessarily the dataset or the weights, and dataset terms and availability have changed over time. Read the current terms
of VGGFace2 and of this checkpoint and obtain approval **before any real deployment** or use with real people's images.

## Enable it

```dotenv
RECOGNITION_BACKEND=facenet
RECOGNITION_MODEL_DIR=models/recognition
```

Start the API as usual. Behaviour you can rely on:

- Nothing heavy happens at start-up. The models load on first use, under a lock. A missing or unreadable weights file or a
  missing or broken PyTorch raises `RecognitionUnavailableError`: the API starts and answers **503** for recognition,
  with a generic message (paths and library text never reach the response or the logs; the log carries a reason code
  such as `weights_missing`, `packages_missing`, `weights_invalid`).
- `/health` reports recognition `ready` only when the models are loaded and usable, otherwise `unconfigured` (the API
  stays healthy). The first call that finds everything in place loads the models (a few seconds); later calls never
  reload them. A health check never waits for a load that another request has started: it says "not ready yet".
- A missing file or package is re-checked cheaply on every call, so **no restart is needed** after you download the
  weights or install the packages. A failed heavy load (corrupt file) is retried at most every 30 seconds.
  Replacing the weights file of a running, already loaded process needs a restart.
- `model_info()`: name `facenet-vggface2`, dimension 512, version `facenet-pytorch-<package version>-vgg2-<first 12 hex of
  the weights file's SHA-256>-p1` (at most 50 characters, the database column limit; `p1` is the preprocessing revision).
  A different weights file, package version or preprocessing therefore never shares embeddings with old ones.
- Approving a case enrols its photos inside that request: expect seconds, not milliseconds, per photo on a CPU.

## Switching backend or weights: re-embedding

Embeddings carry the model name and version; search skips embeddings of any other version, so after a switch **searches
find nothing until the approved cases are re-enrolled**. On the server, from the project root:

1. Back up the database and the weights file.
2. Install the packages, download the weights, set `RECOGNITION_BACKEND` and `RECOGNITION_MODEL_DIR`, restart the API.
3. Check `GET /health` shows `"recognition": "ready"` (the first call may take a few seconds).
4. Run once (during a quiet period: it holds the CPU):

   ```powershell
   .\venv\Scripts\python.exe -m scripts.reembed_cases
   ```

5. Read the counts, e.g. `cases=3 cases_skipped=0 cases_failed=0 photos_done=7 photos_failed=1 photos_pending=0`.
   They are counts only (no names, ids or file names). `photos_failed` are photos without exactly one usable face
   (or whose file is missing) with the new model; they are not searchable until that is fixed (their old vectors, if any,
   belong to the old model and are ignored by search). Exit code 1 means the model was unavailable or `cases_failed` is above zero; run it
   again (finished photos are skipped, so repeating is harmless).
6. Run a controlled search with consented fictional or test images and check the human-review workflow. Do not compare
   scores between models.

Old vectors of a photo are **replaced** when the new embedding succeeds, so switching back means re-embedding again.

## Evaluate on your own data

Use a public dataset whose licence allows it (for example LFW) or photos with explicit informed consent. Never real case
photos, never committed (`evaluation-datasets/` is git-ignored). One folder per person:

```powershell
.\venv\Scripts\python.exe -m scripts.evaluate_recognition evaluation-datasets
.\venv\Scripts\python.exe -m scripts.evaluate_recognition evaluation-datasets --target-far 0.01,0.001 --max-images-per-person 10
.\venv\Scripts\python.exe -m scripts.evaluate_recognition evaluation-datasets --thresholds 0.5,0.6,0.7
```

Every image goes through the same one-face rule as the API; images with no usable face, several faces or that cannot be
read are counted and skipped (a model failure stops the run instead). All pairs of usable images are scored with cosine
similarity. It prints counts, the same-person and different-person distributions (min, 5th percentile, median, mean,
95th percentile, max; percentiles are exact to 0.00025) and, for threshold candidates, the false accepts (different
people at or above the threshold) and false rejects (same person below it), exactly counted, e.g.
`threshold=0.6120 false_accept=12/4000 (0.0030) false_reject=45/900 (0.0500)`. Candidates are the lowest threshold for
each target false-accept rate (only when the data has at least 1/rate different-person pairs) and the equal-error point,
or your own `--thresholds`. It reports what it measured and makes **no accuracy claim** and chooses nothing for you.
Pairs share images and are not independent, so there are no confidence intervals.

## Known limitations

- Large age gaps (the typical missing-person case), hairstyle or appearance changes, pose, occlusion and low resolution
  reduce similarity; poor light, motion blur and camera quality reduce detection and embedding quality.
- Masks and other face coverings hide identity-relevant features: expect misses and misleading similarities.
- Accuracy can differ between groups of people (skin tone, age, sex, ethnicity) and between datasets; aggregate numbers hide
  this. Evaluate on the population and conditions you will actually see.
- The detector probability is not identity confidence; cosine similarity (including its display as
  `similarity_percent`) is not a probability.
- **Thresholds are not universal and must be calibrated on your own representative, lawfully collected data.** No
  threshold in this project is validated.
- CPU inference is slow; search is an exact brute-force comparison in memory.
- **Results are leads for human review only.** Nothing here can establish identity.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Recognition endpoints 503, log reason `weights_missing` | Weights not downloaded, or `RECOGNITION_MODEL_DIR` points elsewhere (relative path resolved from another folder) |
| `packages_missing` | Install steps above not done in this venv |
| `packages_unusable` | `import torch` fails: missing Visual C++ Redistributable, or a broken install |
| `weights_invalid` | Truncated or wrong file: run `--check-only`; re-download with `--force` |
| `/health` stays `unconfigured` | Any of the above; read the warning in the API log |

## What was not verified when this stage was written

No PyTorch, facenet-pytorch, PostgreSQL or network was available in the authoring environment. The loading code was
tested with fake torch modules; real MTCNN and InceptionResnetV1 behaviour (including that the checkpoint holds
`logits.*` tensors that must be dropped, and that facenet-pytorch 2.6.0 runs on numpy 2.1.3 / Pillow 11.3.0 / torch 2.6.0)
is checked only by `tests/test_facenet_torch.py` once torch is installed, and by `--check`. The download URL and the
facenet-pytorch 2.6.0 metadata are recalled from the project, not fetched.
