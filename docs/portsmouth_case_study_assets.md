# Portsmouth Case-Study Assets

The public Portsmouth evidence package is generated from completed production
runs. The marketing frontend reads only committed static assets and never reads
run HDF5 files or calls the serving API.

## Regenerate Scientific Assets

Run from WSL at the repository root. Use the host user's numeric ID so Docker
does not leave root-owned files in the checkout.

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -v "$PWD:/workspace" \
  -v /mnt/c/FGNServing/model_bundle:/model_bundle:ro \
  -v /mnt/c/FGNServing/artifacts:/artifacts:ro \
  -v /mnt/c/FGNServing/case_study_sources:/case_study_sources:ro \
  -w /workspace \
  fgn-serving-python:local \
  python -m neuralop.flood.serving.case_study_export \
  --config /case_study_sources/portsmouth_export_config.json
```

The export invalidates prior hero videos and writes a numbered, map-only Irene
sequence. Encode that sequence immediately after the scientific export:

```bash
python3 neuralop/flood/serving/case_study_video.py \
  --manifest apps/fgn-serving-frontend/public/marketing/portsmouth/manifest.json \
  --ffmpeg "$(command -v ffmpeg)"
```

`--ffmpeg` may also point to a Windows FFmpeg executable under `/mnt/c`; the
encoder stages output on the Windows filesystem and publishes it atomically
back into WSL. Intermediate hero frames are removed only after both MP4 and
WebM outputs succeed. Pass `--keep-frames` only for rendering diagnostics.

## Validate

```bash
cd apps/fgn-serving-frontend
npm run validate:marketing
npm run build
```

The validation checks the evidence provenance, frame milestones, animation
duration, asset presence, and media size budgets. A regenerated evidence package
must not be published unless these checks and the case-study rendering tests pass.
