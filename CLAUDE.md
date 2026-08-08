# MirrorFit — Full-body virtual try-on + brand-aware fit engine

## What this is
Browser-based AR apparel try-on: MediaPipe Pose + Segmentation (WASM, on-device)
drive a Three.js render layer over the live camera; a fit engine maps measured
body landmarks against per-brand size charts to recommend sizes. Backend is
FastAPI + Postgres + S3/CloudFront, deployed with AWS CDK. Later phases package
the same web app as an in-store smart-mirror kiosk (QR/RFID-triggered).

- Master plan: docs/OPERATION_PLAN.md  ← always check current stage/task there
- Full spec: docs/MirrorFit_Advanced_Documentation.pdf
- Working reference implementation of the pipeline pattern: web/reference/glasses-tryon.html
  (face-based, but the capture→track→anchor→render→smooth structure is the template)

## Current stage
Stage 2 (see OPERATION_PLAN.md). Update this line as stages complete.

## How to work with me (IMPORTANT — I am learning this stack)
- I know Python/FastAPI/AWS/DevOps well. I am LEARNING JS, Three.js, MediaPipe, 3D math.
- Write the implementation directly rather than leaving TODOs for me to fill in —
  we moved off the skeleton-first workflow because in practice I kept asking you
  to just implement it.
- Still explain the concept before/after implementing, and keep diffs small and
  readable — I'm learning the "why," even if I'm not typing every line myself.
- After each accepted change, add a short "What you should understand from this" note.
- When I paste a bug, guide me to the cause with questions before showing the fix.
- Weekly: when I ask for a "week review", walk my code and refactor WITH me,
  explaining every change.

## Conventions
- Prototype: vanilla JS ES modules, no framework, no build step
- Three.js + @mediapipe/tasks-vision from CDN (versions pinned in each file)
- Serve locally: `bash scripts/serve.sh` (python http.server on :8000)
- Phone testing: HTTPS tunnel (ngrok) — getUserMedia requires secure context
- Backend: Python 3.11+, FastAPI, SQLAlchemy; `pip install -r backend/requirements.txt`
- Keep secrets out of the repo; .env is gitignored

## Repo map
- web/stage0..stage3/  — learning-stage files (skeletons with TODOs)
- web/reference/       — known-good reference implementation (do not "improve" it)
- web/assets/          — garment PNGs, test models
- backend/             — FastAPI app (Stage 4+)
- infra/               — CDK app (Stage 4+)
- docs/                — plan + spec + accuracy notes
