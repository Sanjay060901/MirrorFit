# MirrorFit — Full-body virtual try-on + brand-aware fit engine

## What this is
Live 3D virtual try-on: a parametric body twin (SMPL-family) fitted to the
shopper and posed from MediaPipe Pose, wearing pattern-graded garments
simulated with a GPU XPBD cloth solver — so wrong-size behavior (pulling,
gaping, pooling) is emergent physics, not a scripted effect. A fit engine maps
measured landmarks against per-brand size charts. Backend is FastAPI +
Postgres + S3/CloudFront, deployed with AWS CDK. Later phases package it as an
in-store smart-mirror kiosk (QR/RFID-triggered, real GPU, depth camera).

**Architecture pivot 2026-08-10** — see OPERATION_PLAN.md Part II. The earlier
"Tier 1 flat PNG warp" framing is superseded: it's now a learning step and
possible degraded phone tier, not the product. Realistic cost of the new
target: ~18–24 months to pilot at a part-time solo pace.

- Master plan: docs/OPERATION_PLAN.md  ← always check current stage/task there
- Full spec: docs/MirrorFit_Advanced_Documentation.pdf
- Working reference implementation of the pipeline pattern: web/reference/glasses-tryon.html
  (face-based, but the capture→track→anchor→render→smooth structure is the template)

## Current stage
Stage 6.5 — live try-on works end to end (see OPERATION_PLAN.md).
Update this line as stages complete.

## How to work with me (IMPORTANT — I am learning this stack)
- I know Python/FastAPI/AWS/DevOps well. I am LEARNING JS, Three.js, MediaPipe, 3D math.
- The Part II architecture also requires: linear algebra for skinning, GPU
  compute shaders (WebGPU), cloth simulation theory, and CLO3D pattern
  authoring. Assume I'm new to all four — teach before building.
- Write the implementation directly rather than leaving TODOs for me to fill in —
  we moved off the skeleton-first workflow because in practice I kept asking you
  to just implement it.
- Still explain the concept before/after implementing, and keep diffs small and
  readable — I'm learning the "why," even if I'm not typing every line myself.
- After each accepted change, add a short "What you should understand from this" note.
- When I paste a bug, guide me to the cause with questions before showing the fix.
- Weekly: when I ask for a "week review", walk my code and refactor WITH me,
  explaining every change.
- Start each planned session in plan mode; show me the plan before building.
- One session per numbered block in the roadmap. Never batch two — each has its
  own acceptance evidence, and mixing them is how you get "it ran" instead of
  "it works".

## Verification (non-negotiable — see docs/TESTING.md)
- Before saying something works, state whether you observed it in MY
  configuration or only that code executed without throwing. If you have not
  seen real numbers from my actual setup, say so plainly.
- For every test you add, prove it fails when the bug is present: reintroduce
  the bug, show the failure, restore, show the pass. Note it in the commit.
- Test fixtures must reproduce the live transform chain (scaled, mirrored
  parent), and assert against measured ground truth — not against whatever the
  code currently outputs.
- A fallback that changes results must throw or set a visible degraded flag.
  console.warn is not a failure mode.

## Conventions
- Prototype: vanilla JS ES modules, no framework, no build step
- Three.js + @mediapipe/tasks-vision from CDN (versions pinned in each file)
- Serve locally: `bash scripts/serve.sh` (python http.server on :8000)
- Phone testing: HTTPS tunnel (ngrok) — getUserMedia requires secure context
- Backend: Python 3.11+, FastAPI, SQLAlchemy; `pip install -r backend/requirements.txt`
- Keep secrets out of the repo; .env is gitignored

## Repo map
- web/index.html       — dev menu linking every stage
- web/stage0..stage6/  — one self-contained page per learning stage
- web/lib/             — shared modules (cloth.js: capsules, drafting, GPU XPBD solver)
- web/tests/           — headless checks; run before touching capsule/drafting code
- web/reference/       — known-good reference implementation (do not "improve" it)
- web/assets/          — bodytwin.glb, bodyfit.json, garment PNGs
- scripts/bodytwin/    — Anny -> glTF exporter (regenerates web/assets/bodytwin.glb)
- backend/             — FastAPI app (Stage 4+)
- infra/               — CDK app (Stage 4+)
- docs/                — plan, spec, testing doctrine, decisions

The live try-on is web/stage6/tryon-live.html — camera, twin and cloth together.
