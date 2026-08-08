# MirrorFit — Operation Code Plan

> The single source of truth for what to build, in what order, with acceptance
> criteria and ready-to-use Claude Code prompts per task. Check boxes as you go.
> Full product/technical spec: `MirrorFit_Advanced_Documentation.pdf` (keep in docs/).

**Cadence:** ~8–10 focused hrs/week alongside full-time work.
**Rule:** every stage ends with something that RUNS. No stage is "done" until its
acceptance criteria pass on your machine.

---

## Stage 0 — Browser & render-loop fundamentals (Week 1–2)

Goal: own the render loop. Everything in this project sits on it.

- [x] **0.1 Webcam on canvas** — `web/stage0/webcam-canvas.html`
  - getUserMedia → `<video>` → draw each frame to `<canvas>` via requestAnimationFrame
  - Acceptance: live mirrored feed at 30fps, FPS counter drawn on canvas
  - Claude Code prompt: *"Explain requestAnimationFrame vs setInterval, then review
    my TODO attempt in web/stage0/webcam-canvas.html. Don't rewrite it — point at
    problems and ask me questions."*
- [x] **0.2 Moving overlay** — extend 0.1
  - Draw a circle that follows the mouse over the video; then one that bounces
  - Acceptance: no flicker, no ghosting (you understand clearRect and draw order)
- [ ] ~~**0.3 Concept check**~~ (no code) — SKIPPED 2026-08-08, moved straight to
  Stage 1. explain to Claude Code in your own words: event loop, async/await, why
  the video element isn't drawn by the browser but by you
  - Acceptance: Claude Code agrees your explanation has no gaps

## Stage 1 — Three.js foundations (Week 3–4)

Goal: scene/camera/renderer trio + transforms, the rendering half of the product.

- [x] **1.1 First scene** — `web/stage1/three-basics.html`
  - Rotating cube, ambient + directional light, resize handling
  - Acceptance: cube spins; you can explain what the camera frustum is
- [x] **1.2 Transforms lab** — same file, keep going
  - Move cube with mouse; scale with wheel; rotate with quaternion (not Euler)
  - Acceptance: you can say why quaternions beat Euler angles (gimbal lock) in 3 sentences
- [x] **1.3 glTF loading**
  - Load any free glTF from the web, orbit it, normalize its size with Box3
  - Acceptance: model loads, is centered, fits view regardless of source scale
- [x] **1.4 Transparent overlay scene**
  - Three.js canvas (alpha:true) stacked over the Stage 0 webcam feed
  - Acceptance: cube floats over your live video — this IS the product's layer stack

## Stage 2 — MediaPipe Pose (Week 5–6)

Goal: the perception half. 33 landmarks, live, on you.

- [x] **2.1 Skeleton overlay** — `web/stage2/pose-skeleton.html`
  - Tasks Vision PoseLandmarker (VIDEO mode), draw dots + bones on canvas
  - Acceptance: skeleton tracks you at 25+fps; landmarks labeled by index
- [x] **2.2 Smoothing**
  - Implement One-Euro filter per landmark (Claude Code explains the math first)
  - Acceptance: visible jitter reduction with a toggle to compare raw vs filtered
- [x] **2.3 Segmentation**
  - Add person segmentation; tint your silhouette; understand the mask
  - Acceptance: mask follows you; you can explain how it will drive occlusion
- [x] **2.4 Measurements v0**
  - Shoulder width + height estimate from world landmarks, printed on screen
  - Acceptance: numbers are stable when you stand still (rolling average works)

## Stage 3 — P1 prototype: Tier 1 garment try-on (Week 7–9)

Goal: THE demo. Garment PNG warped to your body, live.

- [ ] **3.1 Rigid anchor** — `web/stage3/tryon-tier1.html`
  - Rectangle/plane anchored shoulders→hips, scaled by shoulder width
  - Acceptance: rectangle follows torso through lean/step/turn(±30°)
- [ ] **3.2 Control-mesh warp**
  - Garment PNG on a subdivided plane; bind vertices to landmark positions
  - Acceptance: test garment (make a simple PNG) deforms plausibly with movement
- [ ] **3.3 Occlusion + polish**
  - Use segmentation mask so crossed arms render in front of garment; fade garment
    beyond ±35° yaw (honest Tier 1 limit)
  - Acceptance: arms-crossed test passes; rotation fade feels intentional
- [ ] **3.4 Ship the demo**
  - Garment picker (3 PNGs), snapshot button, deploy to S3+CloudFront
  - Acceptance: public URL works on your phone; 10 testers give feedback
  - **← THIS is the "P1 exit gate" from the spec. Celebrate, then user-test.**

## Stage 4 — Fit engine v1 + backend (Week 10–13)  [your home turf]

- [ ] **4.1 Measurement engine** — height-calibrated vector w/ confidence bands
- [ ] **4.2 Chart schema + ingestion** — Postgres per spec §4.6; ingest 3 real
      brand size charts by hand into the canonical schema
- [ ] **4.3 Recommendation** — per-zone deltas → fit score → size + plain-language
      note; fit-strip UI in the web app
- [ ] **4.4 FastAPI + CDK deploy** — endpoints per spec §4.6; events pipeline stub
- [ ] **4.5 Accuracy study** — n=10 friends/family, tape measure vs engine;
      publish honest error bands in docs/accuracy.md
- Acceptance: end-to-end — scan yourself, pick garment, get size rec with reasoning

## Stage 5 — Kiosk pilot prep (P4)  [later]

- [ ] Chrome kiosk image on a mini PC; QR garment trigger; save-look handoff;
      Prometheus health metrics (fps, tracking quality) — your DevOps skills, reused
- Acceptance: appliance runs unattended for 48h; you can see its health remotely

## Stage 6 — Live 360° rotation (P5)  [hardest R&D — do NOT start early]

- [ ] Tier 2: one 3D garment (CLO3D trial or purchased asset), PBD cloth proxy
- [ ] Yaw estimation from shoulder geometry; confidence-gated bridging; swap suppression
- [ ] Turn-replay capture
- Acceptance: 9/10 clean full turns (spec's P5 gate)

---

## Working rules with Claude Code (also encoded in CLAUDE.md)

1. Plan mode for anything non-trivial. Read the plan, comment, then approve.
2. Never auto-accept. Read every diff. If a diff surprises you, ask "why" before accepting.
3. Skeleton-first: ask for TODO skeletons, write the code yourself, then request review.
4. End each week: "Review everything I wrote this week. Refactor WITH me, explaining each change."
5. Stuck >45 min on a bug: paste code + symptom, ask to be Socratically guided, not fixed.

## Definition of done (project-level, P1 scope)

Public URL, phone + laptop, 30fps Tier 1 try-on with 3 garments, occlusion working,
snapshot working, 10-tester feedback collected, code you can explain line by line.
