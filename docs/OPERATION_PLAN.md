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

## Stage 3 — Tier 1 garment try-on (Week 7–9)

Goal: ~~THE demo~~ — **reframed 2026-08-10 (see Part II):** this is now a
learning step and a possible degraded phone-tier renderer, not the product.
A flat warped PNG has a hard realism ceiling no amount of tuning fixes; that
ceiling is exactly why the project moved to body-twin + simulation.
Finish it for the skills (landmark anchoring, mesh deformation, occlusion,
compositing) — all of which Stages 5–7 build on — then move on.

- [x] **3.1 Rigid anchor** — `web/stage3/tryon-tier1.html`
  - Rectangle/plane anchored shoulders→hips, scaled by shoulder width
  - Acceptance: rectangle follows torso through lean/step/turn(±30°)
- [x] **3.2 Control-mesh warp**
  - Garment PNG on a subdivided plane; bind vertices to landmark positions
  - Acceptance: test garment (make a simple PNG) deforms plausibly with movement
- [x] **3.3 Occlusion + polish**
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

---

# Part II — Body-twin + pattern-true simulation (the core product)

> **Decision, 2026-08-10:** the project's target architecture is no longer
> "Tier 1 flat warp, with 3D as a distant maybe." The product is a live 3D
> body twin wearing pattern-graded, physically simulated garments, where
> wrong-size behavior is the *truthful output of simulation*, not a scripted
> effect. Stages 0–4 are unchanged and remain load-bearing (tracking,
> measurement, fit engine, backend all feed this). Stage 3's flat warp is now
> explicitly a throwaway learning step, not the P1 deliverable.
>
> **Honest cost of this decision:** ~18–24 months to pilot quality at a solo,
> part-time pace, and it requires four skills not currently in the stack:
> linear algebra for skinning, GPU compute shaders, cloth simulation theory,
> and CLO3D pattern authoring. Nothing below ships a user-facing demo quickly.
> This is a deliberate trade of time-to-demo for defensibility.

## The four pillars

1. **3D body twin, tracked live** — a parametric human mesh (SMPL/SMPL-X
   family): ~10 shape parameters fitted from the measured vector + segmentation
   silhouette, posed live from tracked landmarks. Gives volume instead of a
   skeleton, and solves occlusion for free (render depth-only as a mask).
2. **Pattern-true garments, graded per size** — don't model "the dress," model
   *each size* from its real graded pattern (CLO3D/Marvelous Designer), with
   measured fabric properties (stretch, weight, bending stiffness). Size M is
   genuinely different geometry from size L, so tightness/gaping/pooling
   emerges from physics rather than being authored.
3. **Real-time cloth on GPU** — XPBD solver as GPU compute: coarse sim mesh
   (2–5k verts/garment) colliding against the body twin, high-detail render
   mesh skinned to it. Strain limiting is what makes tight *look* tight.
4. **Fit made visible** — the solver knows per-triangle strain; surface it as a
   live tension view (tinted where fabric is at stretch limit). This is the
   demo that differentiates from every flat-overlay competitor.

## Stage 5 — SMPL body twin (fitting + live driving)

- [ ] **5.1 Linear algebra + skinning fundamentals** — learn LBS (linear blend
      skinning), joint hierarchies, blend shapes. No product code yet.
- [ ] **5.2 Load and pose a body model** — get an SMPL-family mesh rendering;
      pose it from hardcoded joint rotations, then from live Stage 2 landmarks
- [ ] **5.3 Shape fitting** — fit shape params from the Stage 2.4 measurement
      vector + segmentation silhouette; stabilize across frames
- [ ] **5.4 Depth-only occlusion** — render the twin to a depth mask so real
      limbs correctly occlude anything drawn behind them
- Acceptance: an invisible, metrically-plausible 3D body moves with you in
  camera space; occlusion works without the Stage 3.3 arm-tube hack
- **Licensing gate:** SMPL/SMPL-X have specific licenses (research vs
  commercial). Resolve which model family is legally usable for a commercial
  product BEFORE building on it — this can invalidate the whole approach.

## Stage 6 — XPBD cloth solver

- [ ] **6.1 Cloth simulation theory** — position-based dynamics, constraints,
      substepping, why XPBD over mass-spring. Reading, not code.
- [ ] **6.2 GPU compute pipeline** — WebGPU compute shaders (prototype here
      first: if it holds 60fps, the one-codebase web-first advantage survives;
      if not, the appliance moves to a native engine and phones get a
      degraded web tier)
- [ ] **6.3 Flag on a plane** — simplest possible cloth, pinned corners, wind.
      Prove the solver before involving a body.
- [ ] **6.4 Garment on a static body** — collision against the Stage 5 twin
- [ ] **6.5 Garment on the live body** — full pipeline, 60fps target
- Acceptance: a simulated garment drapes plausibly on your moving body twin at
  60fps, without explosion/jitter/tunneling

## Stage 7 — Graded-size assets + fit visualization

- [ ] **7.1 Garment digitization pipeline** — CLO3D authoring workflow, fabric
      parameter measurement, per-size export. This is a *first-class company
      capability*, not a one-off task.
- [ ] **7.2 One garment, three real sizes** — same SKU graded S/M/L from real
      patterns
- [ ] **7.3 Strain visualization** — per-triangle strain → live tension view
- [ ] **7.4 Wrong-size demo** — select M on a body that needs L; verify the
      pulling/riding-up is emergent, not scripted
- Acceptance: on one real SKU, three sizes visibly and *truthfully* differ on
  the same body; tension view reads clearly to a non-expert
- **Business gate:** this stage needs real graded pattern data. That means a
  brand partner willing to share patterns — a BD problem, not an engineering
  one. Start that conversation early; it's the long pole.

## Stage 8 — Kiosk pilot prep (was Stage 5)

- [ ] Appliance with a real GPU (this architecture is why the kiosk hardware
      earns its cost); depth camera to lock down shape estimation
- [ ] QR garment trigger; save-look handoff; Prometheus health metrics
      (fps, tracking quality, sim stability) — your DevOps skills, reused
- Acceptance: appliance runs unattended for 48h; health visible remotely

---

## Working rules with Claude Code (also encoded in CLAUDE.md)

1. Plan mode for anything non-trivial. Read the plan, comment, then approve.
2. Never auto-accept. Read every diff. If a diff surprises you, ask "why" before accepting.
3. Skeleton-first: ask for TODO skeletons, write the code yourself, then request review.
4. End each week: "Review everything I wrote this week. Refactor WITH me, explaining each change."
5. Stuck >45 min on a bug: paste code + symptom, ask to be Socratically guided, not fixed.

## Definition of done (project-level)

**Superseded 2026-08-10.** The old P1 gate (public URL, 3 flat garments,
10 testers) is no longer the project's definition of done — see Part II.
The real gate is now Stage 7: one real SKU, three real graded sizes, visibly
and truthfully different on a live body twin, with a readable tension view.

Stage 3's flat-warp demo is kept as a learning artifact and a fallback
phone-tier renderer, not as the product.
