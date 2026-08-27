// MirrorFit — GPU XPBD garment solver
// ============================================================================
//
// Extracted from web/stage6/cloth-body.html so the live try-on (Stage 6.5) can
// reuse it instead of forking 500 lines. Three things live here:
//
//   1. measureCapsules()  — derive a collision proxy from the twin's mesh
//   2. draftShirt()       — generate a t-shirt's particles and constraints
//   3. GarmentSim         — the WebGPU solver that runs it
//
// WHAT CHANGED FOR THE LIVE VERSION
//
// In the static demo the body never moved, so a capsule was just two points
// and a radius, and "wider than deep" could be expressed as world x vs world
// z. Neither holds once the shopper turns around:
//
//   - capsule ENDPOINTS move every frame, so they are re-uploaded each frame
//   - the elliptical cross-section has to rotate WITH the torso, so each
//     capsule now carries its own side-axis and the ellipse is evaluated in
//     that local frame rather than in world axes
//   - the body slides underneath the cloth, so friction has to resist motion
//     RELATIVE TO THE SKIN, not motion through the room — otherwise the shirt
//     is simply left behind when you step sideways
//
// Radii, by contrast, are measured ONCE. They are body measurements; they do
// not change when you raise an arm.

import * as THREE from "three";

// Signed distance along an axis. Used before the drafting frame's own
// heightOf() closure exists.
const heightOfRaw = (v, axis) => v.dot(axis);

// Capsules are an approximation and the real mesh bulges outside them at the
// deltoids and chest. Cloth held exactly at the capsule surface gets pierced
// wherever that happens. Inflating the collision radius keeps fabric clear of
// skin, and doubles as a safety buffer against tunnelling at speed.
export const COLLISION_MARGIN = 0.018;

const MIN_RADIUS = 0.02;          // nothing may collide with a zero-radius capsule
const RADIUS_PERCENTILE = 0.95;   // see measureCapsules
const WORKGROUP = 64;
const UNIFORM_STRIDE = 256;       // WebGPU minimum dynamic-uniform alignment

// ---------------------------------------------------------------------------
// 1. Measuring the collision proxy
// ---------------------------------------------------------------------------

// Match THREE.PropertyBinding's reserved-character set: GLTFLoader strips '.'
// from node names, so Anny's "upperarm01.L" arrives as "upperarm01L".
export const norm = (s) => s.replace(/[\[\]\.:\/]/g, "");

function percentile(arr, p) {
  if (!arr.length) return 0;
  arr.sort((x, y) => x - y);
  return arr[Math.min(arr.length - 1, Math.floor(arr.length * p))];
}

// Which body part each vertex belongs to, from its dominant skin weight.
// Without this, measuring the chest sweeps in the arms hanging beside it and
// reports a torso twice as wide as it really is.
function dominantBoneNames(skinned) {
  const si = skinned.geometry.attributes.skinIndex;
  const sw = skinned.geometry.attributes.skinWeight;
  const bones = skinned.skeleton.bones;
  const out = new Array(si.count);
  for (let v = 0; v < si.count; v++) {
    let best = 0, bestW = -1;
    for (let j = 0; j < 4; j++) {
      const w = sw.getComponent(v, j);
      if (w > bestW) { bestW = w; best = si.getComponent(v, j); }
    }
    out[v] = norm(bones[best]?.name ?? "");
  }
  return out;
}

// An endpoint may be a single bone name or a LIST of them, meaning "the
// midpoint of these". That exists because the pelvis has no bone at its
// centre: Anny gives you pelvisL and pelvisR, and the only landmarks at hip
// height are the two leg joints. Without a midpoint, the only capsule you can
// build there runs SIDEWAYS between those joints — and a sideways capsule has
// no meaningful "width vs depth", so it measures the pelvis's vertical extent
// as its radius. Measured against the mesh, that came out 0.246 m wide and
// 0.144 m deep for a pelvis that is actually 0.146 x 0.094.
function endpointBones(byName, spec) {
  const names = Array.isArray(spec) ? spec : [spec];
  const bones = names.map((n) => byName.get(norm(n))).filter(Boolean);
  return bones.length === names.length ? bones : null;
}

export function averagePosition(bones, root) {
  const out = new THREE.Vector3();
  for (const b of bones) out.add(b.getWorldPosition(new THREE.Vector3()));
  return root.worldToLocal(out.divideScalar(bones.length));
}

/**
 * Measure a capsule per spec entry, in `root`'s local space.
 *
 * Works in ROOT-LOCAL space rather than world space because the live twin
 * hangs under a group with a negative x scale (the mirror). Positions
 * transform correctly through a negative-determinant matrix; rotations do not.
 * Keeping the whole simulation in model space sidesteps the problem entirely —
 * the mirror is then applied once, at render time, by the parent.
 *
 * @param spec entries of { label, a, b, owners, elliptical }, where a and b are
 *             each a bone name or an array of names meaning their midpoint
 * @returns [{ label, aBones, bBones, rx, rz, dz, elliptical }]
 */
export function measureCapsules(skinned, root, spec) {
  // updateWorldMatrix(parents, children) — NOT updateMatrixWorld(force).
  //
  // The difference caused a genuinely nasty bug. updateMatrixWorld only walks
  // DOWN, so it computed skinned.matrixWorld against whatever the parent's
  // matrixWorld happened to hold — which, before the first render, is the
  // identity. Then the very first bone.getWorldPosition() call internally does
  // updateWorldMatrix(true, false), walking UP and refreshing the parent (and
  // `root` with it) to its real 279x pixels-per-metre scale.
  //
  // Result: bone positions in the scaled chain, mesh vertices in the unscaled
  // one, and worldToLocal dividing the vertices by a 279 they were never
  // multiplied by. Every vertex collapsed to a speck at the origin, so no
  // vertex fell within any capsule's t-band and the radii silently became
  // "distance from the origin to this body part".
  //
  // Updating parents AND children puts the whole chain in one consistent state
  // before anything is read from it.
  root.updateWorldMatrix(true, true);

  const byName = new Map(skinned.skeleton.bones.map((b) => [norm(b.name), b]));
  const owners = dominantBoneNames(skinned);
  const posAttr = skinned.geometry.attributes.position;

  const toLocal = (v) => root.worldToLocal(v);

  // Morph targets live on the GPU: shape fitting changes how the body LOOKS
  // without touching geometry.attributes.position at all. Re-measuring after a
  // fit therefore has to apply the influences by hand, or it measures the
  // neutral body while the bones sit at the fitted one.
  const morphs = skinned.geometry.morphAttributes?.position ?? null;
  const influences = skinned.morphTargetInfluences ?? null;
  const morphRelative = skinned.geometry.morphTargetsRelative !== false;
  const _base = new THREE.Vector3();
  const _mv = new THREE.Vector3();
  const sampleVertex = (i, out) => {
    out.fromBufferAttribute(posAttr, i);
    if (morphs && influences) {
      _base.copy(out);
      for (let k = 0; k < morphs.length; k++) {
        const w = influences[k];
        if (!w) continue;
        _mv.fromBufferAttribute(morphs[k], i);
        // glTF morphs are deltas; a few other pipelines store absolute
        // positions, hence the flag rather than an assumption.
        if (!morphRelative) _mv.sub(_base);
        out.addScaledVector(_mv, w);
      }
    }
    return out.applyMatrix4(skinned.matrixWorld);
  };

  const v = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ap = new THREE.Vector3();
  const out = [];

  for (const s of spec) {
    const aBones = endpointBones(byName, s.a);
    const bBones = endpointBones(byName, s.b);
    if (!aBones || !bBones) { console.warn("capsule bones missing:", s.a, s.b); continue; }

    const a = averagePosition(aBones, root), b = averagePosition(bBones, root);
    ab.subVectors(b, a);
    const denom = Math.max(ab.dot(ab), 1e-9);
    const prefixes = s.owners.map(norm);

    // The torso's own side/front axes at BIND time. The mesh is in a neutral
    // upright pose here, so world x is the body's side and world z its front;
    // that is only true at bind time, which is exactly why we measure once and
    // then carry the numbers rather than re-deriving them live.
    // Gather every owned vertex once, with where it falls along the segment,
    // then choose the sampling band afterwards.
    const owned = [];
    for (let i = 0; i < posAttr.count; i++) {
      if (!prefixes.some((p) => owners[i].startsWith(p))) continue;
      sampleVertex(i, v);
      toLocal(v);
      ap.subVectors(v, a);
      const t = THREE.MathUtils.clamp(ap.dot(ab) / denom, 0, 1);
      const closest = ab.clone().multiplyScalar(t).add(a);
      owned.push({ t, d: v.clone().sub(closest) });
    }

    // Trimming the ends stops a neighbouring part's bulge inflating this
    // segment's girth — but a fixed 15% trim assumes the bone spans the body
    // part. For a SHORT bone it need not: neck01 -> head is mostly skull, so
    // the neck's own vertices all landed outside [0.15, 0.85] and the capsule
    // silently fell back to 0.35 x segment length. Measured against the mesh
    // that gave 0.036 where the truth is 0.043.
    //
    // So widen the band until there is enough to measure, and only give up
    // when even the whole segment yields nothing.
    const BANDS = [[0.15, 0.85], [0.05, 0.95], [0.0, 1.0]];
    let radial = [], sideways = [], front = [], back = [], usedBand = null;
    for (const [lo, hi] of BANDS) {
      radial = []; sideways = []; front = []; back = [];
      for (const o of owned) {
        if (o.t < lo || o.t > hi) continue;
        radial.push(o.d.length());
        // Measure each semi-axis only from vertices that actually lie near it.
        // A percentile of |d.x| over ALL vertices mixes in the front and back
        // of the torso, where |d.x| is near zero — which understates the width,
        // and is how the chest once came out deeper than it was wide.
        const ax = Math.abs(o.d.x), az = Math.abs(o.d.z);
        if (ax > az * 2.0) sideways.push(ax);
        if (az > ax * 2.0) { if (o.d.z > 0) front.push(o.d.z); else back.push(-o.d.z); }
      }
      if (radial.length >= 8) { usedBand = [lo, hi]; break; }
    }
    if (usedBand && usedBand[0] !== 0.15) {
      console.warn(`capsule ${s.label}: widened sampling band to ` +
                   `[${usedBand[0]}, ${usedBand[1]}] to find ${radial.length} vertices`);
    }

    const clamp = (r) => Math.max(r, MIN_RADIUS);
    if (radial.length < 8) {
      console.warn(`capsule ${s.label}: only ${owned.length} owned vertices, ` +
                   `falling back to segment length — RADIUS IS A GUESS`);
      const r = clamp(ab.length() * 0.35);
      out.push({ ...s, aBones, bBones, rx: r, rz: r, rx0: r, rz0: r, dz: 0, degraded: true });
      continue;
    }

    const rad = percentile(radial, RADIUS_PERCENTILE);
    if (!s.elliptical) {
      const r = clamp(rad);
      out.push({ ...s, aBones, bBones, rx: r, rz: r, rx0: r, rz0: r, dz: 0 });
      continue;
    }

    const rx = clamp(sideways.length >= 8
      ? percentile(sideways, RADIUS_PERCENTILE) : rad);

    // front/back are kept SIGNED and separate because the spine runs along the
    // BACK of the torso rather than through its centre. Measuring |d.z| would
    // report nearly the full torso depth as a radius, and leave every torso
    // capsule hugging the shopper's back. Find both extents, place the axis
    // between them, and record how far forward it had to move.
    let rz = rad, dz = 0;
    if (front.length >= 8 && back.length >= 8) {
      const f = percentile(front, RADIUS_PERCENTILE);
      const bk = percentile(back, RADIUS_PERCENTILE);
      rz = (f + bk) / 2;      // half-depth, not full depth
      dz = (f - bk) / 2;      // how far the axis must shift forward
    }
    out.push({ ...s, aBones, bBones, rx, rz: clamp(rz), dz,
               rx0: rx, rz0: clamp(rz) });
  }

  console.log("capsules:\n" + out.map((c) =>
    `  ${c.label.padEnd(11)} rx=${c.rx.toFixed(3)} rz=${c.rz.toFixed(3)} ` +
    `shift=${c.dz.toFixed(3)}`).join("\n"));
  return out;
}

/**
 * Scale a capsule's girth to the wearer, rather than leaving it at the stock
 * body's.
 *
 * THIS IS THE DIFFERENCE BETWEEN "a correctly fitted shirt" AND "a correctly
 * fitted shirt on the right person". Everything else in this file measures the
 * TWIN — Anny's neutral body, 77.5 cm around the chest. Posing that twin to the
 * wearer's joints puts it in the right place with the wrong dimensions, so the
 * garment is drafted for somebody else and then drawn over the shopper.
 *
 * Silhouette width from the segmentation mask is directly observable, unlike
 * anything the shape-fit solver tries to infer, so it needs no convergence and
 * cannot saturate. Its honest limitation: it measures the shopper's CLOTHING,
 * so a loose top reads wider than the body underneath. That still makes the
 * garment match the visible outline, which is what "it fits me" looks like —
 * it just means the resulting number is a clothed silhouette, not a body
 * measurement, and must not be presented as one.
 *
 * Depth is not observable from one frontal view, so the twin's own
 * width-to-depth ratio is carried over. Wrong for unusual builds, and far
 * closer than assuming the stock body.
 *
 * @param scaleByLabel e.g. { chest: 1.12, waist: 1.05, hips: 1.08 }
 */
export function applyGirthScale(caps, scaleByLabel, limits = [0.75, 1.6]) {
  for (const c of caps) {
    const s = scaleByLabel[c.label];
    if (!s || !isFinite(s)) continue;
    const k = THREE.MathUtils.clamp(s, limits[0], limits[1]);
    // Always from the AS-MEASURED radius, never from the current one, or
    // repeated calls compound into a balloon.
    c.rx = (c.rx0 ?? c.rx) * k;
    c.rz = (c.rz0 ?? c.rz) * k;
    c.girthScale = k;
  }
  return caps;
}

/**
 * Recompute every capsule's endpoints from the CURRENT pose.
 *
 * `frame` is the torso's live orthonormal basis {left, up, front} in root-local
 * space — the same three vectors orientTorso() derives from the shoulder and
 * hip landmarks. Two things need it:
 *
 *   - `dz`, the forward shift that centres a spine-anchored capsule inside the
 *     torso, must follow the body when it turns. Left as a world +z offset it
 *     would swing out of the chest the moment the shopper rotates.
 *   - `u`, the direction the capsule's rx semi-axis points, likewise.
 *
 * Deriving the frame from bone POSITIONS (rather than a bone's world
 * quaternion) keeps this safe under the mirror's negative scale.
 */
export function poseCapsules(caps, root, frame, dt) {
  // Parents too — see the note in measureCapsules. Here it matters every
  // frame, because alignGroup's scale changes as the shopper moves toward or
  // away from the camera.
  root.updateWorldMatrix(true, true);
  for (const c of caps) {
    const a = averagePosition(c.aBones, root);
    const b = averagePosition(c.bBones, root);
    if (c.dz) {
      const shift = frame.front.clone().multiplyScalar(c.dz);
      a.add(shift); b.add(shift);
    }
    // Velocity in metres per SECOND, not per call. Pose updates arrive at the
    // camera's rate (~30 Hz) while the solver runs at display rate, so a
    // per-call delta would be applied a variable number of times and friction
    // would depend on the frame rate.
    c.vel = c.a && dt > 0
      ? a.clone().sub(c.a).divideScalar(dt)
      : new THREE.Vector3();
    c.a = a; c.b = b;
    c.u = frame.left.clone();
  }
  return caps;
}

export function capsuleByLabel(caps) {
  return new Map(caps.map((c) => [c.label, c]));
}

// ---------------------------------------------------------------------------
// 2. Drafting the garment
// ---------------------------------------------------------------------------

/**
 * Generate a t-shirt around the posed capsules.
 *
 * A flat sheet cannot fit a body: it has no neck hole to catch on the
 * shoulders, no seams to wrap, and no shaping to follow the waist — it tents
 * over the top exactly as a bedsheet does. A garment stays on because it is
 * MECHANICALLY CAPTURED (the neck hole rests on the shoulders, the arms pass
 * through the sleeves) and it fits because its dimensions are drafted from the
 * body plus an ease allowance.
 *
 * Topology is a tube: columns wrap around the body, rows run neck to hem. The
 * first rows form a near-horizontal yoke spreading from the neck opening out
 * to the shoulders; the rest fall vertically, interpolating chest → waist →
 * hip radii. Two armholes are cut in it, and the sleeves are separate tubes
 * sewn to the armhole rims.
 */
export function draftShirt({ caps, res = 48, ease = 0.03, hemDrop = null,
                             neckCircum = 0.42, sleeveLength = 0.34,
                             lengthRatio = 1.37 }) {
  const m = capsuleByLabel(caps);
  const chest = m.get("chest"), waist = m.get("waist"), hips = m.get("hips");
  const neck = m.get("neck"), shL = m.get("shoulderL"), shR = m.get("shoulderR");
  const armL = m.get("armL"), armR = m.get("armR");
  if (!chest || !waist || !hips || !neck) throw new Error("missing torso capsules");

  // Local frame of the garment: the tube's columns sweep around `side` and
  // `front`, its rows run down `up`. Using the body's own axes rather than
  // world axes is what lets the shirt spawn correctly on a shopper standing at
  // an angle to the camera, or leaning.
  //
  // `up` comes from the CHEST CAPSULE's own axis (spine03 → spine01), not from
  // chest-minus-hip: the hip capsule's endpoint is one hip joint, off to the
  // side, so that difference would tilt the whole garment sideways.
  const up = new THREE.Vector3().subVectors(chest.b, chest.a).normalize();
  // Gram-Schmidt the side axis against it, so the three axes are orthonormal
  // even though the shoulder line and the spine are never quite perpendicular.
  const side = chest.u.clone().addScaledVector(up, -chest.u.dot(up)).normalize();
  const front = new THREE.Vector3().crossVectors(side, up).normalize();

  // Half the distance between the two shoulder joints: how far the yoke must
  // reach sideways before it turns and runs down the body.
  const halfShoulder = shL && shR
    ? Math.abs(shL.b.clone().sub(shR.b).dot(side)) / 2
    : chest.rx * 1.35;

  // hips.a is the midpoint of the two leg joints, so the hip capsule is
  // vertical like the others and its rx/rz mean the same thing they do on the
  // chest. Checked against the mesh: 0.146 half-width, 0.094 half-depth.
  const hipMid = hips.a.clone();
  const hipRx = hips.rx, hipRz = hips.rz;

  // HEM LENGTH from a garment spec, not a magic number.
  //
  // A men's medium tee measures ~71 cm high-point-shoulder to hem, on a body
  // whose shoulder-to-hip-joint is ~52 cm — a ratio of about 1.37. Applying
  // the ratio rather than the absolute length is what makes it grade with the
  // wearer instead of being right for one body only.
  const TEE_LENGTH_RATIO = lengthRatio;

  // The neck opening has to clear the neck's COLLISION radius, not its
  // measured radius. Cut to the raw measurement, the hole is smaller than the
  // thing it must pass over, and collision ejects the whole garment sideways
  // every frame. Anything the garment slips over is sized against the inflated
  // collision geometry.
  //
  // Sized against a real crew neck rather than a guess: a men's tee measures
  // 40-44 cm around the opening, so the radius wants to be ~0.066-0.070 m. The
  // previous +0.022 slack gave 47.8 cm, noticeably wider than any real tee,
  // which let the whole garment drop through. Clamped, not just computed, so a
  // thick-necked body still gets an opening it can physically wear.
  const NECK_CIRCUM_TARGET = neckCircum;
  const neckFloor = neck.rx + COLLISION_MARGIN + 0.004;   // must clear the neck
  const neckR = Math.max(neckFloor, NECK_CIRCUM_TARGET / (2 * Math.PI));

  // ---- the garment's centre line ----
  //
  // Each torso capsule was recentred forward by its own amount when it was
  // measured (the spine runs along the BACK of the body, so a spine-anchored
  // capsule has to move forward to sit inside the chest). Those amounts
  // differ: the chest's is about 6 cm, the hips' is zero. Drafting the whole
  // tube around a single axis would therefore leave the hem hanging several
  // centimetres in front of the hips.
  //
  // So the tube is generated along a POLYLINE through anchor points that are
  // real places on the body, each carrying the cross-section measured there.
  // This is close to how a pattern is actually drafted: a centre-front line
  // plus a girth at each level.
  const anchor = (p, rx, rz) => ({ p: p.clone(), rx, rz });

  // THE SHOULDER LINE. Measured, not assumed.
  //
  // This was the single biggest error in the garment: the yoke's widest ring
  // was anchored to spine01, on the belief that it marked the shoulders. It
  // does not — measured on the twin, spine01 sits 76 mm BELOW the shoulder
  // joints. So the garment's widest point landed at mid-chest and nothing at
  // all rested on the shoulders. A shirt held up by nothing slides down and
  // pools, which is exactly what "settles as a sack" looks like.
  //
  // Take the HEIGHT from the shoulder joints and everything else from the
  // torso's centre line, so the ring keeps the forward offset that puts it
  // inside the chest rather than behind it.
  const shoulderHeight = armL && armR
    ? (heightOfRaw(armL.a, up) + heightOfRaw(armR.a, up)) / 2
    : heightOfRaw(chest.b, up);
  const shoulderPoint = chest.b.clone()
    .addScaledVector(up, shoulderHeight - heightOfRaw(chest.b, up));

  const yokeChain = [
    anchor(neck.a, neckR, neckR),
    anchor(shoulderPoint, halfShoulder, chest.rz + ease),
  ];
  // Resolve the hem now that the shoulder line is known. hemDrop is how far
  // below the hip JOINTS the hem falls.
  const shoulderToHip = shoulderHeight - heightOfRaw(hipMid, up);
  const hemDropM = hemDrop ?? Math.max(0.04, shoulderToHip * (TEE_LENGTH_RATIO - 1));

  const bodyChain = [
    anchor(shoulderPoint, halfShoulder, chest.rz + ease),
    anchor(chest.a, chest.rx + ease, chest.rz + ease),
    anchor(waist.a, waist.rx + ease, waist.rz + ease),
    anchor(hipMid, hipRx + ease, hipRz + ease),
    anchor(hipMid.clone().addScaledVector(up, -hemDropM), hipRx + ease, hipRz + ease),
  ];

  // Sample a chain by arc length, so rows are evenly spaced down the fabric
  // rather than bunching wherever two anchors happen to be close together.
  const sampler = (chain) => {
    const seg = [];
    let total = 0;
    for (let i = 1; i < chain.length; i++) {
      const d = chain[i].p.distanceTo(chain[i - 1].p);
      seg.push(d); total += d;
    }
    return (s) => {
      let x = THREE.MathUtils.clamp(s, 0, 1) * total;
      for (let i = 0; i < seg.length; i++) {
        if (x <= seg[i] || i === seg.length - 1) {
          const t = seg[i] > 1e-9 ? THREE.MathUtils.clamp(x / seg[i], 0, 1) : 0;
          const A = chain[i], B = chain[i + 1];
          return {
            p: A.p.clone().lerp(B.p, t),
            rx: THREE.MathUtils.lerp(A.rx, B.rx, t),
            rz: THREE.MathUtils.lerp(A.rz, B.rz, t),
          };
        }
        x -= seg[i];
      }
    };
  };
  const yokeAt = sampler(yokeChain);
  const bodyAt = sampler(bodyChain);

  const NCOL = res;
  const NROW = Math.round(res * 0.85);
  // A broad yoke seats the shirt across both shoulders instead of balancing it
  // on a narrow cone.
  const YOKE = Math.max(4, Math.round(NROW * 0.22));
  const N = NCOL * NROW;
  const idx = (c, r) => r * NCOL + ((c % NCOL) + NCOL) % NCOL;   // columns wrap

  // ARMHOLES.
  //
  // A closed tube cannot coexist with arms held near the sides: the cylinder
  // gets generated straight through both upper arms and collision ejects the
  // garment. Real shirts solve this with a hole, so we cut one.
  //
  // WHERE to cut is the interesting decision. Cutting wherever the arm happens
  // to be right now would be wrong: a real armhole is a FIXED hole in the
  // fabric, positioned at the shoulder, and it stays there when you raise your
  // arm. So the hole is anchored to the SHOULDER JOINT — a sphere centred just
  // outboard of it — which is pose-independent. With the hole in place the arm
  // hangs outside the tube and presses against it, which is what a real sleeve
  // seam feels like, instead of being trapped inside it.
  // Anything that silently changes the result must end up visible, not in a
  // console.warn nobody reads. draftShirt returns this and the panel shows it.
  const degraded = [];
  const armholeCentres = [];
  for (const [sh, arm] of [[shL, armL], [shR, armR]]) {
    if (!sh || !arm) continue;
    // arm.a IS the shoulder joint. Nudge the hole outboard (horizontally, away
    // from the spine) so it straddles the seam rather than eating into the
    // chest panel.
    const outward = arm.a.clone().sub(chest.b);
    outward.addScaledVector(up, -outward.dot(up));
    if (outward.lengthSq() > 1e-8) outward.normalize().multiplyScalar(0.02);
    else outward.set(0, 0, 0);
    armholeCentres.push({
      p: arm.a.clone().add(outward),
      // Just enough for the arm plus its collision margin to pass through.
      // An earlier +0.030 made this 0.126 m — a 25 cm hole centred on the
      // shoulder, which removed the entire shoulder seam and left the shirt
      // hanging off nothing. The armhole should clear the arm, not the torso.
      arm,
      r: arm.rx + COLLISION_MARGIN + 0.012,
      // Nothing above the shoulder joint may be cut. On a real shirt the
      // shoulder seam runs OVER the top of the shoulder and the armhole opens
      // below it; without this ceiling a sphere centred at the joint eats
      // upward through the yoke, which is exactly what shredded it.
      ceiling: arm.a.dot(up),
    });
  }
  const heightOf = (v) => v.dot(up);

  const pos = new Float32Array(N * 4);
  const uv = new Float32Array(N * 2);
  const active = new Uint8Array(N).fill(1);
  const tmp = new THREE.Vector3();

  for (let r = 0; r < NROW; r++) {
    // Yoke: opens out from the neck hole to the shoulder line, dropping only
    // slightly. This is the part that catches on the shoulders, and it is the
    // only thing holding the shirt up before the sleeves take some of the load.
    // Body: shoulder → chest → waist → hip, so the shirt follows the torso
    // instead of hanging as a cylinder.
    const lvl = r < YOKE
      ? yokeAt(r / YOKE)
      : bodyAt((r - YOKE) / (NROW - 1 - YOKE));

    for (let c = 0; c < NCOL; c++) {
      const th = (c / NCOL) * Math.PI * 2;
      const i = idx(c, r);
      tmp.copy(lvl.p)
        .addScaledVector(side, lvl.rx * Math.cos(th))
        .addScaledVector(front, lvl.rz * Math.sin(th));
      pos[i * 4] = tmp.x; pos[i * 4 + 1] = tmp.y; pos[i * 4 + 2] = tmp.z;
      uv[i * 2] = c / NCOL; uv[i * 2 + 1] = 1 - r / NROW;

      const cut = armholeCentres.some(
        (h) => heightOf(tmp) <= h.ceiling && tmp.distanceTo(h.p) < h.r);
      // Inactive particles get w = 0 so predict/collide skip them outright,
      // and they are excluded from constraints and triangles below — inert
      // entries in the buffer. Keeping them in place (rather than compacting
      // the array) means idx() stays a plain grid lookup.
      active[i] = cut ? 0 : 1;
      pos[i * 4 + 3] = cut ? 0 : 1;   // nothing else is pinned; the neck holds it
    }
  }

  // Tidy the rim. Where the armhole boundary crosses the grid diagonally it
  // leaves single cells clinging on by one neighbour. Those render as the
  // spikes fringing the opening, and physically they are a scrap of fabric
  // hanging off a single thread — it flaps, and it contributes nothing.
  // Repeatedly drop any particle with fewer than two surviving neighbours
  // until the edge stops changing.
  for (let pass = 0; pass < 4; pass++) {
    let removed = 0;
    const before = active.slice();
    for (let r = 0; r < NROW; r++) {
      for (let c = 0; c < NCOL; c++) {
        const i = idx(c, r);
        if (!before[i]) continue;
        let n = 0;
        if (before[idx(c - 1, r)]) n++;
        if (before[idx(c + 1, r)]) n++;
        if (r > 0 && before[idx(c, r - 1)]) n++;
        if (r < NROW - 1 && before[idx(c, r + 1)]) n++;
        if (n < 2) { active[i] = 0; pos[i * 4 + 3] = 0; removed++; }
      }
    }
    if (!removed) break;
  }

  // ---- sleeves ----
  // Built as separate small tubes around each upper arm and STITCHED to the
  // armhole rim, rather than branching the main tube's topology. Physically
  // the same capture, far simpler to generate correctly.
  const SCOL = Math.max(10, Math.round(NCOL * 0.32));
  const SROW = 6;
  const sleeves = [];
  const sleevePos = [], sleeveUv = [];

  // Find the particles on the edge of each armhole: active, but with at least
  // one neighbour removed. That ring of points IS the armhole.
  const rimFor = (hole) => {
    const out = [];
    for (let r = 0; r < NROW; r++) {
      for (let c = 0; c < NCOL; c++) {
        const i = idx(c, r);
        if (!active[i]) continue;
        const edge = !active[idx(c - 1, r)] || !active[idx(c + 1, r)] ||
          (r > 0 && !active[idx(c, r - 1)]) ||
          (r < NROW - 1 && !active[idx(c, r + 1)]);
        if (!edge) continue;
        tmp.set(pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2]);
        // Only this hole's rim, not the other arm's or the neck's.
        if (tmp.distanceTo(hole.p) > hole.r * 2.0) continue;
        out.push(i);
      }
    }
    return out;
  };

  for (let k = 0; k < 2; k++) {
    const arm = k === 0 ? armL : armR;
    if (!arm) continue;
    const hole = armholeCentres.find((h) => h.arm === arm);
    const axis = arm.b.clone().sub(arm.a).normalize();
    const alt = Math.abs(axis.dot(up)) < 0.9 ? up : side;
    const u = new THREE.Vector3().crossVectors(axis, alt).normalize();
    const v2 = new THREE.Vector3().crossVectors(axis, u).normalize();
    const rad = arm.rx + COLLISION_MARGIN + ease;
    const len = sleeveLength;           // fraction of the upper arm covered

    // THE SLEEVE HEAD IS THE ARMHOLE.
    //
    // Previously the sleeve's top ring was generated around the ARM AXIS while
    // the armhole rim sits on the TUBE SURFACE — two different surfaces, up to
    // 157 mm apart on this body. The seam constraints were correctly created
    // and correctly inextensible, and they faithfully held the sleeve 157 mm
    // away from the panel. Rigid, and rigidly wrong.
    //
    // A real sleeve is cut so its head matches the armhole and is then sewn
    // edge to edge. So: take the rim, sort it by angle about the arm, resample
    // it to SCOL points, and make THAT the sleeve's first row. Every seam
    // stitch then has a rest length of ~0, which is what "sewn" means.
    const rim = hole ? rimFor(hole) : [];
    const ringPts = [];
    let rimOrdered = [];
    if (rim.length >= 6) {
      const withAngle = rim.map((i) => {
        const p = new THREE.Vector3(pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2]);
        const d = p.clone().sub(arm.a);
        return { i, p, th: Math.atan2(d.dot(v2), d.dot(u)) };
      }).sort((a, b) => a.th - b.th);
      rimOrdered = withAngle.map((w) => w.i);
      for (let c = 0; c < SCOL; c++) {
        // Walk the rim in angular order and pick the nearest sample, so the
        // sleeve's columns wrap the hole the same way round.
        const want = -Math.PI + (c / SCOL) * Math.PI * 2;
        let best = withAngle[0], bestD = Infinity;
        for (const w of withAngle) {
          const d = Math.abs(Math.atan2(Math.sin(w.th - want), Math.cos(w.th - want)));
          if (d < bestD) { bestD = d; best = w; }
        }
        ringPts.push({ p: best.p.clone(), body: best.i });
      }
    } else {
      // Degenerate: no armhole was cut (arms raised clear of the tube, say).
      // Fall back to a plain ring and flag it — silently producing a detached
      // sleeve is exactly the failure this whole change is about.
      degraded.push(`sleeve ${k === 0 ? "L" : "R"}: armhole rim had only ` +
                    `${rim.length} points, sleeve not sewn to the panel`);
      for (let c = 0; c < SCOL; c++) {
        const th = (c / SCOL) * Math.PI * 2;
        ringPts.push({
          p: arm.a.clone().lerp(arm.b, 0.02)
            .addScaledVector(u, rad * Math.cos(th))
            .addScaledVector(v2, rad * Math.sin(th)),
          body: -1,
        });
      }
    }

    const base = N + sleevePos.length / 4;
    for (let r = 0; r < SROW; r++) {
      const f = r / (SROW - 1);
      const t = 0.02 + f * len;
      const c0 = arm.a.clone().lerp(arm.b, t);
      for (let c = 0; c < SCOL; c++) {
        const th = (c / SCOL) * Math.PI * 2;
        // Row 0 sits exactly on the rim; lower rows relax toward a clean tube
        // around the arm, which is how a set-in sleeve actually transitions.
        const round = c0.clone()
          .addScaledVector(u, rad * Math.cos(th))
          .addScaledVector(v2, rad * Math.sin(th));
        const pt = ringPts[c].p.clone().lerp(round, Math.min(1, f * 1.6));
        sleevePos.push(pt.x, pt.y, pt.z, 1);
        sleeveUv.push(c / SCOL, 1 - r / SROW);
      }
    }
    // rimOrdered is every rim particle in angular order. The resampled `ring`
    // drives the seam CONSTRAINTS; the full rim drives the seam GEOMETRY,
    // because the rim usually has more vertices than the sleeve has columns
    // and stitching only the sampled ones leaves the rest of the hole open.
    sleeves.push({ base, SCOL, SROW, ring: ringPts.map((r) => r.body),
                   rim: rimOrdered });
  }

  const TOTAL = N + sleevePos.length / 4;
  const allPos = new Float32Array(TOTAL * 4);
  allPos.set(pos, 0);
  allPos.set(sleevePos, N * 4);
  const allUv = new Float32Array(TOTAL * 2);
  allUv.set(uv, 0);
  allUv.set(sleeveUv, N * 2);

  // ---- constraints ----
  const A = [], B = [], rest = [], kind = [];
  const seam = { count: 0, maxRest: 0, sumRest: 0 };
  // Sleeve particles live past index N and are always active; body particles
  // may have been removed to form an armhole. A constraint touching a removed
  // particle would tether fabric to a dead point, so drop it. One guard covers
  // the tube lattice, the bend constraints AND the sleeve seam.
  const isActive = (i) => i >= N || active[i] === 1;
  const add = (a, b, k) => {
    if (!isActive(a) || !isActive(b)) return;
    A.push(a); B.push(b); kind.push(k);
    rest.push(Math.hypot(
      allPos[a * 4] - allPos[b * 4],
      allPos[a * 4 + 1] - allPos[b * 4 + 1],
      allPos[a * 4 + 2] - allPos[b * 4 + 2]));
  };

  // kind 0 = stretch (near-inextensible), kind 1 = bend (compliant).
  // The last rows use STRETCH compliance for their bend constraints: a real
  // hem is a folded, doubled band, noticeably stiffer than the body of the
  // shirt, and that stiffness is what stops a free edge curling up on itself.
  const HEM_ROWS = 3;
  for (let r = 0; r < NROW; r++) {
    const stiffHem = r >= NROW - HEM_ROWS ? 0 : 1;
    for (let c = 0; c < NCOL; c++) {
      add(idx(c, r), idx(c + 1, r), 0);
      if (r + 1 < NROW) {
        add(idx(c, r), idx(c, r + 1), 0);
        add(idx(c, r), idx(c + 1, r + 1), 0);       // shear
        add(idx(c + 1, r), idx(c, r + 1), 0);       // shear
      }
      add(idx(c, r), idx(c + 2, r), stiffHem);
      if (r + 2 < NROW) add(idx(c, r), idx(c, r + 2), stiffHem);
    }
  }

  const sidx = (s, c, r) => s.base + r * s.SCOL + ((c % s.SCOL) + s.SCOL) % s.SCOL;
  for (const s of sleeves) {
    for (let r = 0; r < s.SROW; r++) {
      for (let c = 0; c < s.SCOL; c++) {
        add(sidx(s, c, r), sidx(s, c + 1, r), 0);
        if (r + 1 < s.SROW) {
          add(sidx(s, c, r), sidx(s, c, r + 1), 0);
          add(sidx(s, c, r), sidx(s, c + 1, r + 1), 0);
          add(sidx(s, c + 1, r), sidx(s, c, r + 1), 0);
        }
        add(sidx(s, c, r), sidx(s, c + 2, r), 1);
      }
    }

    // Sew row 0 to the rim particle it was built from. Because the sleeve head
    // was cut FROM the rim, each stitch has a rest length of about zero — the
    // two edges are coincident, which is what a sewn seam is.
    for (let c = 0; c < s.SCOL; c++) {
      const si = sidx(s, c, 0);
      const best = s.ring ? s.ring[c] : -1;
      const bestD = best >= 0 ? (
        (allPos[si * 4] - allPos[best * 4]) ** 2 +
        (allPos[si * 4 + 1] - allPos[best * 4 + 1]) ** 2 +
        (allPos[si * 4 + 2] - allPos[best * 4 + 2]) ** 2) : 0;
      // kind 0 == the stretch group, whose compliance is hard-wired to 0 in
      // GarmentSim.step — i.e. an inextensible seam, which is what a stitched
      // seam is. Recorded so the debug panel can prove the seam exists and how
      // long each stitch is: a stitch whose rest length is a large fraction of
      // the sleeve radius means the sleeve is tacked on at a distance rather
      // than sewn to the armhole rim, and it will read as "floating".
      if (best >= 0) {
        add(si, best, 0);
        seam.count++;
        const d = Math.sqrt(bestD);
        seam.maxRest = Math.max(seam.maxRest, d);
        seam.sumRest += d;
      }
    }
  }

  // ---- graph colouring ----
  // Two constraints that share a particle must not be solved by concurrent
  // threads, or they race on the same memory. Colouring partitions them into
  // sets where no two members touch the same particle; each set is then one
  // fully parallel dispatch.
  const groups = [];
  for (const wantKind of [0, 1]) {
    const map = [];
    for (let k = 0; k < A.length; k++) if (kind[k] === wantKind) map.push(k);
    const sa = map.map((k) => A[k]);
    const sb = map.map((k) => B[k]);
    const { colour, numColours } = greedyColour(sa, sb, TOTAL);
    const order = map.map((_, i) => i).sort((x, y) => colour[x] - colour[y]);
    const ranges = [];
    let start = 0;
    for (let c = 0; c < numColours; c++) {
      const count = order.filter((i) => colour[i] === c).length;
      ranges.push({ start, count });
      start += count;
    }
    groups.push({
      kind: wantKind,
      a: new Uint32Array(order.map((i) => sa[i])),
      b: new Uint32Array(order.map((i) => sb[i])),
      rest: new Float32Array(order.map((i) => rest[map[i]])),
      ranges,
    });
  }

  const cut = N - active.reduce((s, v) => s + v, 0);
  console.log(`garment: ${TOTAL} particles (${cut} cut for armholes), ` +
              `${A.length} constraints, ${groups[0].ranges.length}+` +
              `${groups[1].ranges.length} colours`);

  // Everything the debug panel needs to show measured-body -> drafted-garment
  // without recomputing any of it.
  const profile = {
    neckR, neckCircum: 2 * Math.PI * neckR,
    halfShoulder,
    chestRx: chest.rx, waistRx: waist.rx, hipRx,
    ease,
    shoulderRingHeight: heightOf(yokeChain[1].p),
    shoulderJointHeight: armL ? heightOf(armL.a) : null,
    neckHeight: heightOf(neck.a),
    hemHeight: heightOf(bodyChain[bodyChain.length - 1].p),
    garmentLength: heightOf(yokeChain[0].p) - heightOf(bodyChain[bodyChain.length - 1].p),
    shoulderToHip,
    shoulderToHem: shoulderHeight - heightOf(bodyChain[bodyChain.length - 1].p),
    lengthRatio: (shoulderHeight - heightOf(bodyChain[bodyChain.length - 1].p)) / shoulderToHip,
    targetLengthRatio: TEE_LENGTH_RATIO,
    hemDrop: hemDropM,
    // Real-world dimensions of the UV square, so artwork can be placed in
    // centimetres rather than in fractions of a texture.
    circumCm: ellipsePerimeter(chest.rx + ease, chest.rz + ease) * 100,
    lengthCm: (heightOf(yokeChain[0].p) - heightOf(bodyChain[bodyChain.length - 1].p)) * 100,
    degraded,
    seam: { count: seam.count, maxRest: seam.maxRest,
            avgRest: seam.count ? seam.sumRest / seam.count : 0 },
    constraints: { total: A.length,
                   stretch: kind.filter((k) => k === 0).length,
                   bend: kind.filter((k) => k === 1).length },
  };
  return { N: TOTAL, pos: allPos, uv: allUv, active, groups, idx,
           NCOL, NROW, bodyCount: N, sleeves, profile };
}

function greedyColour(cA, cB, numParticles) {
  const n = cA.length;
  const head = new Int32Array(numParticles).fill(-1);
  const next = new Int32Array(n * 2).fill(-1);
  const owner = new Int32Array(n * 2);
  let e = 0;
  for (let k = 0; k < n; k++) {
    for (const p of [cA[k], cB[k]]) { owner[e] = k; next[e] = head[p]; head[p] = e; e++; }
  }
  const colour = new Int32Array(n).fill(-1);
  const used = new Uint8Array(64);
  let maxColour = 0;
  for (let k = 0; k < n; k++) {
    used.fill(0);
    for (const p of [cA[k], cB[k]]) {
      for (let it = head[p]; it !== -1; it = next[it]) {
        const o = owner[it];
        if (o !== k && colour[o] >= 0) used[colour[o]] = 1;
      }
    }
    let c = 0;
    while (c < used.length && used[c]) c++;
    colour[k] = c;
    if (c > maxColour) maxColour = c;
  }
  return { colour, numColours: maxColour + 1 };
}

// ---------------------------------------------------------------------------
// 3. The solver
// ---------------------------------------------------------------------------

const SHADER = `
struct Params  { dt : f32, compliance : f32, colourOffset : u32, colourCount : u32, };
// capScale converts a capsule's metres-per-second velocity into the distance
// it travels during ONE substep — i.e. it is just the substep's dt, named
// separately so the friction term reads as a deliberate choice.
struct Globals { dt : f32, gravity : f32, friction : f32, capScale : f32,
                 numParticles : u32, numCapsules : u32, };
// a.xyz/b.xyz = segment ends, a.w = rx (across the body), b.w = rz (front to
// back), u.xyz = the direction rx points in, v.xyz = this frame's movement.
struct Capsule { a : vec4<f32>, b : vec4<f32>, u : vec4<f32>, v : vec4<f32>, };

@group(0) @binding(0) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> prevPos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cA : array<u32>;
@group(0) @binding(4) var<storage, read> cB : array<u32>;
@group(0) @binding(5) var<storage, read> cRest : array<f32>;
@group(0) @binding(6) var<storage, read_write> lambda : array<f32>;
@group(0) @binding(7) var<uniform> g : Globals;
@group(0) @binding(8) var<storage, read> caps : array<Capsule>;
@group(1) @binding(0) var<uniform> p : Params;

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= g.numParticles) { return; }
  let w = pos[i].w;
  if (w == 0.0) { return; }
  var v = vel[i].xyz;
  v.y = v.y + g.gravity * g.dt;
  v = v * 0.998;
  prevPos[i] = pos[i];
  pos[i] = vec4<f32>(pos[i].xyz + v * g.dt, w);
  vel[i] = vec4<f32>(v, 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= p.colourCount) { return; }
  let k = gid.x + p.colourOffset;
  let a = cA[k]; let b = cB[k];
  let pa = pos[a]; let pb = pos[b];
  let wa = pa.w; let wb = pb.w;
  let wsum = wa + wb;
  if (wsum == 0.0) { return; }
  var d = pa.xyz - pb.xyz;
  let len = length(d);
  if (len < 1e-9) { return; }
  let C = len - cRest[k];
  // XPBD: compliance alpha = 1/stiffness, scaled by dt^2 so behaviour is
  // independent of timestep. lambda accumulates within the substep.
  let alpha = p.compliance / (p.dt * p.dt);
  let dl = (-C - alpha * lambda[k]) / (wsum + alpha);
  lambda[k] = lambda[k] + dl;
  let corr = d * (dl / len);
  if (wa > 0.0) { pos[a] = vec4<f32>(pa.xyz + corr * wa, wa); }
  if (wb > 0.0) { pos[b] = vec4<f32>(pb.xyz - corr * wb, wb); }
}

struct Hit { hit : bool, p : vec3<f32>, n : vec3<f32>, };

// Point vs oriented elliptical capsule.
//
// The ellipse is evaluated in the capsule's OWN frame (side / along-axis /
// front) rather than in world axes, so it keeps pointing the right way when
// the shopper turns. Inside that frame we use the scaled-space trick: divide
// by the radii so the ellipse becomes a unit sphere, test there, then scale
// the push-out back.
fn capsuleHit(cap : Capsule, p3 : vec3<f32>) -> Hit {
  var h : Hit;
  h.hit = false; h.p = p3; h.n = vec3<f32>(0.0, 1.0, 0.0);

  let a = cap.a.xyz;
  let ab = cap.b.xyz - a;
  let denom = max(dot(ab, ab), 1e-9);
  // clamp keeps the closest point on the SEGMENT, which is what turns an
  // infinite cylinder into a capsule with rounded ends
  let t = clamp(dot(p3 - a, ab) / denom, 0.0, 1.0);
  let closest = a + ab * t;
  let d = p3 - closest;
  let axis = ab * inverseSqrt(denom);

  // Gram-Schmidt: the supplied side direction is not exactly perpendicular to
  // the bone, and for a capsule that runs sideways (the hips) it may be nearly
  // parallel to it. Fall back to any perpendicular in that case — such
  // capsules are round, so the choice cannot matter.
  var uu = cap.u.xyz - axis * dot(cap.u.xyz, axis);
  if (dot(uu, uu) < 1e-8) {
    let alt = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
    uu = alt - axis * dot(alt, axis);
  }
  uu = normalize(uu);
  let ww = cross(axis, uu);

  let rx = cap.a.w;
  let rz = cap.b.w;
  let loc = vec3<f32>(dot(d, uu), dot(d, axis), dot(d, ww));
  let e = vec3<f32>(loc.x / rx, loc.y / rx, loc.z / rz);
  let el = length(e);
  if (el >= 1.0) { return h; }

  // Degenerate case: a particle exactly on the axis has no push-out direction,
  // so pick an arbitrary one rather than dividing by zero.
  var un = e / max(el, 1e-9);
  if (el < 1e-6) { un = vec3<f32>(0.0, 0.0, 1.0); }

  let surf = vec3<f32>(un.x * rx, un.y * rx, un.z * rz);
  h.p = closest + uu * surf.x + axis * surf.y + ww * surf.z;

  // Surface NORMAL, which on an ellipse is NOT the radial direction. For
  // (X/rx)^2 + (Y/rx)^2 + (Z/rz)^2 = 1 the gradient is (X/rx^2, Y/rx^2,
  // Z/rz^2), so with X = un.x*rx it reduces to (un.x/rx, un.y/rx, un.z/rz).
  // Using the radial offset instead would misclassify part of the sliding
  // motion as normal motion and let the cloth creep.
  let nl = normalize(vec3<f32>(un.x / rx, un.y / rx, un.z / rz) + vec3<f32>(1e-9, 0.0, 0.0));
  h.n = uu * nl.x + axis * nl.y + ww * nl.z;
  h.hit = true;
  return h;
}

// Collision as a projection, run every iteration alongside the distance
// constraints. Pushing particles out once at the end would let the next
// constraint pass drag the cloth straight back inside. One thread per
// PARTICLE, so each particle is written by exactly one thread and no colouring
// is needed here.
@compute @workgroup_size(64)
fn collide(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= g.numParticles) { return; }
  let w = pos[i].w;
  if (w == 0.0) { return; }
  var p3 = pos[i].xyz;

  for (var c : u32 = 0u; c < g.numCapsules; c = c + 1u) {
    let h = capsuleHit(caps[c], p3);
    if (h.hit) {
      p3 = h.p;
      // POSITION-based friction, applied at the moment of contact. Damping
      // velocity afterwards is too weak: by then the particle has already
      // slid. Instead undo part of the tangential displacement it made since
      // the last substep.
      //
      // The capsule's own movement is subtracted first, so what is resisted is
      // sliding ACROSS THE SKIN rather than movement through the room. Without
      // that term the shirt is left standing where it was every time the
      // shopper steps sideways.
      let dp = (p3 - prevPos[i].xyz) - caps[c].v.xyz * g.capScale;
      let dpt = dp - dot(dp, h.n) * h.n;
      p3 = p3 - dpt * g.friction;
    }
  }
  pos[i] = vec4<f32>(p3, w);
}

// No separate velocity friction pass. In PBD the velocity IS derived from the
// position change, so the friction already applied in collide() is carried
// through here automatically — and skipping it saves a second sweep over every
// capsule per substep, which matters once pose inference shares the GPU.
@compute @workgroup_size(64)
fn updateVel(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= g.numParticles) { return; }
  if (pos[i].w == 0.0) { return; }
  vel[i] = vec4<f32>((pos[i].xyz - prevPos[i].xyz) / g.dt, 0.0);
}

@compute @workgroup_size(64)
fn clearLambda(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= arrayLength(&lambda)) { return; }
  lambda[gid.x] = 0.0;
}
`;

export async function createDevice() {
  if (!navigator.gpu) throw new Error("WebGPU unavailable — see stage6/webgpu-probe.html");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no GPU adapter");
  return adapter.requestDevice();
}

export class GarmentSim {
  constructor(device, { substeps = 6 } = {}) {
    this.device = device;
    this.substeps = substeps;
    this.iterations = 6;
    this.bendCompliance = 4e-6;
    this.friction = 0.85;
    this.gravity = -9.81;
    this.gpu = null;
    this.sim = null;
    this.numCapsules = 0;
  }

  destroy() {
    if (!this.gpu) return;
    for (const b of Object.values(this.gpu.buffers)) {
      try { b.destroy(); } catch { /* already gone */ }
    }
    this.gpu = null;
  }

  /** Upload a freshly drafted garment and allocate every GPU buffer for it. */
  spawn(sim, numCapsules) {
    this.destroy();
    this.sim = sim;
    this.numCapsules = numCapsules;
    const device = this.device;
    const { N, pos, groups } = sim;
    const S = GPUBufferUsage.STORAGE;
    const mk = (bytes, usage) => device.createBuffer({ size: Math.max(bytes, 16), usage });

    const posBuf = mk(pos.byteLength, S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    device.queue.writeBuffer(posBuf, 0, pos);
    const prevBuf = mk(pos.byteLength, S | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(prevBuf, 0, pos);
    const velBuf = mk(pos.byteLength, S | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(velBuf, 0, new Float32Array(N * 4));

    const totalC = groups.reduce((s, g) => s + g.a.length, 0);
    const allA = new Uint32Array(totalC), allB = new Uint32Array(totalC);
    const allR = new Float32Array(totalC);
    let off = 0;
    for (const g of groups) {
      allA.set(g.a, off); allB.set(g.b, off); allR.set(g.rest, off);
      g.base = off; off += g.a.length;
    }
    const aBuf = mk(allA.byteLength, S | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(aBuf, 0, allA);
    const bBuf = mk(allB.byteLength, S | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(bBuf, 0, allB);
    const rBuf = mk(allR.byteLength, S | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(rBuf, 0, allR);
    const lamBuf = mk(totalC * 4, S | GPUBufferUsage.COPY_DST);

    const capBuf = mk(Math.max(numCapsules * 64, 64), S | GPUBufferUsage.COPY_DST);
    const globalsBuf = device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const slots = groups.reduce((s, g) => s + g.ranges.length, 0);
    const paramsBuf = device.createBuffer({
      size: slots * UNIFORM_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readBuf = device.createBuffer({
      size: pos.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const module = device.createShaderModule({ code: SHADER });
    const bgl0 = device.createBindGroupLayout({
      entries: [
        ...[0, 1, 2].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" } })),
        ...[3, 4, 5].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } })),
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
      ],
    });
    // hasDynamicOffset lets one buffer serve every colour: WebGPU has no push
    // constants, so per-dispatch parameters are strided slots in a uniform
    // buffer selected by offset at bind time.
    const bgl1 = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 } }],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] });
    const pipe = (entryPoint) => device.createComputePipeline({
      layout, compute: { module, entryPoint } });

    const bind0 = device.createBindGroup({
      layout: bgl0,
      entries: [
        { binding: 0, resource: { buffer: posBuf } },
        { binding: 1, resource: { buffer: prevBuf } },
        { binding: 2, resource: { buffer: velBuf } },
        { binding: 3, resource: { buffer: aBuf } },
        { binding: 4, resource: { buffer: bBuf } },
        { binding: 5, resource: { buffer: rBuf } },
        { binding: 6, resource: { buffer: lamBuf } },
        { binding: 7, resource: { buffer: globalsBuf } },
        { binding: 8, resource: { buffer: capBuf } },
      ],
    });
    const bind1 = device.createBindGroup({
      layout: bgl1, entries: [{ binding: 0, resource: { buffer: paramsBuf, size: 16 } }] });

    this.gpu = {
      buffers: { posBuf, prevBuf, velBuf, aBuf, bBuf, rBuf, lamBuf, capBuf,
                 globalsBuf, paramsBuf, readBuf },
      posBuf, readBuf, capBuf, globalsBuf, paramsBuf,
      bind0, bind1, totalC,
      pipelines: {
        predict: pipe("predict"), solve: pipe("solve"), collide: pipe("collide"),
        updateVel: pipe("updateVel"), clearLambda: pipe("clearLambda"),
      },
      paramsData: new ArrayBuffer(slots * UNIFORM_STRIDE),
      capData: new Float32Array(Math.max(numCapsules * 16, 16)),
    };
    return this;
  }

  /** Push the current frame's capsule positions to the GPU. */
  setCapsules(caps) {
    if (!this.gpu) return;
    const d = this.gpu.capData;
    caps.forEach((c, i) => {
      const o = i * 16;
      // Only the COLLISION surface is inflated by the margin — the garment is
      // still cut to the measured radii, so `ease` keeps meaning what it says.
      d[o]      = c.a.x; d[o + 1]  = c.a.y; d[o + 2]  = c.a.z;
      d[o + 3]  = c.rx + COLLISION_MARGIN;
      d[o + 4]  = c.b.x; d[o + 5]  = c.b.y; d[o + 6]  = c.b.z;
      d[o + 7]  = c.rz + COLLISION_MARGIN;
      d[o + 8]  = c.u.x; d[o + 9]  = c.u.y; d[o + 10] = c.u.z; d[o + 11] = 0;
      const v = c.vel ?? { x: 0, y: 0, z: 0 };
      d[o + 12] = v.x; d[o + 13] = v.y; d[o + 14] = v.z; d[o + 15] = 0;
    });
    this.device.queue.writeBuffer(this.gpu.capBuf, 0, d, 0, caps.length * 16);
  }

  /** Advance one rendered frame. Returns true if a readback was queued. */
  step(dtFrame, wantReadback) {
    if (!this.gpu) return false;
    const { device, gpu, sim } = this;
    const dt = dtFrame / this.substeps;

    const view = new DataView(gpu.paramsData);
    let slot = 0;
    for (const g of sim.groups) {
      const compliance = g.kind === 0 ? 0 : this.bendCompliance;
      for (const r of g.ranges) {
        const o = slot * UNIFORM_STRIDE;
        view.setFloat32(o, dt, true);
        view.setFloat32(o + 4, compliance, true);
        view.setUint32(o + 8, g.base + r.start, true);
        view.setUint32(o + 12, r.count, true);
        r.slot = slot; slot++;
      }
    }
    device.queue.writeBuffer(gpu.paramsBuf, 0, gpu.paramsData);

    const gb = new ArrayBuffer(32); const gv = new DataView(gb);
    gv.setFloat32(0, dt, true);
    gv.setFloat32(4, this.gravity, true);
    gv.setFloat32(8, this.friction, true);
    gv.setFloat32(12, dt, true);                  // capScale = one substep
    gv.setUint32(16, sim.N, true);
    gv.setUint32(20, this.numCapsules, true);
    device.queue.writeBuffer(gpu.globalsBuf, 0, gb);

    const enc = device.createCommandEncoder();
    const pGroups = Math.ceil(sim.N / WORKGROUP);
    const lGroups = Math.ceil(gpu.totalC / WORKGROUP);

    for (let s = 0; s < this.substeps; s++) {
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, gpu.bind0);
      pass.setBindGroup(1, gpu.bind1, [0]);

      pass.setPipeline(gpu.pipelines.clearLambda);
      pass.dispatchWorkgroups(lGroups);
      pass.setPipeline(gpu.pipelines.predict);
      pass.dispatchWorkgroups(pGroups);

      for (let it = 0; it < this.iterations; it++) {
        pass.setPipeline(gpu.pipelines.solve);
        for (const g of sim.groups) {
          for (const r of g.ranges) {
            if (!r.count) continue;
            pass.setBindGroup(1, gpu.bind1, [r.slot * UNIFORM_STRIDE]);
            pass.dispatchWorkgroups(Math.ceil(r.count / WORKGROUP));
          }
        }
        pass.setPipeline(gpu.pipelines.collide);
        pass.setBindGroup(1, gpu.bind1, [0]);
        pass.dispatchWorkgroups(pGroups);
      }

      pass.setPipeline(gpu.pipelines.updateVel);
      pass.dispatchWorkgroups(pGroups);
      pass.end();
    }

    // mapState is the AUTHORITATIVE answer to "is this buffer busy". A
    // hand-rolled boolean drifts out of sync with the driver and produces
    // "buffer used in submit while mapped".
    const doRead = wantReadback && gpu.readBuf.mapState === "unmapped";
    if (doRead) enc.copyBufferToBuffer(gpu.posBuf, 0, gpu.readBuf, 0, sim.pos.byteLength);
    device.queue.submit([enc.finish()]);
    return doRead;
  }

  /** Copy solved positions into a BufferGeometry, asynchronously. */
  readbackInto(geometry, target) {
    const buf = this.gpu.readBuf;
    if (buf.mapState !== "unmapped") return;
    const count = this.sim.N;
    buf.mapAsync(GPUMapMode.READ).then(() => {
      const src = new Float32Array(buf.getMappedRange());
      for (let i = 0; i < count; i++) {
        target[i * 3]     = src[i * 4];
        target[i * 3 + 1] = src[i * 4 + 1];
        target[i * 3 + 2] = src[i * 4 + 2];
      }
      buf.unmap();
      geometry.attributes.position.needsUpdate = true;
      geometry.computeVertexNormals();
    }).catch(() => { /* device lost or buffer destroyed mid-flight */ });
  }
}

/**
 * Fabric material for a garment.
 *
 * Three things, all of which the previous MeshStandardMaterial lacked:
 *
 *  - OPAQUE and double-sided. Cloth is not glass. Double-sided is not
 *    optional either: the mirror's negative x scale inverts triangle winding,
 *    so single-sided fabric is backface-culled into invisibility.
 *  - SHEEN. Real cloth has a soft retroreflective lobe from the fibre nap —
 *    it is the main thing that separates fabric from painted plastic, and
 *    MeshPhysicalMaterial models it directly.
 *  - A THICKNESS OFFSET. The simulation treats the garment as an infinitely
 *    thin surface, so wherever it rests exactly on the collision surface the
 *    body co-planar with it wins pixels at random and shows through as
 *    mottling. Pushing the rendered surface a millimetre or two out along its
 *    normal gives the fabric apparent thickness and removes the fight. It is
 *    done in the vertex shader, so it costs nothing and never feeds back into
 *    the physics.
 */
// ---------------------------------------------------------------------------
// Fabric appearance
// ---------------------------------------------------------------------------
//
// A correct simulation still reads as a bag if the surface is one flat colour.
// Three things do most of the work of making cloth look like cloth, and none
// of them need an authored asset:
//
//   KNIT STRUCTURE  jersey is a grid of interlocking loops. At normal viewing
//                   distance you don't resolve a loop, but you do see the
//                   anisotropic micro-shading it produces — slightly stronger
//                   vertically than horizontally. A normal map gets that.
//   TONAL VARIATION dyed cotton is never uniform. A little low-frequency noise
//                   in the albedo removes the "painted plastic" read instantly.
//   SHEEN           fabric has a retroreflective lobe from the fibre nap that
//                   metals and plastics do not. MeshPhysicalMaterial models it.

function makeCanvas(size) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  return c;
}

// Jersey knit as a normal map. Columns of interlocking loops: a V shape
// repeated, with the vertical rib stronger than the horizontal course, which
// is what gives knit its directional sheen.
function knitNormalMap(size = 512, loops = 64) {
  const c = makeCanvas(size);
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(size, size);
  const step = size / loops;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (x % step) / step - 0.5;      // across a loop
      const v = (y % step) / step - 0.5;      // along a course
      // A V-shaped loop: the surface tilts left on one half, right on the
      // other, and dips between courses.
      const nx = Math.sin(u * Math.PI * 2) * 0.55;
      const ny = Math.sin(v * Math.PI * 2) * 0.30 + Math.cos(u * Math.PI * 4) * 0.12;
      const nz = Math.sqrt(Math.max(1e-4, 1 - nx * nx - ny * ny));
      const i = (y * size + x) * 4;
      img.data[i] = (nx * 0.5 + 0.5) * 255;
      img.data[i + 1] = (ny * 0.5 + 0.5) * 255;
      img.data[i + 2] = nz * 255;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

// Base colour, tonal variation, and — if supplied — a chest print composited
// straight into the albedo.
//
// Compositing rather than a second texture and a shader mix: the print has to
// land in one specific place on the UV layout, the layout is a plain
// column/row grid, and a canvas draw expresses that in two lines. The cost is
// rebuilding the canvas when the colour changes, which is once per click.
// Perimeter of an ellipse has no closed form; Ramanujan's second approximation
// is accurate to about 1e-5 for the eccentricities a torso reaches.
export function ellipsePerimeter(a, b) {
  const h = ((a - b) ** 2) / ((a + b) ** 2);
  return Math.PI * (a + b) * (1 + (3 * h) / (10 + Math.sqrt(4 - 3 * h)));
}

function albedoMap(colour, print, geom = null, size = 2048) {
  const c = makeCanvas(size);
  const ctx = c.getContext("2d");
  const col = new THREE.Color(colour);
  ctx.fillStyle = `#${col.getHexString()}`;
  ctx.fillRect(0, 0, size, size);

  // Low-frequency mottling: dye is never perfectly even.
  ctx.globalAlpha = 0.055;
  for (let i = 0; i < 900; i++) {
    const r = 12 + Math.random() * 70;
    ctx.fillStyle = Math.random() < 0.5 ? "#000" : "#fff";
    ctx.beginPath();
    ctx.arc(Math.random() * size, Math.random() * size, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // RIBBING at the collar and hem. On a real tee these are a separate, denser,
  // doubled-over knit, and they read as darker with a fine vertical rib. v runs
  // 1 at the neck to 0 at the hem, so the collar is the top sliver and the hem
  // the bottom one. Getting these two bands right does more for "this is a
  // t-shirt" than anything else on the surface — their absence is most of why
  // the garment looked like a bag.
  const band = (v0, v1) => {
    const y0 = size * (1 - v1), y1 = size * (1 - v0);
    ctx.save();
    ctx.globalCompositeOperation = "multiply";
    ctx.fillStyle = "rgba(0,0,0,0.16)";
    ctx.fillRect(0, y0, size, y1 - y0);
    // fine vertical rib
    ctx.globalAlpha = 0.30;
    ctx.fillStyle = "#000";
    for (let x = 0; x < size; x += 6) ctx.fillRect(x, y0, 2, y1 - y0);
    ctx.restore();
  };
  band(0.955, 1.0);     // collar
  band(0.0, 0.035);     // hem band

  // Shadow in the seam line where the sleeve meets the body, and down the
  // sides. Ambient occlusion in a crease is a strong depth cue and the solver
  // gives us none of it.
  ctx.save();
  ctx.globalCompositeOperation = "multiply";
  const g = ctx.createLinearGradient(0, 0, size, 0);
  for (const u of [0.0, 0.5, 1.0]) {          // the two side seams
    g.addColorStop(Math.max(0, u - 0.02), "rgba(255,255,255,1)");
    g.addColorStop(Math.min(1, u), "rgba(180,180,180,1)");
    g.addColorStop(Math.min(1, u + 0.02), "rgba(255,255,255,1)");
  }
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  ctx.restore();

  if (print) {
    // PLACEMENT AND SIZE IN CENTIMETRES, not in fractions of the atlas.
    //
    // The UV grid is NOT isotropic. u spans the garment's circumference and v
    // its neck-to-hem length, and on this body those are 96.1 cm and 68.1 cm —
    // so a 2048 atlas is 21.3 px/cm across but 30.1 px/cm down. Sizing a print
    // as "20% of the atlas" therefore stretched it horizontally by 1.4x, and a
    // circular logo came out an ellipse.
    //
    // Converting through real dimensions fixes the aspect AND means artwork is
    // specified the way a brand actually specifies it: "print is 25 cm wide,
    // centred, 20 cm below the neck".
    const circumCm = (geom?.circumCm ?? 96.1);
    const lengthCm = (geom?.lengthCm ?? 68.1);
    const wCm = print.widthCm ?? 25;
    const hCm = wCm * (print.height / print.width || 1);

    const w = size * (wCm / circumCm);
    const h = size * (hCm / lengthCm);
    // u = 0.25 is the front of the chest: u wraps from the wearer's LEFT, so a
    // quarter turn lands centre-front.
    const cx = size * (print.u ?? 0.25);
    const cy = size * (1 - (print.v ?? 0.62));
    ctx.drawImage(print, cx - w / 2, cy - h / 2, w, h);
  }

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Fabric material for a garment.
 *
 *  - OPAQUE and double-sided. Cloth is not glass, and the mirror's negative x
 *    scale inverts winding, so single-sided fabric is culled into invisibility.
 *  - A THICKNESS OFFSET in the vertex shader. The simulation treats the garment
 *    as an infinitely thin surface, so where it rests exactly on the collision
 *    surface the body wins pixels at random and mottles through. Displacing the
 *    rendered surface along its normal gives apparent thickness, costs nothing,
 *    and never feeds back into the physics.
 */
export function makeGarmentMaterial(colour = 0xc94f4f, opts = {}) {
  const { thickness = 0.004, print = null, knitScale = 5, geom = null } = opts;
  const normalMap = knitNormalMap();
  normalMap.repeat.set(knitScale, knitScale * 1.4);   // courses denser than wales

  const mat = new THREE.MeshPhysicalMaterial({
    map: albedoMap(colour, print, opts.geom),
    normalMap,
    normalScale: new THREE.Vector2(0.35, 0.35),
    roughness: 0.92,
    metalness: 0.0,
    sheen: 0.7,
    sheenRoughness: 0.65,
    sheenColor: new THREE.Color(0xffffff),
    side: THREE.DoubleSide,
    transparent: false,
  });

  mat.userData.thickness = { value: thickness };
  mat.userData.rebuild = (newColour, newPrint) => {
    mat.map?.dispose();
    mat.map = albedoMap(newColour ?? colour, newPrint ?? print, geom);
    mat.needsUpdate = true;
  };
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uThickness = mat.userData.thickness;
    // DoubleSide lights the inside of the garment exactly like the outside,
    // which is why it reads as paper rather than cloth — you see through a neck
    // opening or a hem and the interior is as bright as the chest. Real fabric
    // interiors are shadowed and slightly desaturated.
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <color_fragment>",
      `#include <color_fragment>
       if (!gl_FrontFacing) { diffuseColor.rgb *= 0.55; }`);
    // Backtick templates, not "..." with escapes: the newline matters to GLSL
    // and an escaped \n inside a quoted string is exactly what broke this file
    // once already.
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>",
               `#include <common>
                uniform float uThickness;`)
      .replace("#include <begin_vertex>",
               `#include <begin_vertex>
                transformed += normalize(objectNormal) * uThickness;`);
  };
  return mat;
}

/**
 * Load artwork for use as a garment print.
 *
 * `opts` carries the placement a brand would specify on a spec sheet:
 *   widthCm  printed width across the chest (default 25, a common chest print)
 *   u        0.25 is centre-front; u wraps from the wearer's left
 *   v        1 at the neck, 0 at the hem; 0.62 sits high on the chest
 *
 * Resolves to null when the file is absent — a plain shirt is a valid state,
 * not an error.
 */
export function loadPrint(url, opts = {}) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      img.widthCm = opts.widthCm ?? 25;
      img.u = opts.u ?? 0.25;
      img.v = opts.v ?? 0.62;
      if (img.naturalWidth < 512) {
        // At 2048 atlas the garment resolves ~21 px/cm across, so a 25 cm print
        // wants ~533 px. Below that it will visibly soften.
        console.warn(`garment print is ${img.naturalWidth}px wide; ` +
                     `at least 512px recommended for a ${img.widthCm} cm print`);
      }
      resolve(img);
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}


// Ordered boundary loops of a triangle soup.
//
// A directed edge a->b is on the boundary when b->a does not also exist: an
// interior edge is always traversed once in each direction by the two triangles
// sharing it. Following those directed edges yields loops in TRAVERSAL ORDER,
// which is the part that matters — an earlier attempt sorted rim vertices by
// angle instead, and because the rim is a jagged staircase rather than a circle
// the sorted order was not the adjacency order, so stitching it produced a
// zigzag full of small holes.
function boundaryLoops(tri) {
  const dir = new Set();
  for (let i = 0; i < tri.length; i += 3) {
    const [a, b, c] = [tri[i], tri[i + 1], tri[i + 2]];
    dir.add(`${a}_${b}`); dir.add(`${b}_${c}`); dir.add(`${c}_${a}`);
  }
  const next = new Map();
  for (const e of dir) {
    const [a, b] = e.split("_").map(Number);
    if (!dir.has(`${b}_${a}`) && !next.has(a)) next.set(a, b);
  }
  const seen = new Set(), loops = [];
  for (const start of next.keys()) {
    if (seen.has(start)) continue;
    const loop = []; let v = start;
    while (v !== undefined && !seen.has(v)) { seen.add(v); loop.push(v); v = next.get(v); }
    if (loop.length >= 3) loops.push(loop);
  }
  return loops;
}

/** Build the renderable mesh for a drafted garment. */
export function buildGarmentGeometry(sim) {
  const geometry = new THREE.BufferGeometry();
  const renderPos = new Float32Array(sim.N * 3);
  for (let i = 0; i < sim.N; i++) {
    renderPos[i * 3] = sim.pos[i * 4];
    renderPos[i * 3 + 1] = sim.pos[i * 4 + 1];
    renderPos[i * 3 + 2] = sim.pos[i * 4 + 2];
  }
  geometry.setAttribute("position", new THREE.BufferAttribute(renderPos, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(sim.uv, 2));

  // Each quad becomes two triangles, and each triangle is emitted only if all
  // three of its corners survived the armhole cut. Testing per TRIANGLE rather
  // than per quad keeps a diagonal half-quad along the rim, so the hole's edge
  // is stepped by half a cell instead of a whole one.
  const act = sim.active;
  const tri = [];
  for (let r = 0; r < sim.NROW - 1; r++) {
    for (let c = 0; c < sim.NCOL; c++) {
      const a = sim.idx(c, r), b = sim.idx(c + 1, r);
      const d = sim.idx(c, r + 1), e = sim.idx(c + 1, r + 1);
      if (act[a] && act[d] && act[b]) tri.push(a, d, b);
      if (act[b] && act[d] && act[e]) tri.push(b, d, e);
    }
  }
  for (const s of sim.sleeves) {
    const si = (c, r) => s.base + r * s.SCOL + ((c % s.SCOL) + s.SCOL) % s.SCOL;
    for (let r = 0; r < s.SROW - 1; r++) {
      for (let c = 0; c < s.SCOL; c++) {
        const a = si(c, r), b = si(c + 1, r);
        const d = si(c, r + 1), e = si(c + 1, r + 1);
        tri.push(a, d, b, b, d, e);
      }
    }

  }

  // ---- sew each sleeve to its armhole ----
  //
  // Done AFTER both panels exist, from the real topology: find the body's
  // boundary loops, take the one nearest each sleeve, and zipper it to the
  // sleeve's first ring. The loops carry different vertex counts — on the twin
  // the rim has 48 against the sleeve's 20 — so the walk advances whichever
  // loop is further behind, which closes the seam for any pair of counts.
  // Body-panel loops only. The sleeve tubes are open at both ends, so their own
  // rings are boundary loops as well — and the loop nearest a sleeve ring is
  // that very ring, at distance zero. Filtering by index keeps the search
  // looking at the panel it is supposed to sew to.
  const bodyLoops = boundaryLoops(tri)
    .filter((l) => l.length >= 4 && l.every((v) => v < sim.bodyCount));
  const centroid = (loop) => {
    const c = { x: 0, y: 0, z: 0 };
    for (const v of loop) { c.x += sim.pos[v * 4]; c.y += sim.pos[v * 4 + 1]; c.z += sim.pos[v * 4 + 2]; }
    c.x /= loop.length; c.y /= loop.length; c.z /= loop.length;
    return c;
  };
  const dist2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2;

  for (const s of sim.sleeves) {
    const si = (c, r) => s.base + r * s.SCOL + ((c % s.SCOL) + s.SCOL) % s.SCOL;
    const ring = [];
    for (let c = 0; c < s.SCOL; c++) ring.push(si(c, 0));
    const ringC = centroid(ring);

    let hole = null, bestD = Infinity;
    for (const l of bodyLoops) {
      if (l.used) continue;
      const d = dist2(centroid(l), ringC);
      if (d < bestD) { bestD = d; hole = l; }
    }
    if (!hole) continue;
    hole.used = true;                       // one armhole per sleeve

    // Start both loops at the same place, or the seam spirals.
    let off = 0, bo = Infinity;
    for (let k = 0; k < hole.length; k++) {
      const v = hole[k];
      const d = (sim.pos[v * 4] - sim.pos[ring[0] * 4]) ** 2 +
                (sim.pos[v * 4 + 1] - sim.pos[ring[0] * 4 + 1]) ** 2 +
                (sim.pos[v * 4 + 2] - sim.pos[ring[0] * 4 + 2]) ** 2;
      if (d < bo) { bo = d; off = k; }
    }

    const nR = hole.length, nS = ring.length;
    let i = 0, j = 0;
    while (i < nR || j < nS) {
      const r0 = hole[(off + i) % nR], sA = ring[j % nS];
      if (j >= nS || (i < nR && (i + 1) / nR <= (j + 1) / nS)) {
        const r1 = hole[(off + i + 1) % nR];
        if (r0 !== r1) tri.push(r0, sA, r1);
        i++;
      } else {
        tri.push(r0, ring[(j + 1) % nS], sA);
        j++;
      }
    }
  }

  geometry.setIndex(tri);
  geometry.computeVertexNormals();
  // The simulation moves vertices far outside the bounding volume computed
  // from their spawn positions, and three.js would frustum-cull the mesh out
  // of existence. Same trap as SkinnedMesh.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 100);
  return { geometry, renderPos };
}
