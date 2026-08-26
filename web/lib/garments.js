// MirrorFit — garment catalogue, size grading, and the fit engine
// ============================================================================
//
// A garment TYPE is a set of pattern parameters, and a SIZE is an ease
// allowance applied to those parameters. That is not a simplification for our
// convenience — it is how grading actually works in the trade. A size M and a
// size L of the same style share a block; what differs is how much room is cut
// beyond the body.
//
// Keeping it that way means wrong-size behaviour is emergent: an XS on a large
// frame produces negative ease, the solver cannot satisfy the constraints
// without stretching, and you see strain. Nothing scripts that.
//
// THE FIT ENGINE HERE DOES NOT DEPEND ON SHAPE FITTING. It compares the drafted
// garment against the MEASURED COLLISION CAPSULES, which are taken from the
// body mesh and agree with it to within 5 mm. Shape fitting — inferring the
// wearer's build from a webcam — does not converge on frontal RGB and is
// refused elsewhere in the app. This is the honest fit signal we can actually
// produce today.

import { ellipsePerimeter } from "./cloth.js";

// ---------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------
//
// Every field is a real pattern dimension:
//   neckCircum   finished neck opening, metres
//   sleeve       fraction of the upper-arm bone the sleeve covers
//   lengthRatio  shoulder-to-hem over shoulder-to-hip; a men's tee is ~1.37
//   baseEase     ease at size M, metres, added to every body radius
//   drape        bend compliance; a hoodie is far stiffer than a jersey tee

// Ease is written in CENTIMETRES OF CHEST CIRCUMFERENCE, because that is how
// every size chart in the trade is written, and converting once here is far
// safer than carrying radii around. A first pass specified it as a radius and
// the numbers looked plausible — until the test printed a size chart with
// 12.5 cm between sizes and 25 cm of ease at M, roughly 2.5x a real tee.
const cmToRadius = (cm) => cm / 100 / (2 * Math.PI);

export const GARMENTS = {
  tee: {
    id: "tee", name: "Crew T-Shirt", blurb: "Cotton jersey, set-in sleeve",
    neckCircum: 0.42, sleeve: 0.34, lengthRatio: 1.37,
    easeCm: 10, drape: 4e-6,
  },
  polo: {
    id: "polo", name: "Polo", blurb: "Piqué knit, ribbed collar",
    neckCircum: 0.40, sleeve: 0.40, lengthRatio: 1.34,
    easeCm: 8, drape: 2e-6,
  },
  longSleeve: {
    id: "longSleeve", name: "Long Sleeve", blurb: "Jersey, full sleeve",
    neckCircum: 0.42, sleeve: 0.95, lengthRatio: 1.39,
    easeCm: 10, drape: 4e-6,
  },
  hoodie: {
    id: "hoodie", name: "Hoodie", blurb: "Brushed fleece, relaxed",
    neckCircum: 0.46, sleeve: 0.95, lengthRatio: 1.30,
    // Fleece is thick and stiff: more ease because the fabric itself occupies
    // room, and an order of magnitude less compliance so it holds its shape
    // rather than clinging.
    easeCm: 20, drape: 4e-7,
  },
  dress: {
    id: "dress", name: "T-Shirt Dress", blurb: "Jersey, mid-thigh",
    neckCircum: 0.42, sleeve: 0.26, lengthRatio: 2.05,
    easeCm: 12, drape: 8e-6,
  },
};

// High-street grading is about 5 cm of chest per size. The first version used
// 20 mm of RADIUS, which is 12.6 cm around — two and a half sizes per step.
export const SIZES = ["XS", "S", "M", "L", "XL", "XXL"];
const SIZE_STEP_CM = 5;

export function easeFor(garment, size) {
  const i = SIZES.indexOf(size);
  const m = SIZES.indexOf("M");
  const cm = garment.easeCm + (i < 0 ? 0 : (i - m) * SIZE_STEP_CM);
  return cmToRadius(cm);
}

/** Pattern parameters ready to hand to draftGarment(). */
export function patternFor(garmentId, size) {
  const g = GARMENTS[garmentId] ?? GARMENTS.tee;
  return {
    garment: g,
    size,
    ease: easeFor(g, size),
    easeCm: g.easeCm,
    neckCircum: g.neckCircum,
    sleeveLength: g.sleeve,
    lengthRatio: g.lengthRatio,
    drape: g.drape,
  };
}

// ---------------------------------------------------------------------------
// Fit engine
// ---------------------------------------------------------------------------

// Ease bands in CENTIMETRES OF CIRCUMFERENCE, which is how a size chart is
// written. Below the tight bound the garment must stretch to close; above the
// loose bound it hangs off the body rather than following it.
//
// These are for knitwear. A woven shirt needs more ease at every level because
// it cannot stretch to accommodate movement, which is exactly the kind of
// per-fabric difference a real size chart encodes.
const BANDS = [
  { max: -2, verdict: "too tight", note: "will not close without stretching" },
  { max: 4, verdict: "tight", note: "close-fitting, shows body line" },
  { max: 14, verdict: "true to size", note: "sits with normal room" },
  { max: 24, verdict: "loose", note: "relaxed, drapes away from the body" },
  { max: Infinity, verdict: "too loose", note: "oversized, hangs off the frame" },
];

function band(easeCm) {
  return BANDS.find((b) => easeCm <= b.max);
}

/**
 * Compare a drafted garment against the measured body, zone by zone.
 *
 * Works off the collision capsules rather than any inferred body shape, so it
 * is as accurate as the capsule measurement — about 5 mm — and carries no
 * dependency on shape fitting converging.
 *
 * @returns { zones: [...], verdict, worst }
 */
export function fitReport(caps, ease) {
  const byLabel = new Map(caps.map((c) => [c.label, c]));
  const ZONES = [
    { key: "chest", label: "Chest", weight: 3 },
    { key: "waist", label: "Waist", weight: 2 },
    { key: "hips", label: "Hip", weight: 1 },
  ];

  const zones = [];
  for (const z of ZONES) {
    const c = byLabel.get(z.key);
    if (!c) continue;
    // Circumference, not radius: a size chart is written in circumference and
    // a 20 mm radius change is a 12 cm change around, which is a whole size.
    const body = ellipsePerimeter(c.rx, c.rz) * 100;
    const garment = ellipsePerimeter(c.rx + ease, c.rz + ease) * 100;
    const easeCm = garment - body;
    const b = band(easeCm);
    zones.push({
      ...z, bodyCm: body, garmentCm: garment, easeCm,
      verdict: b.verdict, note: b.note,
    });
  }

  // The chest governs whether a top fits — you can tolerate a loose waist, but
  // a tight chest is the one that makes a garment unwearable. Weighting rather
  // than averaging keeps that true.
  const worst = zones.slice().sort((a, b2) =>
    (Math.abs(a.easeCm - 9) * a.weight) - (Math.abs(b2.easeCm - 9) * b2.weight)).pop();
  const chest = zones.find((z) => z.key === "chest") ?? worst;

  return { zones, verdict: chest?.verdict ?? "unknown", worst };
}

/**
 * Which size sits closest to the middle of the true-to-size band.
 * This is the recommendation a shopper actually wants.
 */
export function recommendSize(caps, garmentId) {
  const g = GARMENTS[garmentId] ?? GARMENTS.tee;
  const TARGET_EASE_CM = 9;          // centre of the true-to-size band
  let best = null;
  for (const size of SIZES) {
    const r = fitReport(caps, easeFor(g, size));
    const chest = r.zones.find((z) => z.key === "chest");
    if (!chest) continue;
    const off = Math.abs(chest.easeCm - TARGET_EASE_CM);
    if (!best || off < best.off) best = { size, off, easeCm: chest.easeCm, report: r };
  }
  return best;
}
