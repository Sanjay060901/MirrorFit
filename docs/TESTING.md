# Testing doctrine

These rules exist because of one specific incident. Read that first — the rules
are unmemorable without it.

## The incident

`web/tests/garment-geometry.html` reported every capsule correct. The live page
was badly broken at the same moment: every capsule came out round, `dz=0`, with
radii that were really "distance from the model origin to this body part".

The test measured a bare `gltf.scene` sitting at the identity. The live page
nests that same model under a group carrying a 279x metres-to-pixels scale and
a negative x axis for the mirror. `measureCapsules` called
`updateMatrixWorld(true)`, which walks *down* only, so it computed
`skinned.matrixWorld` against a parent matrix that was still the identity. The
first `bone.getWorldPosition()` then internally called
`updateWorldMatrix(true, false)`, which walks *up* and refreshed the parent —
and `root` with it — to the real 279x scale. `worldToLocal` then divided every
vertex by a 279 it had never been multiplied by, collapsing the mesh to a speck
at the origin.

Nothing threw. No vertex landed inside any capsule's t-band, the
sideways/front/back classification never reached its 8-sample threshold, and
every capsule fell back to a round radius with a `console.warn` nobody was
blocking on.

So: a green test, a broken page, and a silent fallback in between. Each rule
below is one of those three failures.

## The rules

**1. Reproduce the live transform chain.** A fixture at the identity does not
test a page whose model hangs under a scaled, mirrored parent. Nest fixtures
exactly as the real page does, including the ordering that leaves matrices
stale. `garment-geometry.html` now does this deliberately, and comments say so,
because "helpfully" calling `updateMatrixWorld` on the parent would destroy the
condition the bug needs.

**2. Assert against ground truth, not against self.** Compare to measured
values — `body-slices.html` reports the twin's real half-width and half-depth
by height — not to whatever the code currently prints. A test that pins current
behaviour detects *change*, not *correctness*, and would have happily locked in
the collapsed radii.

**3. Prove the test has teeth.** For every new test: reintroduce the bug or
inject a plausible one, confirm it fails, restore, confirm it passes. Record it
in the commit. An unfalsified test is an assumption wearing a test's clothing.

**4. Fail loudly, never silently.** A fallback that changes results must throw
or raise a visible degraded flag that the debug panel shows. `console.warn` is
not a failure mode; it is a diary entry.

**5. Numbers, not adjectives.** Output states measured, expected, and
tolerance. Bare `ok` is not evidence.

**6. "It ran" is not "it works".** Before claiming something works, say whether
it was observed in the user's actual configuration or whether code merely
executed without throwing. If the real numbers from the real setup have not
been seen, say so plainly.

## Running the headless checks

Needs the dev server (`bash scripts/serve.sh`). No camera and no WebGPU
required, which is what makes them usable in headless Edge.

```sh
"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
  --headless=new --disable-gpu --virtual-time-budget=30000 --dump-dom \
  http://localhost:8000/web/tests/garment-geometry.html
```

Or open the page in a browser. See `web/tests/README.md` for what each one
answers.
