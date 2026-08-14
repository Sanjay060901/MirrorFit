# Headless checks

Pages that exercise the non-GPU half of `web/lib/cloth.js` against the real
`bodytwin.glb` and print numbers. They need no camera and no WebGPU, so they
run in headless Chrome/Edge — which makes them usable as a regression check
before touching the capsule or drafting code.

| page | what it answers |
|---|---|
| `garment-geometry.html` | Do the capsules measure sane radii? Does `draftShirt` produce sleeves, armholes and no NaNs, at several ease values? |
| `body-slices.html` | Ground truth: the twin's real half-width and half-depth at 40 mm height intervals. Use this to check a capsule, never the other way round. |
| `bone-ownership.html` | Which bone dominates which vertices, and where those vertices sit. Use it to choose an `owners` list. |

Run one:

```sh
bash scripts/serve.sh                       # in another terminal
"$PROGRAMFILES (x86)/Microsoft/Edge/Application/msedge.exe" \
  --headless=new --disable-gpu --virtual-time-budget=30000 --dump-dom \
  http://localhost:8000/web/tests/garment-geometry.html
```

Or just open them in a browser.

## Two bugs these found

Both were invisible from looking at the code, and both would have read as
"the shirt just doesn't fit right":

- **The hips capsule ran sideways.** Between the two leg joints, `rx`/`rz` —
  which assume a vertical capsule — measured the pelvis's *vertical* extent.
  It reported a blob 0.246 m wide and 0.144 m deep for a pelvis that
  `body-slices.html` says is 0.146 x 0.094.
- **`spine05` doesn't own the hips.** `bone-ownership.html` shows it dominates
  only the inner crotch (|x| <= 0.057), so measuring hip girth from it gave a
  0.052 m half-width. The hip flare belongs to `pelvisL/R` and `spine04`.

The lesson worth keeping: a capsule's radius is only meaningful relative to the
vertices it was measured from, and the skinning decides which those are.
