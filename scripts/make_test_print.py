#!/usr/bin/env python3
"""Generate a calibration print for the garment UV.

Why a calibration target rather than a logo: the garment's UV is anisotropic —
u spans the chest circumference (96.1 cm) and v the neck-to-hem length
(68.1 cm), so a naive placement stretches artwork horizontally by ~1.4x. A
CIRCLE makes that instantly visible; a logo hides it. The square and the ruler
ticks let you check placement and scale at the same time.

Read the result on screen:
  circle looks circular      -> aspect correction is working
  circle looks like an ellipse -> the cm-based sizing is not being applied
  crosshair sits centre-chest  -> u = 0.25 is correct for this UV
  square edges stay parallel    -> the front panel is not being distorted by drape

    python scripts/make_test_print.py            # writes web/assets/garment-print.png
    python scripts/make_test_print.py --plain    # a plain mark instead of a target
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "web", "assets", "garment-print.png")
SIZE = 2048               # gives headroom for a 4K kiosk later
INK = (250, 250, 250, 255)
ACCENT = (255, 92, 92, 255)


def font(px):
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def calibration():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = SIZE // 2
    r = int(SIZE * 0.44)

    # The circle is the whole point: any UV aspect error turns it into an ellipse.
    # Dark halo under the light ink: the swatches include a cream (#e8e4dc),
    # and a white-only target would be invisible on it.
    d.ellipse([c - r, c - r, c + r, c + r], outline=(20, 20, 20, 160),
              width=int(SIZE * 0.022))
    d.ellipse([c - r, c - r, c + r, c + r], outline=INK, width=int(SIZE * 0.012))
    # A square inscribed the same way — edges should stay straight and parallel.
    q = int(r * 0.707)
    d.rectangle([c - q, c - q, c + q, c + q], outline=ACCENT, width=int(SIZE * 0.008))
    # Crosshair for placement.
    d.line([c, c - r, c, c + r], fill=INK, width=int(SIZE * 0.004))
    d.line([c - r, c, c + r, c], fill=INK, width=int(SIZE * 0.004))

    # Ruler ticks every 1/10 of the print's width. At the default widthCm = 25
    # each gap is 2.5 cm on the garment, so you can measure the render.
    step = (2 * r) / 10
    for i in range(11):
        x = c - r + i * step
        long = i % 5 == 0
        h = int(SIZE * (0.035 if long else 0.02))
        d.line([x, c + r, x, c + r - h], fill=INK, width=int(SIZE * 0.004))

    f = font(int(SIZE * 0.055))
    d.text((c, c - int(r * 0.55)), "UV CHECK", font=f, fill=INK, anchor="mm")
    d.text((c, c + int(r * 0.55)), "circle must be round",
           font=font(int(SIZE * 0.032)), fill=ACCENT, anchor="mm")
    return img


def plain():
    """A neutral wordmark, for judging fabric and drape without a busy target."""
    img = Image.new("RGBA", (SIZE, int(SIZE * 0.42)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.text((w // 2, h // 2), "MIRRORFIT", font=font(int(h * 0.55)),
           fill=INK, anchor="mm")
    d.line([w * 0.18, h * 0.80, w * 0.82, h * 0.80], fill=ACCENT,
           width=max(2, int(h * 0.05)))
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", action="store_true",
                    help="plain wordmark instead of the calibration target")
    ap.add_argument("-o", "--out", default=OUT)
    a = ap.parse_args()

    img = plain() if a.plain else calibration()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # PNG-24 straight alpha, sRGB — matches the spec the material expects.
    img.save(a.out, "PNG")
    print(f"wrote {a.out}  {img.size[0]}x{img.size[1]}  "
          f"{os.path.getsize(a.out) / 1024:.0f} KB")
