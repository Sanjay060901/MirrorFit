"""Export an Anny body model to a skinned .glb with shape morph targets.

WHAT THIS PRODUCES
------------------
  web/assets/bodytwin.glb   mesh + skeleton + skin weights + shape morphs
  web/assets/bodyfit.json   what the browser needs to SOLVE for shape

WHY MORPH TARGETS
-----------------
A shopper walks up to the camera and must get THEIR twin, not a default body.
Shape therefore has to be settable in the browser at runtime — an offline bake
would mean one fixed body forever. glTF morph targets do exactly this, and
they blend linearly, which is only valid because we measured Anny's
phenotypes and found five of the six are linear to within a few millimetres
(see docs/OPERATION_PLAN.md). `age` is excluded: it is wildly non-linear
(221 mm error) because child->adult changes proportions qualitatively.

ONE TARGET PER PARAM, SIGNED
----------------------------
Target_i = mesh(param_i = 1.0) - mesh(param_i = 0.5), everything else at 0.5.
To render param value p, set influence = 2p - 1, so p=0 gives -1 and p=1
gives +1. Three.js is happy with negative morph influences, and this halves
the file compared with separate plus/minus targets.

THE JACOBIAN
------------
Morphing is only half the job — the browser also has to work out WHICH shape
matches the person. So we emit, for each param, how much each body
measurement changes per unit influence. The browser then solves a small
least-squares problem against the measurements from Stage 2.4. Measurements
are taken between bone heads, chosen to mirror what MediaPipe landmarks can
actually give us.

Usage:
    python scripts/bodytwin/export_anny_gltf.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

MAX_INFLUENCES = 4  # glTF JOINTS_0/WEIGHTS_0 capacity

# Linear phenotypes only — `age` is excluded, see module docstring.
SHAPE_PARAMS = ["gender", "muscle", "weight", "height", "proportions"]

# Measurements taken between bone heads. Each has a MediaPipe landmark
# equivalent so the browser can measure the same quantity on a real person.
#   name: (boneA, boneB)  or  (boneA, boneB, boneC, boneD) for midpoint pairs
MEASUREMENTS = {
    "shoulder_width": ("upperarm01.L", "upperarm01.R"),
    "hip_width":      ("upperleg01.L", "upperleg01.R"),
    "upper_arm":      ("upperarm01.L", "lowerarm01.L"),
    "forearm":        ("lowerarm01.L", "wrist.L"),
    "thigh":          ("upperleg01.L", "lowerleg01.L"),
    "shin":           ("lowerleg01.L", "foot.L"),
    "torso":          ("upperarm01.L", "upperarm01.R", "upperleg01.L", "upperleg01.R"),
}

# Silhouette widths, sampled as fractions of the shoulder->hip span.
#
# These matter because every measurement above is a distance between two
# joints, and joints tell you nothing about GIRTH — which is exactly what
# `muscle` and `weight` control, and why those params were unobservable.
# Silhouette width also avoids depth entirely: it is measured across the
# image plane, where both segmentation and MediaPipe are strong, unlike z.
SILHOUETTE_LEVELS = {"chest": 0.30, "waist": 0.62, "hips": 0.95}

# Only torso vertices count. In the rest pose the arms hang beside the body,
# so a naive x-extent at chest height would measure "torso + both arms".
TORSO_BONE_PREFIXES = ("spine", "pelvis", "breast")


def zup_to_yup(v: np.ndarray) -> np.ndarray:
    """Rotate -90 deg about X: Anny (x, y, z_up) -> glTF (x, z_up, -y)."""
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = v[..., 2]
    out[..., 2] = -v[..., 1]
    return out


def prune_to_four(indices: np.ndarray, weights: np.ndarray):
    """Keep the 4 heaviest influences per vertex, renormalise to sum to 1."""
    order = np.argsort(-weights, axis=1)[:, :MAX_INFLUENCES]
    rows = np.arange(weights.shape[0])[:, None]
    w = weights[rows, order]
    j = indices[rows, order]
    dropped = 1.0 - w.sum(axis=1)
    total = w.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    w = w / total
    j = np.where(w > 0, j, 0)
    return j.astype(np.uint16), w.astype(np.float32), dropped


def torso_vertex_mask(indices: np.ndarray, weights: np.ndarray,
                      bone_names: list[str]) -> np.ndarray:
    """Vertices whose dominant bone is part of the torso.

    Needed because the rest pose has the arms hanging beside the body, so a
    plain x-extent at chest height would measure torso PLUS both arms.
    """
    dominant = indices[np.arange(indices.shape[0]), np.argmax(weights, axis=1)]
    is_torso = np.array(
        [n.lower().startswith(TORSO_BONE_PREFIXES) for n in bone_names]
    )
    return is_torso[dominant]


def measure(heads: np.ndarray, name_to_idx: dict[str, int],
            verts: np.ndarray | None = None,
            torso_mask: np.ndarray | None = None) -> dict[str, float]:
    """Measurements in metres, in Anny's own space (z is up, x is across)."""
    out = {}
    for name, spec in MEASUREMENTS.items():
        if len(spec) == 2:
            a, b = (heads[name_to_idx[s]] for s in spec)
            out[name] = float(np.linalg.norm(a - b))
        else:
            a, b, c, d = (heads[name_to_idx[s]] for s in spec)
            out[name] = float(np.linalg.norm((a + b) / 2 - (c + d) / 2))

    if verts is None or torso_mask is None:
        return out

    # Silhouette widths: horizontal extent of the torso at fixed fractions of
    # the shoulder->hip span. Anny is z-up, so height is z and width is x.
    shoulder_z = float(
        (heads[name_to_idx["upperarm01.L"]][2] + heads[name_to_idx["upperarm01.R"]][2]) / 2)
    hip_z = float(
        (heads[name_to_idx["upperleg01.L"]][2] + heads[name_to_idx["upperleg01.R"]][2]) / 2)
    span = hip_z - shoulder_z

    tv = verts[torso_mask]
    slab = abs(span) * 0.06  # thin enough to be a "level", thick enough to catch vertices

    for label, t in SILHOUETTE_LEVELS.items():
        z = shoulder_z + t * span
        band = tv[np.abs(tv[:, 2] - z) < slab]
        # Fall back to the nearest ring of vertices rather than emitting a
        # bogus zero if the slab happens to land in a gap.
        if band.shape[0] < 8:
            order = np.argsort(np.abs(tv[:, 2] - z))[:64]
            band = tv[order]
        out[f"width_{label}"] = float(band[:, 0].max() - band[:, 0].min())

    return out


def build_glb(vertices, faces, joints, weights, bone_parents, bone_heads,
              bone_names, morph_deltas, morph_names, out_path: Path) -> None:
    from pygltflib import (
        GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
        Accessor, Skin, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT,
        UNSIGNED_INT, UNSIGNED_SHORT, SCALAR, VEC3, VEC4, MAT4,
    )

    n_bones = len(bone_parents)

    # Inverse bind matrices. Rest bones are pure translations, so the inverse
    # is a translation by -head. If Anny ever ships a rotated rest pose this
    # must become a real matrix inverse.
    ibm = np.tile(np.eye(4, dtype=np.float32), (n_bones, 1, 1))
    ibm[:, 0, 3] = -bone_heads[:, 0]
    ibm[:, 1, 3] = -bone_heads[:, 1]
    ibm[:, 2, 3] = -bone_heads[:, 2]
    ibm = np.transpose(ibm, (0, 2, 1)).copy()  # glTF is column-major

    local_t = bone_heads.copy()
    for i, p in enumerate(bone_parents):
        if p >= 0:
            local_t[i] = bone_heads[i] - bone_heads[p]

    children_of: dict[int, list[int]] = {i: [] for i in range(n_bones)}
    for i, p in enumerate(bone_parents):
        if p >= 0:
            children_of[p].append(i)

    nodes = [
        Node(name=bone_names[i],
             translation=[float(x) for x in local_t[i]],
             children=children_of[i] or None)
        for i in range(n_bones)
    ]
    mesh_node_index = n_bones
    roots = [i for i, p in enumerate(bone_parents) if p < 0]
    nodes.append(Node(name="BodyTwin", mesh=0, skin=0))

    blobs: list[bytes] = []
    views: list[BufferView] = []
    accessors: list[Accessor] = []
    offset = 0

    def add(data: np.ndarray, target, comp_type, acc_type, with_minmax=False):
        nonlocal offset
        raw = data.tobytes()
        pad = (-len(raw)) % 4
        blobs.append(raw + b"\x00" * pad)
        views.append(BufferView(buffer=0, byteOffset=offset,
                                byteLength=len(raw), target=target))
        offset += len(raw) + pad
        acc = Accessor(bufferView=len(views) - 1, componentType=comp_type,
                       count=data.shape[0], type=acc_type)
        if with_minmax:
            acc.min = [float(x) for x in data.min(axis=0)]
            acc.max = [float(x) for x in data.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    a_pos = add(vertices.astype(np.float32), ARRAY_BUFFER, FLOAT, VEC3, with_minmax=True)
    a_joints = add(joints, ARRAY_BUFFER, UNSIGNED_SHORT, VEC4)
    a_weights = add(weights, ARRAY_BUFFER, FLOAT, VEC4)
    a_idx = add(faces.astype(np.uint32).reshape(-1, 1), ELEMENT_ARRAY_BUFFER,
                UNSIGNED_INT, SCALAR)
    a_ibm = add(ibm.reshape(n_bones, 16), None, FLOAT, MAT4)

    # Morph targets need min/max too — some loaders rely on it for bounds.
    targets = []
    for delta in morph_deltas:
        a_d = add(delta.astype(np.float32), ARRAY_BUFFER, FLOAT, VEC3, with_minmax=True)
        targets.append({"POSITION": a_d})

    blob = b"".join(blobs)

    prim = Primitive(
        attributes=Attributes(POSITION=a_pos, JOINTS_0=a_joints, WEIGHTS_0=a_weights),
        indices=a_idx,
        targets=targets,
    )
    mesh = Mesh(primitives=[prim], weights=[0.0] * len(targets))
    # targetNames is how three.js populates morphTargetDictionary, which is
    # what lets the browser address morphs by name instead of index order.
    mesh.extras = {"targetNames": morph_names}

    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=roots + [mesh_node_index])],
        nodes=nodes,
        meshes=[mesh],
        skins=[Skin(joints=list(range(n_bones)), inverseBindMatrices=a_ibm,
                    skeleton=roots[0])],
        bufferViews=views,
        accessors=accessors,
        buffers=[Buffer(byteLength=len(blob))],
    )
    gltf.set_binary_blob(blob)
    gltf.save_binary(str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/assets/bodytwin.glb")
    ap.add_argument("--fit-out", default="web/assets/bodyfit.json")
    args = ap.parse_args()

    import anny

    model = anny.Anny(local_changes="default", facial_actions="none").to(dtype=torch.float32)
    name_to_idx = {n: i for i, n in enumerate(model.bone_labels)}
    neutral = {k: 0.5 for k in model.phenotype_labels}

    def run(**over):
        p = dict(neutral)
        p.update(over)
        with torch.no_grad():
            return model(phenotype_kwargs=p)

    faces = model.faces.cpu().numpy()
    idx = model.vertex_bone_indices.cpu().numpy()
    wts = model.vertex_bone_weights.cpu().numpy()
    parents = list(model.bone_parents)
    names = list(model.bone_labels)
    torso_mask = torso_vertex_mask(idx, wts, names)

    base = run()
    base_verts = base["vertices"][0].cpu().numpy()
    base_heads = base["rest_bone_heads"][0].cpu().numpy()
    base_meas = measure(base_heads, name_to_idx, base_verts, torso_mask)
    print(f"torso vertices: {int(torso_mask.sum())} / {torso_mask.size}")

    print(f"mesh: {base_verts.shape[0]} verts, {faces.shape[0]} faces, {len(parents)} bones")
    print(f"base measurements (m): "
          + ", ".join(f"{k}={v:.3f}" for k, v in base_meas.items()))

    # One signed morph target per linear phenotype, plus the Jacobian row
    # telling the browser how each measurement responds to it.
    morph_deltas, morph_names, jacobian, bone_deltas = [], [], {}, {}
    for param in SHAPE_PARAMS:
        out = run(**{param: 1.0})
        pverts = out["vertices"][0].cpu().numpy()
        delta = pverts - base_verts
        heads = out["rest_bone_heads"][0].cpu().numpy()
        meas = measure(heads, name_to_idx, pverts, torso_mask)

        morph_deltas.append(zup_to_yup(delta))
        morph_names.append(param)
        jacobian[param] = {k: meas[k] - base_meas[k] for k in base_meas}

        # glTF morph targets move MESH VERTICES ONLY — the skeleton is
        # untouched. Morphing the twin taller would stretch the skin while the
        # bones stayed at default proportions, and skinning would be wrong. So
        # ship the bone-head deltas too and let the browser move the joints to
        # match. Tiny next to the vertex data (104 bones vs 13718 verts).
        bone_deltas[param] = zup_to_yup(heads - base_heads).round(6).tolist()

        print(f"  {param:12s} max vertex delta {np.abs(delta).max()*1000:6.1f} mm"
              f"  max bone delta {np.abs(heads - base_heads).max()*1000:6.1f} mm")

    joints, weights, dropped = prune_to_four(idx, wts)
    print(f"influence pruning 9 -> {MAX_INFLUENCES}: worst vertex drops "
          f"{dropped.max():.4f}, mean {dropped.mean():.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_glb(zup_to_yup(base_verts), faces, joints, weights, parents,
              zup_to_yup(base_heads), names, morph_deltas, morph_names, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")

    fit_path = Path(args.fit_out)
    fit_path.write_text(json.dumps({
        "params": SHAPE_PARAMS,
        "measurements": list(base_meas.keys()),
        "silhouetteLevels": SILHOUETTE_LEVELS,
        "base": base_meas,
        # jacobian[param][measurement] = metres of change per +1 influence
        "jacobian": jacobian,
        # boneDeltas[param][boneIndex] = [dx, dy, dz] per +1 influence, already
        # in glTF (y-up) space
        "boneNames": names,
        "boneDeltas": bone_deltas,
        "note": ("influence = 2*p - 1 for phenotype p in [0,1]. age excluded "
                 "(non-linear). muscle/weight barely affect skeletal "
                 "measurements — they change girth, so they are close to "
                 "unobservable from landmarks and need the segmentation "
                 "silhouette to estimate."),
    }, indent=2))
    print(f"wrote {fit_path}")


if __name__ == "__main__":
    main()
