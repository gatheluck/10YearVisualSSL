"""Task-specific native-coordinate metrics; no cross-task metric substitution."""

import numpy as np


def require(condition, code, detail=""):
    if not condition:
        raise ValueError(code)


def finite(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite metric input")
    return value


def decode_heatmaps(heat, crop):
    crop = finite(crop)
    if crop.shape != (4,) or (crop[2:] <= 0).any():
        raise ValueError("invalid person crop")
    heat = np.asarray(heat, np.float64)
    if heat.ndim != 3 or not np.isfinite(heat).all():
        raise ValueError("pose: invalid heatmap prediction")
    j, h, w = heat.shape
    idx = heat.reshape(j, -1).argmax(-1)
    xy = np.stack([idx % w, idx // w], axis=-1).astype(float)
    # Standard quarter-pixel heatmap offset, then inverse person-crop transform.
    for k, (x, y) in enumerate(xy.astype(int)):
        if 1 < x < w - 1 and 1 < y < h - 1:
            xy[k, 0] += np.sign(heat[k, y, x + 1] - heat[k, y, x - 1]) * 0.25
            xy[k, 1] += np.sign(heat[k, y + 1, x] - heat[k, y - 1, x]) * 0.25
    peak = heat.reshape(j, -1).max(-1)
    xy = xy / [w, h] * crop[2:] + crop[:2]
    return xy, peak


def pckh_counts(pred, pos_gt_src, jnt_missing, headboxes_src):
    """HRNet mpii.evaluate: pred is zero-based; MAT coordinates stay one-based."""
    pred, pos, heads = finite(pred), finite(pos_gt_src), finite(headboxes_src)
    missing = np.asarray(jnt_missing)
    if (
        pos.ndim != 3
        or pos.shape[1:] != (16, 2)
        or pred.shape != pos.shape
        or missing.shape != pos.shape[:2]
        or heads.shape != (len(pos), 2, 2)
        or not np.isin(missing, [0, 1]).all()
    ):
        raise ValueError("MPII_HRNET_GT_VAL_SHAPE")
    headsize = 0.6 * np.linalg.norm(heads[:, 1] - heads[:, 0], axis=1)
    if not len(pos) or np.any(headsize <= 0):
        raise ValueError("MPII_HRNET_GT_VAL_HEADSIZE")
    visible = (1 - missing).astype(bool)
    visible[:, [6, 7]] = False
    uv_err = np.linalg.norm((pred + 1) - pos, axis=-1)
    correct = uv_err / headsize[:, None] <= 0.5
    return int((correct & visible).sum()), int(visible.sum())


CROWDPOSE_SIGMAS = (
    np.array(
        [
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
            0.79,
            0.79,
        ]
    )
    / 10
)


def oks(pred, gt, area, bbox):
    p, g = finite(pred).reshape(14, 3), finite(gt).reshape(14, 3)
    x, y, w, h = finite(bbox).reshape(4)
    area = 0.53 * w * h  # crowdposetools cocoeval.py:258, even if an area was supplied
    if w <= 0 or h <= 0:
        raise ValueError("CrowdPose: invalid bbox")
    visible = g[:, 2] > 0
    if visible.any():
        delta = p[:, :2] - g[:, :2]
    else:
        x, y, w, h = finite(bbox).reshape(4)
        delta = np.stack(
            (
                np.maximum(x - w - p[:, 0], 0) + np.maximum(p[:, 0] - (x + 2 * w), 0),
                np.maximum(y - h - p[:, 1], 0) + np.maximum(p[:, 1] - (y + 2 * h), 0),
            ),
            axis=1,
        )
    e = (delta**2).sum(-1) / ((2 * CROWDPOSE_SIGMAS) ** 2) / (area + np.spacing(1)) / 2
    return float(np.exp(-e[visible] if visible.any() else -e).mean())


class KeypointAP:
    """CrowdPose COCO keypoint AP, area=all, maxDets=20, 101 recall points."""

    def __init__(self):
        self.subsets = {}
        self.records = []
        self.n = 0
        self.ngt = 0
        self.seen = set()

    def add_image(self, image_id, gt, pred, crowd_index=None):
        if crowd_index is not None:
            value = float(crowd_index)
            if not np.isfinite(value):
                raise ValueError("CrowdPose: nonfinite crowdIndex")
            label = "easy" if value < 0.2 else "medium" if value < 0.8 else "hard"
            self.subsets.setdefault(label, KeypointAP()).add_image(image_id, gt, pred)
        if image_id in self.seen:
            raise ValueError("CrowdPose: duplicate image")
        self.seen.add(image_id)
        ignored = lambda a: (
            bool(a.get("iscrowd", 0))
            or int(
                a.get(
                    "num_keypoints",
                    np.count_nonzero(
                        np.asarray(a["keypoints"]).reshape(14, 3)[:, 2] > 0
                    ),
                )
            )
            == 0
        )
        gt = sorted(gt, key=ignored)
        pred = sorted(pred, key=lambda a: -float(a["score"]))[:20]
        scores = finite([p["score"] for p in pred])
        sim = np.array(
            [
                [
                    oks(p["keypoints"], g["keypoints"], g.get("area"), g["bbox"])
                    for g in gt
                ]
                for p in pred
            ]
        ).reshape(len(pred), len(gt))
        gi = np.array([ignored(g) for g in gt], bool)
        crowd = np.array([bool(g.get("iscrowd", 0)) for g in gt])
        dt, di = np.zeros((10, len(pred)), bool), np.zeros((10, len(pred)), bool)
        for k, threshold in enumerate(np.linspace(0.5, 0.95, 10)):
            matched = np.zeros(len(gt), bool)
            for d in range(len(pred)):
                best, best_iou = -1, min(threshold, 1 - 1e-10)
                for j in range(len(gt)):
                    if matched[j] and not crowd[j]:
                        continue
                    if best >= 0 and not gi[best] and gi[j]:
                        break
                    if sim[d, j] < best_iou:
                        continue
                    best, best_iou = j, sim[d, j]
                if best >= 0:
                    dt[k, d], di[k, d], matched[best] = True, gi[best], True
        self.records.append((scores, dt, di))
        self.ngt += int((~gi).sum())
        self.n += 1

    def result(self):
        if self.n == 0 or self.ngt == 0:
            raise ValueError("CrowdPose: no evaluable ground truth")
        scores = np.concatenate([r[0] for r in self.records])
        order = np.argsort(-scores, kind="stable")
        dt = np.concatenate([r[1] for r in self.records], axis=1)[:, order]
        di = np.concatenate([r[2] for r in self.records], axis=1)[:, order]
        aps = []
        for k in range(10):
            tp, fp = np.cumsum(dt[k] & ~di[k]), np.cumsum(~dt[k] & ~di[k])
            rec = tp / self.ngt
            precision = tp / np.maximum(tp + fp, np.spacing(1))
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
            q = np.zeros(101)
            valid = idx < len(precision)
            q[valid] = precision[idx[valid]]
            aps.append(float(q.mean()))
        result = {
            "primary": float(np.mean(aps)),
            "keypoint_AP": float(np.mean(aps)),
            "AP50": aps[0],
            "AP75": aps[5],
            "n": self.n,
        }
        if self.subsets:
            for label in ("easy", "medium", "hard"):
                subset = self.subsets.get(label)
                result["AP_" + label] = (
                    round(subset.result()["primary"], 4)
                    if subset and subset.ngt
                    else -1.0
                )
            result["crowd_subset_images"] = {k: v.n for k, v in self.subsets.items()}
        return result


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    require(
        q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 1e-10,
        "POSE_QUATERNION",
        q.tolist(),
    )
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def check_pose(T):
    T = np.asarray(T, dtype=float)
    require(
        T.shape == (4, 4) and np.isfinite(T).all() and np.allclose(T[3], [0, 0, 0, 1]),
        "POSE_MATRIX",
        "",
    )
    R = T[:3, :3]
    require(
        np.allclose(R.T @ R, np.eye(3), atol=2e-4) and abs(np.linalg.det(R) - 1) < 2e-4,
        "POSE_NOT_SO3",
        "",
    )
    return T


SEVEN_ORTH_BOUND = 2 * 0.0006874892261451437

SEVEN_DET_BOUND = 2 * 0.0009653044666610988


def project_so3(R):
    R = np.asarray(R, dtype=float)
    require(R.shape == (3, 3) and np.isfinite(R).all(), "POSE_MATRIX", "")
    U, _, Vt = np.linalg.svd(R)
    U[:, -1] *= 1 if np.linalg.det(U @ Vt) >= 0 else -1
    return U @ Vt


def check_seven_pose(T):
    T = np.asarray(T, dtype=float)
    require(
        T.shape == (4, 4)
        and np.isfinite(T).all()
        and np.array_equal(T[3], [0, 0, 0, 1]),
        "POSE_MATRIX",
        "",
    )
    R = T[:3, :3]
    require(
        np.abs(R.T @ R - np.eye(3)).max() <= SEVEN_ORTH_BOUND
        and abs(np.linalg.det(R) - 1) <= SEVEN_DET_BOUND,
        "POSE_NOT_SO3",
        "",
    )
    return T


def pose_errors(prediction, truth, *, seven_scenes=False):
    gt = check_seven_pose(truth) if seven_scenes else check_pose(truth)
    if prediction is None:
        return float("inf"), float("inf")
    pred = check_pose(prediction)
    t = float(np.linalg.norm(pred[:3, 3] - gt[:3, 3]))
    Rp, Rg = pred[:3, :3], gt[:3, :3]
    if seven_scenes:
        Rp, Rg = project_so3(Rp), project_so3(Rg)
    cos = np.clip((np.trace(Rp.T @ Rg) - 1) / 2, -1, 1)
    return t, float(np.degrees(np.arccos(cos)))


class CameraScore:
    def __init__(self, dataset, scenes):
        if (
            dataset not in ("seven_scenes", "cambridge_landmarks")
            or not scenes
            or len(set(scenes)) != len(scenes)
        ):
            raise ValueError("explicit camera dataset and scene vocabulary required")
        self.dataset, self.scenes = dataset, list(scenes)
        self.errors = {s: [] for s in scenes}

    def update(self, scene, prediction, truth):
        if scene not in self.errors:
            raise ValueError("unknown evaluation scene")
        self.errors[scene].append(
            pose_errors(prediction, truth, seven_scenes=self.dataset == "seven_scenes")
        )

    def result(self):
        if not all(self.errors.values()):
            raise ValueError("missing camera scene evaluation")
        medians = np.array([np.median(v, axis=0) for v in self.errors.values()])
        return {
            "median_translation_m": float(medians[:, 0].mean()),
            "median_rotation_deg": float(medians[:, 1].mean()),
            "images": sum(map(len, self.errors.values())),
            "per_scene": {
                s: {"translation_m": float(v[0]), "rotation_deg": float(v[1])}
                for s, v in zip(self.scenes, medians)
            },
        }
