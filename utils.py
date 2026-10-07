import torch
import config
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from VOC import decode_label_tensor

import numpy as np


BOX_ATTRS = 6
BOX_CONF = 0
BOX_X = 1
BOX_Y = 2
BOX_W = 3
BOX_H = 4
BOX_SIN = 5


def get_box_tensor(data, box_idx):
    start = config.C + (box_idx * BOX_ATTRS)
    end = start + BOX_ATTRS
    return data[..., start:end]


def box_geometry(box_tensor):
    return torch.stack([
        box_tensor[..., BOX_X],
        box_tensor[..., BOX_Y],
        box_tensor[..., BOX_W],
        box_tensor[..., BOX_H],
        box_tensor[..., BOX_SIN],
    ], dim=-1)

def monte_carlo_iou(pred_boxes, gt_boxes, num_samples=1000, device=None):
    if device is None:
        device = pred_boxes.device
    N = pred_boxes.shape[0]

    def ellipse_aabb(boxes):
        cx, cy = boxes[:, 0], boxes[:, 1]
        a      = boxes[:, 2] / 2
        b      = boxes[:, 3] / 2
        sin_a  = boxes[:, 4]

        # Since angle is between -20 and 20, cos(theta) is always positive
        cos_a = torch.sqrt(1 - sin_a**2 + 1e-6)

        hx = torch.sqrt((a * cos_a)**2 + (b * sin_a)**2)
        hy = torch.sqrt((a * sin_a)**2 + (b * cos_a)**2)
        return cx - hx, cx + hx, cy - hy, cy + hy

    px1, px2, py1, py2 = ellipse_aabb(pred_boxes)
    gx1, gx2, gy1, gy2 = ellipse_aabb(gt_boxes)

    x1 = torch.min(px1, gx1)
    x2 = torch.max(px2, gx2)
    y1 = torch.min(py1, gy1)
    y2 = torch.max(py2, gy2)

    rand_x = torch.rand((N, num_samples), device=device) * (x2 - x1).unsqueeze(1) + x1.unsqueeze(1)
    rand_y = torch.rand((N, num_samples), device=device) * (y2 - y1).unsqueeze(1) + y1.unsqueeze(1)
    points = torch.stack([rand_x, rand_y], dim=-1)

    def is_inside(pts, boxes):
        sin_a = boxes[:, 4].unsqueeze(1)
        
        # Since angle is between -20 and 20, cos(theta) is always positive
        cos_a = torch.sqrt(1 - sin_a**2 + 1e-6)

        dx = pts[..., 0] - boxes[:, 0].unsqueeze(1)
        dy = pts[..., 1] - boxes[:, 1].unsqueeze(1)

        rot_x = dx * cos_a + dy * sin_a
        rot_y = -dx * sin_a + dy * cos_a

        a = boxes[:, 2].unsqueeze(1) / 2
        b = boxes[:, 3].unsqueeze(1) / 2

        return (rot_x / a)**2 + (rot_y / b)**2 <= 1

    in_pred = is_inside(points, pred_boxes)
    in_gt   = is_inside(points, gt_boxes)

    intersection = (in_pred & in_gt).float().sum(dim=1)
    union        = (in_pred | in_gt).float().sum(dim=1)

    return intersection / union.clamp(min=1e-6)

def get_yolo_iou(output, target):
    batch_size = output.shape[0]
    pred_boxes = []
    target_boxes = []

    for box_idx in range(config.B):
        pred_boxes.append(box_geometry(get_box_tensor(output, box_idx)))
        target_boxes.append(box_geometry(get_box_tensor(target, box_idx)))

    pred_boxes = torch.stack(pred_boxes, dim=-2).reshape(-1, 5)
    target_boxes = torch.stack(target_boxes, dim=-2).reshape(-1, 5)

    iou = monte_carlo_iou(pred_boxes, target_boxes)
    return iou.view(batch_size, config.S, config.S, config.B)


def ellipse_nms_per_class(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    num_mc_samples: int = 500,
) -> torch.Tensor:
    """
    Runs ellipse_nms independently for each class so a car and a person
    that heavily overlap do not suppress each other.

    Args:
        boxes:      (N, 5)  — [cx, cy, w, h, sin_angle]
        scores:     (N,)    — confidence scores
        class_ids:  (N,)    — integer class index per box
        ...rest same as ellipse_nms...

    Returns:
        kept_indices: (K,) LongTensor into the original N boxes.
    """
    all_kept = []
    for cls in class_ids.unique():
        cls_mask = class_ids == cls
        cls_indices = cls_mask.nonzero(as_tuple=False).squeeze(1)

        kept_local = ellipse_nms(
            boxes[cls_indices],
            scores[cls_indices],
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            num_mc_samples=num_mc_samples,
        )
        # Map local indices back to global
        all_kept.append(cls_indices[kept_local])

    if not all_kept:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    return torch.cat(all_kept)

def ellipse_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    num_mc_samples: int = 1000,
) -> torch.Tensor:
    """
    Non-Maximum Suppression for rotated ellipses using Monte Carlo IoU.

    Args:
        boxes:           (N, 5) tensor — [cx, cy, w, h, sin_angle], same format
                         as the rest of utils.py / monte_carlo_iou.
        scores:          (N,) confidence scores for each box.
        iou_threshold:   Suppress candidates whose MC-IoU with the kept box
                         exceeds this value.  Default 0.5.
        score_threshold: Discard boxes below this confidence before NMS.
                         Default 0.25.
        num_mc_samples:  Passed straight through to monte_carlo_iou.
                         Lower = faster but noisier IoU estimates.

    Returns:
        kept_indices:    (K,) LongTensor of indices into the original N boxes,
                         sorted by descending score.
    """
    # ------------------------------------------------------------------ #
    # 1. Confidence gate — cheap pre-filter before any IoU work           #
    # ------------------------------------------------------------------ #
    mask = scores > score_threshold
    if mask.sum() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    candidate_idx = mask.nonzero(as_tuple=False).squeeze(1)  # original indices
    candidate_boxes  = boxes[candidate_idx]                  # (M, 5)
    candidate_scores = scores[candidate_idx]                 # (M,)

    # ------------------------------------------------------------------ #
    # 2. Sort candidates by descending score                              #
    # ------------------------------------------------------------------ #
    order = candidate_scores.argsort(descending=True)
    candidate_idx    = candidate_idx[order]
    candidate_boxes  = candidate_boxes[order]

    kept = []

    # ------------------------------------------------------------------ #
    # 3. Greedy NMS loop                                                  #
    # ------------------------------------------------------------------ #
    while candidate_boxes.shape[0] > 0:
        # Always keep the highest-scoring surviving box
        kept.append(candidate_idx[0].item())

        if candidate_boxes.shape[0] == 1:
            break  # nothing left to compare

        top_box  = candidate_boxes[0:1]          # (1, 5)
        rest     = candidate_boxes[1:]            # (M-1, 5)
        rest_idx = candidate_idx[1:]              # (M-1,)

        # Batch all remaining boxes against the top box in one MC call.
        # monte_carlo_iou expects (N, 5) pairs, so we tile the top box.
        top_tiled = top_box.expand(rest.shape[0], -1)   # (M-1, 5)
        ious = monte_carlo_iou(top_tiled, rest, num_samples=num_mc_samples)

        # Keep only those with IoU below the suppression threshold
        survive = ious <= iou_threshold
        candidate_boxes  = rest[survive]
        candidate_idx    = rest_idx[survive]

    return torch.tensor(kept, dtype=torch.long, device=boxes.device)

def _compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    101-point interpolated AP (COCO/VOC2010+ standard).
    Integrates the precision-recall curve at recall thresholds 0, 0.01, ..., 1.0.
    """
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        prec_at_thr = precisions[recalls >= thr]
        ap += prec_at_thr.max() if prec_at_thr.size > 0 else 0.0
    return ap / 101
def _collect_boxes_single_image(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    conf_threshold: float,
):
    IMG_SIZE = 448.0
    S = config.S
    cell_size = IMG_SIZE / S

    pred_boxes, pred_scores, pred_cls = [], [], []
    gt_boxes, gt_cls = [], []

    # ---- predictions -------------------------------------------------- #
    for box_idx in range(config.B):
        bt       = get_box_tensor(pred, box_idx)
        conf_map = bt[..., BOX_CONF]
        mask     = conf_map > conf_threshold

        if not mask.any():
            continue

        rows, cols = mask.nonzero(as_tuple=True)
        bt_masked  = bt[mask]

        cx_px = (cols.float() + bt_masked[:, BOX_X]) * cell_size
        cy_px = (rows.float() + bt_masked[:, BOX_Y]) * cell_size
        w_px  = bt_masked[:, BOX_W] * IMG_SIZE
        h_px  = bt_masked[:, BOX_H] * IMG_SIZE
        sin_a = bt_masked[:, BOX_SIN]

        geoms   = torch.stack([cx_px, cy_px, w_px, h_px, sin_a], dim=-1)
        confs   = conf_map[mask]
        cls_ids = pred[..., :config.C][mask].argmax(dim=-1)

        pred_boxes.append(geoms)
        pred_scores.append(confs)
        pred_cls.append(cls_ids)

    # ---- ground truth — one box per occupied cell only ---------------- #
    # In YOLO, both box slots in a cell may have conf=1 during training,
    # but only one object is assigned per cell. We deduplicate by cell.
    seen_cells = set()
    for box_idx in range(config.B):
        bt       = get_box_tensor(tgt, box_idx)
        conf_map = bt[..., BOX_CONF]
        mask     = conf_map > 0.5

        if not mask.any():
            continue

        rows, cols = mask.nonzero(as_tuple=True)
        bt_masked  = bt[mask]
        cls_probs  = tgt[..., :config.C][mask]

        for k in range(rows.shape[0]):
            cell = (rows[k].item(), cols[k].item())
            if cell in seen_cells:
                continue                          # deduplicate
            seen_cells.add(cell)

            cx_px = (cols[k].float() + bt_masked[k, BOX_X]) * cell_size
            cy_px = (rows[k].float() + bt_masked[k, BOX_Y]) * cell_size
            w_px  = bt_masked[k, BOX_W] * IMG_SIZE
            h_px  = bt_masked[k, BOX_H] * IMG_SIZE
            sin_a = bt_masked[k, BOX_SIN]

            gt_boxes.append(torch.stack([cx_px, cy_px, w_px, h_px, sin_a]))
            gt_cls.append(cls_probs[k].argmax())

    pred_boxes_t  = torch.cat(pred_boxes,  dim=0) if pred_boxes  else None
    pred_scores_t = torch.cat(pred_scores, dim=0) if pred_scores else None
    pred_cls_t    = torch.cat(pred_cls,    dim=0) if pred_cls    else None
    gt_boxes_t    = torch.stack(gt_boxes,  dim=0) if gt_boxes    else None
    gt_cls_t      = torch.stack(gt_cls,    dim=0) if gt_cls      else None

    return pred_boxes_t, pred_scores_t, pred_cls_t, gt_boxes_t, gt_cls_t

def evaluate(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.3,
    nms_iou_threshold: float = 0.05,
    num_mc_samples: int = 1000,
) -> dict:
    """
    Unified evaluation: Precision, Recall, F1 and mAP — with NMS applied
    to predictions before all metrics.

    NMS is run per-class per-image before any TP/FP/FN counting or mAP
    accumulation, matching what happens at real inference time.

    Args:
        predictions:        (Batch, S, S, C + B*BOX_ATTRS)
        targets:            (Batch, S, S, C + B*BOX_ATTRS)
        conf_threshold:     Confidence cutoff — boxes below this are dropped
                            before NMS.
        iou_threshold:      IoU cutoff for TP matching in Precision/Recall/F1.
        nms_iou_threshold:  IoU cutoff used inside ellipse_nms_per_class.
                            Keep this >= iou_threshold (typically 0.5).
        iou_thresholds_map: IoU thresholds swept for mAP.
                            Defaults to [0.3].
                            Pass list(np.arange(0.5, 1.0, 0.05)) for COCO style.
        num_mc_samples:     Passed to monte_carlo_iou everywhere.

    Returns:
        dict with keys:
            precision, recall, f1,
            mAP
    """
    
    iou_thresholds_map = [0.3]
    batch_size = predictions.shape[0]

    # Ensure both tensors are on the same device
    device = predictions.device
    targets = targets.to(device)

    # P/R/F1 accumulators
    total_tp = total_fp = total_fn = 0

    # mAP accumulators — keyed by class, built from post-NMS detections
    detections      = {}   # cls -> [(score, img_id, box), ...]
    gt_by_image_cls = {}   # img_id -> cls -> [box, ...]

    # ------------------------------------------------------------------ #
    # 1. Per-image: extract → NMS → accumulate                           #
    # ------------------------------------------------------------------ #
    for b in range(batch_size):
        pred = predictions[b]
        tgt  = targets[b]
        gt_by_image_cls[b] = {}

        pred_boxes_t, pred_scores_t, pred_cls_t, gt_boxes_t, gt_cls_t = \
            _collect_boxes_single_image(pred, tgt, conf_threshold)

        # ---- store GT ------------------------------------------------- #
        if gt_boxes_t is not None:
            for k in range(gt_boxes_t.shape[0]):
                cls = gt_cls_t[k].item()
                gt_by_image_cls[b].setdefault(cls, []).append(gt_boxes_t[k])

        num_gt = gt_boxes_t.shape[0] if gt_boxes_t is not None else 0

        # ---- NMS on predictions --------------------------------------- #
        if pred_boxes_t is None:
            total_fn += num_gt
            continue

        kept = ellipse_nms_per_class(
            pred_boxes_t,
            pred_scores_t,
            pred_cls_t,
            iou_threshold=nms_iou_threshold,
            score_threshold=conf_threshold,
            num_mc_samples=num_mc_samples,
        )

        if kept.numel() == 0:
            total_fn += num_gt
            continue

        # Post-NMS tensors
        nms_boxes  = pred_boxes_t[kept]     # (K, 5)
        nms_scores = pred_scores_t[kept]    # (K,)
        nms_cls    = pred_cls_t[kept]       # (K,)

        # ---- accumulate for mAP --------------------------------------- #
        for k in range(nms_boxes.shape[0]):
            cls = nms_cls[k].item()
            detections.setdefault(cls, []).append((
                nms_scores[k].item(), b, nms_boxes[k]
            ))

        # ---- P/R/F1 matching for this image --------------------------- #
        if num_gt == 0:
            total_fp += nms_boxes.shape[0]
            continue

        P = nms_boxes.shape[0]
        G = gt_boxes_t.shape[0]

        # Full IoU matrix in one batched MC call
        iou_matrix = monte_carlo_iou(
            nms_boxes.unsqueeze(1).expand(P, G, 5).reshape(P * G, 5),
            gt_boxes_t.unsqueeze(0).expand(P, G, 5).reshape(P * G, 5),
            num_samples=num_mc_samples,
        ).view(P, G)

        # Zero out cross-class pairs
        cls_match  = nms_cls.unsqueeze(1) == gt_cls_t.unsqueeze(0)  # (P, G)
        iou_matrix = iou_matrix * cls_match.float()

        order      = nms_scores.argsort(descending=True)
        matched_gt = set()
        tp = fp = 0

        for pred_i in order.tolist():
            best_iou, best_gt = iou_matrix[pred_i].max(dim=0)
            if best_iou.item() >= iou_threshold and best_gt.item() not in matched_gt:
                tp += 1
                matched_gt.add(best_gt.item())
            else:
                fp += 1

        total_tp += tp
        total_fp += fp
        total_fn += G - tp

    # ------------------------------------------------------------------ #
    # 2. Precision / Recall / F1                                          #
    # ------------------------------------------------------------------ #
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ------------------------------------------------------------------ #
    # 3. mAP over post-NMS detections                                     #
    # ------------------------------------------------------------------ #
    all_classes = set(detections.keys())
    for img_gts in gt_by_image_cls.values():
        all_classes.update(img_gts.keys())

    ap_per_class_per_iou = {iou_t: {} for iou_t in iou_thresholds_map}

    for cls in all_classes:
        cls_dets = sorted(detections.get(cls, []), key=lambda x: x[0], reverse=True)
        num_gt   = sum(len(gt_by_image_cls[i].get(cls, [])) for i in gt_by_image_cls)

        if num_gt == 0 or not cls_dets:
            for iou_t in iou_thresholds_map:
                ap_per_class_per_iou[iou_t][cls] = 0.0
            continue

        for iou_t in iou_thresholds_map:
            matched = {i: [False] * len(gt_by_image_cls[i].get(cls, []))
                       for i in gt_by_image_cls}
            tp_arr = np.zeros(len(cls_dets))
            fp_arr = np.zeros(len(cls_dets))

            for det_i, (score, img_id, det_box) in enumerate(cls_dets):
                gt_list = gt_by_image_cls[img_id].get(cls, [])
                if not gt_list:
                    fp_arr[det_i] = 1
                    continue

                gt_stack  = torch.stack(gt_list)
                det_tiled = det_box.unsqueeze(0).expand(gt_stack.shape[0], -1)
                ious      = monte_carlo_iou(det_tiled, gt_stack, num_samples=num_mc_samples)
                best_iou, best_gt_i = ious.max(dim=0)

                if best_iou.item() >= iou_t and not matched[img_id][best_gt_i.item()]:
                    tp_arr[det_i] = 1
                    matched[img_id][best_gt_i.item()] = True
                else:
                    fp_arr[det_i] = 1

            tp_cum     = np.cumsum(tp_arr)
            fp_cum     = np.cumsum(fp_arr)
            recalls    = tp_cum / num_gt
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap_per_class_per_iou[iou_t][cls] = _compute_ap_101(recalls, precisions)

    ap_per_class = {
        cls: float(np.mean([ap_per_class_per_iou[iou_t].get(cls, 0.0)
                             for iou_t in iou_thresholds_map]))
        for cls in all_classes
    }

    mAP    = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0

    return {
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
        "mAP":          mAP,
    }


def plot_detection_summary(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    conf_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    iou_threshold: float = 0.1,
    num_mc_samples: int = 500,
    output_dir: str = 'detections',
):
    """
    Saves up to 3 TP, 3 FN, and 3 FP examples as individual 448x448 images,
    organised into subfolders:
        detections/
            tp/  — correct predictions
            fn/  — missed detections
            fp/  — false positives

    Each image is a full 448x448 crop with the relevant ellipse drawn on it.

    Args:
        images:            (B, C, H, W)
        targets:           (B, S, S, depth)
        predictions:       (B, S, S, depth)
        conf_threshold:    Confidence cutoff before NMS.
        nms_iou_threshold: IoU cutoff for ellipse NMS.
        iou_threshold:     IoU cutoff for TP/FP/FN classification.
        num_mc_samples:    Samples passed to monte_carlo_iou.
        output_dir:        Root folder. Subfolders tp/, fn/, fp/ created inside.
    """
    import os
    from VOC import idx_to_class

    IMG_SIZE = 448
    MAX_EACH = 5

    # ------------------------------------------------------------------ #
    # 0. Create output folders                                            #
    # ------------------------------------------------------------------ #
    tp_dir = os.path.join(output_dir, 'tp')
    fn_dir = os.path.join(output_dir, 'fn')
    fp_dir = os.path.join(output_dir, 'fp')
    for d in (tp_dir, fn_dir, fp_dir):
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Walk the batch and classify every detection / GT                 #
    # ------------------------------------------------------------------ #
    tp_examples = []   # (img, pred_box, gt_box, pred_cls, gt_cls, score)
    fn_examples = []   # (img, gt_box, gt_cls)
    fp_examples = []   # (img, pred_box, pred_cls, score)

    for b in range(images.shape[0]):
        if (len(tp_examples) >= MAX_EACH and
                len(fn_examples) >= MAX_EACH and
                len(fp_examples) >= MAX_EACH):
            break

        img  = images[b]
        pred = predictions[b]
        tgt  = targets[b]

        pred_boxes_t, pred_scores_t, pred_cls_t, gt_boxes_t, gt_cls_t = \
            _collect_boxes_single_image(pred, tgt, conf_threshold)

        # NMS
        if pred_boxes_t is not None and pred_boxes_t.shape[0] > 0:
            kept = ellipse_nms_per_class(
                pred_boxes_t, pred_scores_t, pred_cls_t,
                iou_threshold=nms_iou_threshold,
                score_threshold=conf_threshold,
                num_mc_samples=num_mc_samples,
            )
            nms_boxes  = pred_boxes_t[kept]
            nms_scores = pred_scores_t[kept]
            nms_cls    = pred_cls_t[kept]
        else:
            nms_boxes = nms_scores = nms_cls = None

        num_gt   = gt_boxes_t.shape[0] if gt_boxes_t  is not None else 0
        num_pred = nms_boxes.shape[0]  if nms_boxes   is not None else 0

        # IoU matrix
        if num_pred > 0 and num_gt > 0:
            P, G = num_pred, num_gt
            iou_matrix = monte_carlo_iou(
                nms_boxes.unsqueeze(1).expand(P, G, 5).reshape(P * G, 5),
                gt_boxes_t.unsqueeze(0).expand(P, G, 5).reshape(P * G, 5),
                num_samples=num_mc_samples,
            ).view(P, G)
            cls_match  = nms_cls.unsqueeze(1) == gt_cls_t.unsqueeze(0)
            iou_matrix = iou_matrix * cls_match.float()
        else:
            iou_matrix = None

        # Greedy matching
        matched_pred = set()
        matched_gt   = set()

        if iou_matrix is not None:
            for pred_i in nms_scores.argsort(descending=True).tolist():
                best_iou, best_gt_i = iou_matrix[pred_i].max(dim=0)
                best_gt_i = best_gt_i.item()
                if best_iou.item() >= iou_threshold and best_gt_i not in matched_gt:
                    if len(tp_examples) < MAX_EACH:
                        tp_examples.append((
                            img, nms_boxes[pred_i], gt_boxes_t[best_gt_i],
                            nms_cls[pred_i].item(), gt_cls_t[best_gt_i].item(),
                            nms_scores[pred_i].item(),
                        ))
                    matched_pred.add(pred_i)
                    matched_gt.add(best_gt_i)

        # FP
        if nms_boxes is not None:
            for pred_i in range(num_pred):
                if pred_i not in matched_pred and len(fp_examples) < MAX_EACH:
                    fp_examples.append((
                        img, nms_boxes[pred_i],
                        nms_cls[pred_i].item(), nms_scores[pred_i].item(),
                    ))

        # FN
        if gt_boxes_t is not None:
            for gt_i in range(num_gt):
                if gt_i not in matched_gt and len(fn_examples) < MAX_EACH:
                    fn_examples.append((
                        img, gt_boxes_t[gt_i], gt_cls_t[gt_i].item(),
                    ))

    # ------------------------------------------------------------------ #
    # 2. Helpers                                                          #
    # ------------------------------------------------------------------ #
    def _draw_ellipse(ax, box_px, color, linestyle='-', linewidth=2.5, label=None):
        cx, cy, w, h, sin_a = box_px.cpu().tolist()
        angle_deg = float(np.rad2deg(np.arcsin(np.clip(sin_a, -1.0, 1.0))))
        ellipse = patches.Ellipse(
            xy=(cx, cy), width=w, height=h, angle=angle_deg,
            edgecolor=color, facecolor='none',
            linewidth=linewidth, linestyle=linestyle, label=label,
        )
        ax.add_patch(ellipse)
        return cx, cy

    def _base_ax(img_tensor):
        """Create a 448x448 figure/axis with the image already drawn."""
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=IMG_SIZE // 4)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
        ax.set_xlim(0, IMG_SIZE)
        ax.set_ylim(IMG_SIZE, 0)
        ax.axis('off')
        return fig, ax

    def _save(fig, path):
        fig.savefig(path, bbox_inches='tight', pad_inches=0,
                    dpi=IMG_SIZE // 4)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # 3. Save TP images                                                   #
    # ------------------------------------------------------------------ #
    for idx, (img, pred_box, gt_box, pred_cls, gt_cls, score) in enumerate(tp_examples):
        fig, ax = _base_ax(img)

        _draw_ellipse(ax, gt_box,   color='lime',  linestyle='-',
                      label=f"GT: {idx_to_class.get(gt_cls, gt_cls)}")
        _draw_ellipse(ax, pred_box, color='green', linestyle='--',
                      label=f"Pred: {idx_to_class.get(pred_cls, pred_cls)} ({score:.2f})")

        ax.legend(loc='upper right', fontsize=7,
                  facecolor='black', labelcolor='white', framealpha=0.6)
        ax.set_title(f"TP — {idx_to_class.get(pred_cls, pred_cls)} ({score:.2f})",
                     fontsize=9, color='green', pad=2)

        _save(fig, os.path.join(tp_dir, f'tp_{idx:02d}.png'))

    # ------------------------------------------------------------------ #
    # 4. Save FN images                                                   #
    # ------------------------------------------------------------------ #
    for idx, (img, gt_box, gt_cls) in enumerate(fn_examples):
        fig, ax = _base_ax(img)

        _draw_ellipse(ax, gt_box, color='orange', linestyle='-',
                      label=f"Missed GT: {idx_to_class.get(gt_cls, gt_cls)}")

        ax.legend(loc='upper right', fontsize=7,
                  facecolor='black', labelcolor='white', framealpha=0.6)
        ax.set_title(f"FN — missed: {idx_to_class.get(gt_cls, gt_cls)}",
                     fontsize=9, color='orange', pad=2)

        _save(fig, os.path.join(fn_dir, f'fn_{idx:02d}.png'))

    # ------------------------------------------------------------------ #
    # 5. Save FP images                                                   #
    # ------------------------------------------------------------------ #
    for idx, (img, pred_box, pred_cls, score) in enumerate(fp_examples):
        fig, ax = _base_ax(img)

        _draw_ellipse(ax, pred_box, color='red', linestyle='--',
                      label=f"FP: {idx_to_class.get(pred_cls, pred_cls)} ({score:.2f})")

        ax.legend(loc='upper right', fontsize=7,
                  facecolor='black', labelcolor='white', framealpha=0.6)
        ax.set_title(f"FP — {idx_to_class.get(pred_cls, pred_cls)} ({score:.2f})",
                     fontsize=9, color='red', pad=2)

        _save(fig, os.path.join(fp_dir, f'fp_{idx:02d}.png'))

    # ------------------------------------------------------------------ #
    # 6. Summary                                                          #
    # ------------------------------------------------------------------ #
    print(f"Saved to '{output_dir}/':")
    print(f"  tp/ — {len(tp_examples)} image(s)")
    print(f"  fn/ — {len(fn_examples)} image(s)")
    print(f"  fp/ — {len(fp_examples)} image(s)")