import os
import cv2
import math
import numpy as np

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

# DBPostProcess 依赖
from shapely.geometry import Polygon
from utils.ocr_tools import decode_image_bytes_to_bgr, sorted_boxes
import pyclipper


# -------------------------
# 1) CTC decode (简化版，对齐常见 PaddleOCR/RapidOCR 用法)
# -------------------------
class CTCLabelDecode:
    def __init__(self, character_path: str):
        # dict file: one token per line
        with open(character_path, "rb") as f:
            lines = f.readlines()
        chars = []
        for line in lines:
            chars.append(line.decode("utf-8").strip("\n").strip("\r\n"))

        # PaddleOCR/RapidOCR 常见：index0=blank，末尾加一个空格/space
        self.character = ["blank"] + chars + [" "]

    @staticmethod
    def get_ignored_tokens():
        return [0]  # blank id=0

    def decode(self, preds: np.ndarray):
        """
        preds: [B, T, C] logits/prob
        """
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        results = []
        ignored = set(self.get_ignored_tokens())
        for b in range(preds_idx.shape[0]):
            seq = preds_idx[b]
            prob = preds_prob[b]
            # remove duplicate + remove blank
            selection = np.ones_like(seq, dtype=bool)
            selection[1:] = seq[1:] != seq[:-1]
            for t in ignored:
                selection &= (seq != t)

            chosen = seq[selection]
            conf = prob[selection]
            if conf.size == 0:
                conf = np.array([0.0], dtype=np.float32)

            text = "".join([self.character[i] for i in chosen])
            score = float(conf.mean())
            results.append((text, score))
        return results


# -------------------------
# 2) det preprocess: DetResizeForTest + NormalizeImage + ToCHW
#    (按 RapidOCR/PaddleOCR 常见参数：limit_side_len=736, limit_type=min；mean/std/scale 同 RapidOCR)
# -------------------------
def det_resize_for_test(img_bgr: np.ndarray, limit_side_len=736, limit_type="min"):
    """
    return: resized_img, (src_h, src_w, ratio_h, ratio_w)
    """
    src_h, src_w = img_bgr.shape[:2]

    if limit_type == "min":
        ratio = float(limit_side_len) / min(src_h, src_w)
    else:  # "max"
        ratio = float(limit_side_len) / max(src_h, src_w)

    resize_h = int(round(src_h * ratio))
    resize_w = int(round(src_w * ratio))

    # DBNet/PaddleOCR 通常要求尺寸为 32 的倍数
    resize_h = max(32, int(math.ceil(resize_h / 32) * 32))
    resize_w = max(32, int(math.ceil(resize_w / 32) * 32))

    resized = cv2.resize(img_bgr, (resize_w, resize_h))
    ratio_h = resize_h / float(src_h)
    ratio_w = resize_w / float(src_w)
    return resized, (src_h, src_w, ratio_h, ratio_w)


def normalize_hwc_to_chw_rgb(img_bgr: np.ndarray):
    # BGR -> RGB
    img = img_bgr[:, :, ::-1].astype("float32")
    # scale + mean/std (RapidOCR 配置中为 scale=1/255, mean/std 如下)  :contentReference[oaicite:5]{index=5}
    img = img * (1.0 / 255.0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    return img


# -------------------------
# 3) DBPostProcess（按 PaddleOCR 的核心逻辑实现） :contentReference[oaicite:6]{index=6}
# -------------------------
class DBPostProcess:
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=1.6, score_mode="fast"):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        assert score_mode in ["fast", "slow"]
        self.score_mode = score_mode

    def box_score_fast(self, bitmap, box):
        h, w = bitmap.shape[:2]
        box = box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / (poly.length + 1e-6)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        # 4 points order
        idx1, idx2, idx3, idx4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            idx1, idx4 = 0, 1
        else:
            idx1, idx4 = 1, 0
        if points[3][1] > points[2][1]:
            idx2, idx3 = 2, 3
        else:
            idx2, idx3 = 3, 2
        box = [points[idx1], points[idx2], points[idx3], points[idx4]]
        return box, min(bounding_box[1])

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        pred: [H, W] prob map
        bitmap: [H, W] 0/1
        """
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[0] if len(outs) == 2 else outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []
        for i in range(num_contours):
            contour = contours[i]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if score < self.box_thresh:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            # map back to original image size
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)

        return np.array(boxes, dtype=np.int16), scores

    def __call__(self, pred_map: np.ndarray, shape):
        """
        pred_map: [H, W] or [1, H, W]
        shape: (src_h, src_w, ratio_h, ratio_w) from det resize
        """
        if pred_map.ndim == 3:
            pred_map = pred_map[0]
        src_h, src_w, _, _ = shape
        bitmap = (pred_map > self.thresh).astype(np.uint8)
        boxes, scores = self.boxes_from_bitmap(pred_map, bitmap, src_w, src_h)
        return boxes, scores


# -------------------------
# 4) crop + rectify (透视矫正)
# -------------------------
def order_points_clockwise(pts):
    pts = np.array(pts).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def get_rotate_crop_image(img, points):
    points = order_points_clockwise(points)
    (tl, tr, br, bl) = points
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    maxW = max(1, maxW)
    maxH = max(1, maxH)

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 如果竖得太厉害，旋转一下（常见处理）
    if warped.shape[0] / float(warped.shape[1] + 1e-6) >= 1.5:
        warped = np.rot90(warped)
    return warped


# -------------------------
# 5) rec preprocess：按 PaddleOCR/RapidOCR 常见实现 resize_norm_img（48高，宽按比例+pad到batch最大宽） :contentReference[oaicite:7]{index=7}
# -------------------------
def rec_resize_norm_img(img_bgr: np.ndarray, imgH=48, max_wh_ratio=10.0):
    imgC = 3
    imgW = int(imgH * max_wh_ratio)

    img_rgb = img_bgr[:, :, ::-1]
    h, w = img_rgb.shape[:2]
    ratio = w / float(h + 1e-6)

    resized_w = imgW if math.ceil(imgH * ratio) > imgW else int(math.ceil(imgH * ratio))
    resized = cv2.resize(img_rgb, (resized_w, imgH))
    resized = resized.astype("float32").transpose(2, 0, 1) / 255.0
    resized -= 0.5
    resized /= 0.5

    padding = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding[:, :, 0:resized_w] = resized
    return padding, resized_w


# -------------------------
# 6) Triton caller
# -------------------------
class PPOCRv5TritonClient:
    def __init__(self, url="localhost:6000",
                 det_model="PP-OCRv5_server_det",
                 rec_model="PP-OCRv5_server_rec",
                 dict_path="ppocrv5_dict.txt",
                 timeout=30.0):
        self.client = httpclient.InferenceServerClient(
            url=url,
            connection_timeout=timeout,
            network_timeout=timeout,
        )
        self.det_model = det_model
        self.rec_model = rec_model
        self.decoder = CTCLabelDecode(dict_path)
        self.db_post = DBPostProcess(thresh=0.3, box_thresh=0.5, unclip_ratio=1.6)

    def infer_det(self, img_bgr: np.ndarray):
        resized, shape = det_resize_for_test(img_bgr, limit_side_len=736, limit_type="min")
        chw = normalize_hwc_to_chw_rgb(resized)
        inp = chw[np.newaxis, ...].astype(np.float32)  # [1,3,H,W]

        inputs = [httpclient.InferInput("x", inp.shape, "FP32")]
        inputs[0].set_data_from_numpy(inp, binary_data=True)

        outputs = [httpclient.InferRequestedOutput("fetch_name_0", binary_data=True)]
        resp = self.client.infer(self.det_model, inputs=inputs, outputs=outputs)

        pred = resp.as_numpy("fetch_name_0")
        # 兼容：[B,1,H,W] / [B,H,W] / [B,H,W,1]
        pred = np.array(pred)
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0, :, :]
        elif pred.ndim == 4 and pred.shape[-1] == 1:
            pred = pred[..., 0]
        # 取 batch0
        pred_map = pred[0]
        boxes, scores = self.db_post(pred_map, shape)
        return boxes, scores

    # def infer_rec(self, crops_bgr):
    #     if len(crops_bgr) == 0:
    #         return []
    #
    #     # batch内 pad 到相同宽：先算 max_wh_ratio
    #     wh_ratios = [c.shape[1] / float(c.shape[0] + 1e-6) for c in crops_bgr]
    #     max_wh_ratio = max(wh_ratios + [10.0])  # 给个基础值避免过小
    #
    #     batch_imgs = []
    #     for crop in crops_bgr:
    #         norm_img, _ = rec_resize_norm_img(crop, imgH=48, max_wh_ratio=max_wh_ratio)
    #         batch_imgs.append(norm_img[np.newaxis, ...])
    #     batch = np.concatenate(batch_imgs, axis=0).astype(np.float32)  # [B,3,48,W]
    #
    #     inputs = [httpclient.InferInput("x", batch.shape, "FP32")]
    #     inputs[0].set_data_from_numpy(batch, binary_data=True)
    #
    #     outputs = [httpclient.InferRequestedOutput("fetch_name_0", binary_data=True)]
    #     resp = self.client.infer(self.rec_model, inputs=inputs, outputs=outputs)
    #
    #     logits = resp.as_numpy("fetch_name_0")
    #     # 预期类似 [B,T,18385]
    #     logits = np.array(logits)
    #     return self.decoder.decode(logits)


    def infer_rec(self, crops_bgr):
        if len(crops_bgr) == 0:
            return []

        all_results = []
        max_bs = 8  # 对齐 config.pbtxt 里的 max_batch_size

        for start in range(0, len(crops_bgr), max_bs):
            chunk = crops_bgr[start:start + max_bs]

            # === 你原来的“算 max_wh_ratio + pad 到同宽 + 拼 batch”的逻辑照旧 ===
            wh_ratios = [c.shape[1] / float(c.shape[0] + 1e-6) for c in chunk]
            max_wh_ratio = max(wh_ratios + [10.0])

            batch_imgs = []
            for crop in chunk:
                norm_img, _ = rec_resize_norm_img(crop, imgH=48, max_wh_ratio=max_wh_ratio)
                batch_imgs.append(norm_img[np.newaxis, ...])

            batch = np.concatenate(batch_imgs, axis=0).astype(np.float32)  # [B,3,48,W]

            inputs = [httpclient.InferInput("x", batch.shape, "FP32")]
            inputs[0].set_data_from_numpy(batch, binary_data=True)

            outputs = [httpclient.InferRequestedOutput("fetch_name_0", binary_data=True)]
            resp = self.client.infer(self.rec_model, inputs=inputs, outputs=outputs)

            logits = resp.as_numpy("fetch_name_0")
            logits = np.array(logits)

            all_results.extend(self.decoder.decode(logits))

        return all_results


    def ocr(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        boxes, scores = self.infer_det(img)

        # crop
        crops = []
        for box in boxes:
            crop = get_rotate_crop_image(img, box.astype(np.float32))
            crops.append(crop)

        rec_res = self.infer_rec(crops)

        results = []
        for i, box in enumerate(boxes):
            text, conf = rec_res[i] if i < len(rec_res) else ("", 0.0)
            results.append({
                "box": box.tolist(),
                "text": text,
                "score": float(conf),
            })
        return results

    def ocr_bgr(self, img_bgr, sort_reading_order=True, rec_score_thresh=0.0):
        boxes, det_scores = self.infer_det(img_bgr)

        if sort_reading_order:
            boxes, det_scores = sorted_boxes(boxes, det_scores)  # 你已有

        crops = [get_rotate_crop_image(img_bgr, box.astype(np.float32)) for box in boxes]
        rec_res = self.infer_rec(crops)  # [(text, score), ...]

        rec_texts, rec_scores, rec_polys, rec_boxes = [], [], [], []
        for i, box in enumerate(boxes):
            text, score = rec_res[i] if i < len(rec_res) else ("", 0.0)
            score = float(score)
            if score < rec_score_thresh:
                continue
            rec_texts.append(text)
            rec_scores.append(score)
            poly4 = np.array(box).reshape(4, 2).astype(int)
            rec_polys.append(poly4.tolist())
            xs, ys = poly4[:, 0], poly4[:, 1]
            rec_boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

        return {
            "boxes": boxes,  # np.ndarray (N,4,2)
            "det_scores": det_scores,  # list[float]
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "rec_polys": rec_polys,
            "rec_boxes": rec_boxes,
            "textline_orientation_angles": [0] * len(rec_texts),  # 你当前没做该模块，先填0保持结构
            "angle": 0,  # doc_preprocessor_res angle（没做旋转就填0）
        }

    def ocr_bytes(self, image_bytes: bytes, **kwargs):
        img_bgr = decode_image_bytes_to_bgr(image_bytes)
        return self.ocr_bgr(img_bgr, **kwargs)

if __name__ == "__main__":
    # Triton HTTP: http://localhost:6000
    # 确保容器启动时做了 -p 6000:6000 映射
    client = PPOCRv5TritonClient(
        # url="http://192.168.90.156:6000",
        url="192.168.90.156:6000",
        det_model="PP-OCRv5_server_det",
        rec_model="PP-OCRv5_server_rec",
        dict_path="../static/ppocrv5_dict.txt",
    )

    res = client.ocr("../static/images/image_0000.jpg")
    for r in res:
        print(r["text"], r["score"], r["box"])
