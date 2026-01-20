from __future__ import annotations

import base64
import binascii
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import cv2
import threading

# 复用你的 Triton client + 前后处理
# 把你 call_triton_server.py 中的这些类/函数直接复制到本文件或 import：

# - PPOCRv5TritonClient（下面我会对它做一个“输入从 path 改为 ndarray”小改造）

from utils.call_triton_server import PPOCRv5TritonClient  # 你也可以直接复制代码进来避免相对导入问题
from utils.call_triton_server import get_rotate_crop_image



# ----------------------------
# 基础配置（按需改环境变量）
# ----------------------------
TRITON_URL = os.getenv("TRITON_URL", "192.168.90.156:6000")
DET_MODEL = os.getenv("DET_MODEL", "PP-OCRv5_server_det")
REC_MODEL = os.getenv("REC_MODEL", "PP-OCRv5_server_rec")
DICT_PATH = os.getenv("DICT_PATH", "static/ppocrv5_dict.txt")

MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(30 * 1024 * 1024)))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "20"))

DATA_URL_RE = re.compile(r"^data:.*?;base64,", re.IGNORECASE)


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


async def _download_url(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        content = r.content
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"Downloaded file too large: {len(content)} > {MAX_DOWNLOAD_BYTES}")
        return content


def _decode_base64_payload(b64_str: str) -> bytes:
    # 支持 data URL 前缀：data:image/jpeg;base64,xxxx
    b64_str = DATA_URL_RE.sub("", b64_str).strip()
    b64_str = re.sub(r"\s+", "", b64_str)
    try:
        raw = base64.b64decode(b64_str, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 content: {e}") from e
    if not raw:
        raise ValueError("Base64 decoded to empty bytes")
    if len(raw) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"Base64 payload too large: {len(raw)} > {MAX_DOWNLOAD_BYTES}")
    return raw


def _infer_file_type(file_type: Optional[int], file_ref: str, file_bytes: bytes) -> int:
    """
    0=PDF, 1=Image
    - 若用户显式传 fileType(0/1)则信任
    - 否则按 URL 后缀 + magic bytes 推断
    """
    if file_type in (0, 1):
        return int(file_type)

    lower = file_ref.lower()
    if lower.endswith(".pdf"):
        return 0
    if any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]):
        return 1

    # PDF magic
    if file_bytes[:4] == b"%PDF":
        return 0
    return 1


def _decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """
    支持 jpg/png/bmp/tiff/webp...（OpenCV 从内存 buffer 解码）:contentReference[oaicite:3]{index=3}
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("cv2.imdecode failed: invalid/unsupported image bytes")
    return img


def _xyxy_from_poly(poly4: np.ndarray) -> List[int]:
    xs = poly4[:, 0].astype(int)
    ys = poly4[:, 1].astype(int)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _sorted_boxes_with_scores(dt_boxes, scores):
    """
    PaddleOCR 官方同款阅读顺序排序：top->bottom, left->right，并对同一行做微调
    参考 PaddleOCR predict_system.py 的 sorted_boxes 逻辑 :contentReference[oaicite:4]{index=4}
    """
    pairs = list(zip(dt_boxes, scores))
    pairs.sort(key=lambda p: (p[0][0][1], p[0][0][0]))  # (y, x)

    for i in range(len(pairs) - 1):
        for j in range(i, -1, -1):
            if abs(pairs[j + 1][0][0][1] - pairs[j][0][0][1]) < 10 and (pairs[j + 1][0][0][0] < pairs[j][0][0][0]):
                pairs[j], pairs[j + 1] = pairs[j + 1], pairs[j]
            else:
                break

    if not pairs:
        return dt_boxes, scores
    dt_boxes_sorted, scores_sorted = zip(*pairs)
    return np.array(dt_boxes_sorted), list(scores_sorted)


# ----------------------------
# 线程安全：HTTP client 在并发下建议加锁（或每请求新建 client）
# Triton http aio 文档明确“单线程使用，否则行为未定义” :contentReference[oaicite:4]{index=4}
# ----------------------------
class PPOCRService:
    def __init__(self):
        self._lock = threading.Lock()
        self._client = PPOCRv5TritonClient(
            url=TRITON_URL,
            det_model=DET_MODEL,
            rec_model=REC_MODEL,
            dict_path=DICT_PATH,
        )

    def ocr_bgr(
        self,
        img_bgr: np.ndarray,
        textDetLimitSideLen: Optional[int] = None,
        textDetLimitType: Optional[str] = None,
        textDetThresh: Optional[float] = None,
        textDetBoxThresh: Optional[float] = None,
        textDetUnclipRatio: Optional[float] = None,
        textRecScoreThresh: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        核心 OCR（对齐你 call_triton_server.py 的流程：det->crop->rec）:contentReference[oaicite:5]{index=5}
        这里允许用入参覆盖部分 det/DBPost/rec 阈值。
        """
        # 覆盖阈值（你原始默认值在 PPOCRv5TritonClient.__init__ / DBPostProcess 初始化里）:contentReference[oaicite:6]{index=6}
        # 注意：det_resize_for_test 的 limit_side_len/limit_type 在你 infer_det 内部写死了（736/min）:contentReference[oaicite:7]{index=7}
        # 若要完全可配置，需要你把 infer_det 改为接收参数。这里给出一种“最小侵入”写法：直接改 client 的成员参数（如果你把 infer_det 改造了）。
        # 为了保持与你文件一致，我这里不强行改 infer_det 签名，只做 DBPostProcess & rec 侧过滤。
        if textDetThresh is not None:
            self._client.db_post.thresh = float(textDetThresh)
        if textDetBoxThresh is not None:
            self._client.db_post.box_thresh = float(textDetBoxThresh)
        if textDetUnclipRatio is not None:
            self._client.db_post.unclip_ratio = float(textDetUnclipRatio)

        # Triton infer（加锁避免并发踩踏）
        with self._lock:
            boxes, det_scores = self._client.infer_det(img_bgr)

            # crops = []
            # for box in boxes:
            #     crops.append(get_rotate_crop_image(img_bgr, box.astype(np.float32)))
            # rec_res = self._client.infer_rec(crops)

            boxes, det_scores = _sorted_boxes_with_scores(boxes, det_scores)
            # 然后再 crop -> rec
            crops = [get_rotate_crop_image(img_bgr, b.astype(np.float32)) for b in boxes]
            rec_res = self._client.infer_rec(crops)

        rec_texts: List[str] = []
        rec_scores: List[float] = []
        rec_polys: List[List[List[int]]] = []
        rec_boxes: List[List[int]] = []

        # 结果整理（与你原 results 列表类似，但输出更贴 PP-OCRv5 文档）:contentReference[oaicite:8]{index=8}
        for i, poly in enumerate(boxes):
            text, conf = rec_res[i] if i < len(rec_res) else ("", 0.0)
            conf = float(conf)

            if textRecScoreThresh is not None and conf < float(textRecScoreThresh):
                continue

            rec_texts.append(text)
            rec_scores.append(conf)
            poly4 = np.array(poly).reshape(4, 2)
            rec_polys.append(poly4.astype(int).tolist())
            rec_boxes.append(_xyxy_from_poly(poly4))

        pruned = {
            "text_type": "general",
            "text_det_params": {
                "limit_side_len": int(textDetLimitSideLen) if textDetLimitSideLen is not None else 736,
                "limit_type": textDetLimitType if textDetLimitType is not None else "min",
                "thresh": float(textDetThresh) if textDetThresh is not None else float(self._client.db_post.thresh),
                "box_thresh": float(textDetBoxThresh) if textDetBoxThresh is not None else float(self._client.db_post.box_thresh),
                "unclip_ratio": float(textDetUnclipRatio) if textDetUnclipRatio is not None else float(self._client.db_post.unclip_ratio),
            },
            "text_rec_score_thresh": float(textRecScoreThresh) if textRecScoreThresh is not None else 0.0,
            "dt_polys": [np.array(b).astype(int).tolist() for b in boxes],
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "rec_polys": rec_polys,
            "rec_boxes": rec_boxes,
        }
        return pruned


# 单例服务
_SERVICE = PPOCRService()


# ----------------------------
# 1) 对外主函数（业务流程入口）
# ----------------------------
async def run_ppocr(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    入参：你的 API request body dict
    出参：对齐 PP-OCRv5 文档结构（logId/errorCode/errorMsg/result.ocrResults[].prunedResult）
    """
    log_id = str(uuid.uuid4())

    try:
        file_ref = payload.get("file")
        if not isinstance(file_ref, str) or not file_ref.strip():
            return {"logId": log_id, "errorCode": 400, "errorMsg": "file is required"}

        file_ref = file_ref.strip()

        # 读 bytes（url 或 base64）
        if _is_url(file_ref):
            file_bytes = await _download_url(file_ref)
        else:
            file_bytes = _decode_base64_payload(file_ref)

        file_type = _infer_file_type(payload.get("fileType"), file_ref, file_bytes)

        # 目前你的 call_triton_server.py 仅实现“图片路径 OCR”流程 :contentReference[oaicite:9]{index=9}
        # 这里先按你的第3点“各种图片格式 + 字节流”实现 image；PDF 若要支持需要引入 PDF->image 渲染（可后续补）。
        if file_type == 0:
            return {"logId": log_id, "errorCode": 400, "errorMsg": "PDF not supported by current triton client code (image only)."}
        img_bgr = _decode_image_bytes_to_bgr(file_bytes)

        pruned = _SERVICE.ocr_bgr(
            img_bgr,
            textDetLimitSideLen=payload.get("textDetLimitSideLen"),
            textDetLimitType=payload.get("textDetLimitType"),
            textDetThresh=payload.get("textDetThresh"),
            textDetBoxThresh=payload.get("textDetBoxThresh"),
            textDetUnclipRatio=payload.get("textDetUnclipRatio"),
            textRecScoreThresh=payload.get("textRecScoreThresh"),
        )

        # visualize：你文档里说可返回可视化图（这里先不返回图；你要的话我可以补 draw+imencode+b64）
        ocr_item = {
            "prunedResult": pruned,
            "ocrImage": None,
            "docPreprocessingImage": None,
            "inputImage": None,
        }

        return {
            "logId": log_id,
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {
                "ocrResults": [ocr_item],
                "dataInfo": {"type": "image", "width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0])},
            },
        }

    except httpx.HTTPError as e:
        return {"logId": log_id, "errorCode": 502, "errorMsg": f"Download failed: {e}"}
    except Exception as e:
        return {"logId": log_id, "errorCode": 500, "errorMsg": str(e)}
