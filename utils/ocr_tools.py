# ocr_tools.py
import base64, binascii, re
import numpy as np
import cv2
import httpx
from typing import Optional

DATA_URL_RE = re.compile(r"^data:.*?;base64,", re.IGNORECASE)

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def decode_base64_payload(b64_str: str) -> bytes:
    b64_str = DATA_URL_RE.sub("", b64_str).strip()
    b64_str = re.sub(r"\s+", "", b64_str)
    try:
        return base64.b64decode(b64_str, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64: {e}") from e

def infer_file_type(file_type: Optional[int], file_ref: str, file_bytes: bytes) -> int:
    if file_type in (0, 1):
        return int(file_type)
    low = file_ref.lower()
    if low.endswith(".pdf") or file_bytes[:4] == b"%PDF":
        return 0
    return 1

def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("cv2.imdecode failed")
    return img

def sorted_boxes(dt_boxes, scores):
    # PaddleOCR 阅读顺序：top->bottom, left->right（同一行微调）
    pairs = list(zip(dt_boxes, scores))
    pairs.sort(key=lambda p: (p[0][0][1], p[0][0][0]))
    for i in range(len(pairs) - 1):
        for j in range(i, -1, -1):
            if abs(pairs[j + 1][0][0][1] - pairs[j][0][0][1]) < 10 and (pairs[j + 1][0][0][0] < pairs[j][0][0][0]):
                pairs[j], pairs[j + 1] = pairs[j + 1], pairs[j]
            else:
                break
    if not pairs:
        return dt_boxes, scores
    b, s = zip(*pairs)
    return np.array(b), list(s)

async def download_url(client: httpx.AsyncClient, url: str, max_bytes: int = 30 * 1024 * 1024) -> bytes:
    r = await client.get(url)
    r.raise_for_status()
    content = r.content
    if len(content) > max_bytes:
        raise ValueError(f"Downloaded too large: {len(content)}")
    return content
