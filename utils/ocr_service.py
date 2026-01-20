# ocr_service.py
import os, uuid, asyncio
from typing import Any, Dict, Optional
import httpx
import numpy as np
from utils.call_triton_server import PPOCRv5TritonClient
from utils.ocr_tools import is_url, infer_file_type, decode_base64_payload, download_url, decode_image_bytes_to_bgr

# TRITON_URL = os.getenv("TRITON_URL", "192.168.90.156:6000")
TRITON_URL = os.getenv("TRITON_URL", "0.0.0.0:6000")
DET_MODEL = os.getenv("DET_MODEL", "PP-OCRv5_server_det")
REC_MODEL = os.getenv("REC_MODEL", "PP-OCRv5_server_rec")
DICT_PATH = os.getenv("DICT_PATH", "static/ppocrv5_dict.txt")  # 你已经改成项目根目录拼接的版本即可

POOL_SIZE = int(os.getenv("TRITON_CLIENT_POOL_SIZE", "2"))
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(30 * 1024 * 1024)))

_http: Optional[httpx.AsyncClient] = None
_pool: Optional[asyncio.Queue] = None

async def startup():
    global _http, _pool
    _http = httpx.AsyncClient(timeout=20.0, follow_redirects=True)
    _pool = asyncio.Queue(maxsize=POOL_SIZE)
    for _ in range(POOL_SIZE):
        cli = PPOCRv5TritonClient(
            url=TRITON_URL,
            det_model=DET_MODEL,
            rec_model=REC_MODEL,
            dict_path=DICT_PATH,
        )
        _pool.put_nowait(cli)

async def shutdown():
    global _http
    if _http:
        await _http.aclose()






async def run_ppocr(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    对外主函数（业务流程入口）
    - 支持 URL / Base64 输入（图片）
    - 输出结构与官方服务一致：logId/errorCode/errorMsg/result.ocrResults[].prunedResult + dataInfo
    """
    log_id = str(uuid.uuid4())

    try:
        # ---- 防御：确保已初始化（即使你忘了挂 lifespan 也不至于 None）
        global _http, _pool
        if _http is None or _pool is None:
            await startup()

        file_ref = payload.get("file")
        if not isinstance(file_ref, str) or not file_ref.strip():
            return {"logId": log_id, "errorCode": 400, "errorMsg": "file is required"}

        file_ref = file_ref.strip()

        # ---- 1) 获取字节流（URL / Base64）
        if is_url(file_ref):
            file_bytes = await download_url(_http, file_ref, max_bytes=MAX_DOWNLOAD_BYTES)
        else:
            file_bytes = decode_base64_payload(file_ref)

        # ---- 2) 推断 fileType（0=PDF, 1=Image）
        file_type = infer_file_type(payload.get("fileType"), file_ref, file_bytes)

        # 目前你的 triton client 只做图像 det+rec；PDF 先明确报错（格式仍保持官方风格）
        if file_type == 0:
            return {"logId": log_id, "errorCode": 400, "errorMsg": "PDF not supported by current image-only pipeline"}

        # ---- 3) bytes -> image（用于 dataInfo 宽高 + 送入 OCR）
        img_bgr = decode_image_bytes_to_bgr(file_bytes)

        # ---- 4) 从池里借一个 Triton client（避免每次新建连接 & 避免线程不安全）
        cli = await _pool.get()
        try:
            rec_score_thresh = float(payload.get("textRecScoreThresh") or 0.0)

            # 约定：cli.ocr_bytes 返回 dict，包含 boxes/rec_texts/rec_scores/rec_polys/rec_boxes 等
            # 你需要在 call_triton_server.py 的 ocr_bgr/ocr_bytes 按这个结构返回
            out = cli.ocr_bytes(
                file_bytes,
                sort_reading_order=True,
                rec_score_thresh=rec_score_thresh,
            )
        finally:
            _pool.put_nowait(cli)

        # ---- 5) 组装 prunedResult（对齐官方字段）
        # dt_polys：检测框（可与 rec_polys 一致，但官方会回传检测框）
        dt_polys = [np.array(b).astype(int).tolist() for b in out.get("boxes", [])]

        pruned = {
            "model_settings": {
                "use_doc_preprocessor": True,
                "use_textline_orientation": True,
            },
            "doc_preprocessor_res": {
                "model_settings": {
                    "use_doc_orientation_classify": True,
                    "use_doc_unwarping": True,
                },
                "angle": int(out.get("angle", 0)),
            },
            "dt_polys": dt_polys,
            "text_det_params": {
                # 这些是“回显字段”。如果你的 det 参数在代码里固定，就按固定值回填；
                # 如果你已经做成可配置，就从 payload 里读并传给 det，再在这里回显同值。
                "limit_side_len": int(payload.get("textDetLimitSideLen") or 64),
                "limit_type": payload.get("textDetLimitType") or "min",
                "thresh": float(payload.get("textDetThresh") or 0.3),
                "max_side_limit": 4000,
                "box_thresh": float(payload.get("textDetBoxThresh") or 0.6),
                "unclip_ratio": float(payload.get("textDetUnclipRatio") or 1.5),
            },
            "text_type": "general",
            "textline_orientation_angles": out.get("textline_orientation_angles") or [0] * len(out.get("rec_texts", [])),
            "text_rec_score_thresh": float(payload.get("textRecScoreThresh") or 0.0),
            "return_word_box": False,
            "rec_texts": out.get("rec_texts", []),
            "rec_scores": out.get("rec_scores", []),
            "rec_polys": out.get("rec_polys", []),
            "rec_boxes": out.get("rec_boxes", []),
        }

        ocr_item = {
            "prunedResult": pruned,
            # 你当前没做可视化/中间图，先保持字段存在并返回 null
            "ocrImage": None,
            "docPreprocessingImage": None,
            "inputImage": None,
        }

        # ---- 6) 最终对齐官方 response
        return {
            "logId": log_id,
            "result": {
                "ocrResults": [ocr_item],
                "dataInfo": {
                    "width": int(img_bgr.shape[1]),
                    "height": int(img_bgr.shape[0]),
                    "type": "image",
                },
            },
            "errorCode": 0,
            "errorMsg": "Success",
        }

    except httpx.HTTPError as e:
        return {"logId": log_id, "errorCode": 502, "errorMsg": f"Download failed: {e}"}
    except Exception as e:
        return {"logId": log_id, "errorCode": 500, "errorMsg": str(e)}

