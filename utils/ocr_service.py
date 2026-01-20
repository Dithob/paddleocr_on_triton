# ocr_service.py
import os, uuid, asyncio
from typing import Any, Dict, Optional
import httpx
from call_triton_server import PPOCRv5TritonClient
from ocr_tools import is_url, infer_file_type, decode_base64_payload, download_url

TRITON_URL = os.getenv("TRITON_URL", "192.168.90.156:6000")
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
    log_id = str(uuid.uuid4())
    try:
        file_ref = payload.get("file")
        if not isinstance(file_ref, str) or not file_ref.strip():
            return {"logId": log_id, "errorCode": 400, "errorMsg": "file is required"}

        file_ref = file_ref.strip()

        # 取 bytes：URL / base64
        if is_url(file_ref):
            file_bytes = await download_url(_http, file_ref, max_bytes=MAX_DOWNLOAD_BYTES)
        else:
            file_bytes = decode_base64_payload(file_ref)

        file_type = infer_file_type(payload.get("fileType"), file_ref, file_bytes)
        if file_type == 0:
            return {"logId": log_id, "errorCode": 400, "errorMsg": "PDF not supported in current pipeline"}

        # 借 client（复用连接 & 线程安全）
        cli = await _pool.get()
        try:
            rec_score_thresh = float(payload.get("textRecScoreThresh") or 0.0)
            results = cli.ocr_bytes(file_bytes, sort_reading_order=True, rec_score_thresh=rec_score_thresh)
        finally:
            _pool.put_nowait(cli)

        rec_texts = [r["text"] for r in results]
        rec_scores = [r["score"] for r in results]
        rec_polys  = [r["box"] for r in results]

        return {
            "logId": log_id,
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {
                "ocrResults": [{
                    "prunedResult": {
                        "rec_texts": rec_texts,
                        "rec_scores": rec_scores,
                        "rec_polys": rec_polys,
                    },
                    "ocrImage": None,
                    "docPreprocessingImage": None,
                    "inputImage": None,
                }],
            }
        }
    except Exception as e:
        return {"logId": log_id, "errorCode": 500, "errorMsg": str(e)}
