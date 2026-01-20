from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from utils.ocr_service import run_ppocr, startup, shutdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化：创建全局 httpx.AsyncClient + Triton client pool
    await startup()
    yield
    # 关闭：释放 httpx 连接池等资源
    await shutdown()


app = FastAPI(
    lifespan=lifespan,
    title="PP-OCRv5 Triton Service",
    version="1.0.0",
    description="PP-OCRv5 det+rec via Triton. Accepts URL or Base64 for images.",
)


class OCRRequest(BaseModel):
    file: str = Field(..., description="服务器可访问的图像URL 或 图像内容Base64（支持data URL前缀）")
    fileType: Optional[int] = Field(None, description="0=PDF,1=Image；缺省则自动推断")

    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useTextlineOrientation: Optional[bool] = None

    textDetLimitSideLen: Optional[int] = None
    textDetLimitType: Optional[str] = None
    textDetThresh: Optional[float] = None
    textDetBoxThresh: Optional[float] = None
    textDetUnclipRatio: Optional[float] = None
    textRecScoreThresh: Optional[float] = None

    visualize: Optional[bool] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "file": "https://img5.xiujiadian.com/img/prod/ServWorkDelivery/workPhoto/202507/20250722224634_931112.jpg",
                "fileType": 1,
                "visualize": False
            }
        }
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/v1/ocr")
async def ocr(req: OCRRequest) -> Dict[str, Any]:
    return await run_ppocr(req.model_dump(exclude_none=True))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6060)
    # uvicorn.run(app, host="0.0.0.0", port=8080)
