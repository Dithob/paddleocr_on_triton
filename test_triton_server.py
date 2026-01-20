import argparse
import numpy as np
import cv2

import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def get_io_names_and_shapes(client: httpclient.InferenceServerClient, model_name: str, model_version: str = ""):
    """
    从 Triton 读取 model metadata/config，拿到 input/output 的名字、dtype、shape。
    """
    meta = client.get_model_metadata(model_name=model_name, model_version=model_version)
    cfg = client.get_model_config(model_name=model_name, model_version=model_version)

    # metadata 里有 inputs/outputs 的 name + datatype + shape
    inputs = meta["inputs"]
    outputs = meta["outputs"]

    # config 里也有更完整信息（max_batch_size、dims 等）
    # 这里优先用 metadata 的 shape，后续你也可以改成读取 cfg["input"][0]["dims"]
    return inputs, outputs, cfg


def simple_image_to_nchw_float32(img_bgr: np.ndarray, chw_shape):
    """
    一个“通用预处理”占位实现：resize -> BGR2RGB -> 归一化到 [0,1] -> NCHW float32
    注意：OCR 的真实预处理（保持比例、mean/std、padding 等）需与你导出模型时一致。
    """
    # chw_shape 形如 [3, H, W] 或 [-1, 3, H, W] 取后两维
    if len(chw_shape) == 4:
        _, c, h, w = chw_shape
    elif len(chw_shape) == 3:
        c, h, w = chw_shape
    else:
        raise ValueError(f"Unexpected input shape: {chw_shape}")

    assert c == 3, f"Expect 3 channels, got {c}"
    resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0)   # -> NCHW
    return x


def infer_one_model_http(url: str, model_name: str, image_path: str, model_version: str = "", verbose: bool = False):
    client = httpclient.InferenceServerClient(url=url, verbose=verbose)

    # 1) health check
    if not client.is_server_live():
        raise RuntimeError("Triton server is not live")
    if not client.is_server_ready():
        raise RuntimeError("Triton server is not ready")
    if not client.is_model_ready(model_name):
        raise RuntimeError(f"Model not ready: {model_name}")

    # 2) read metadata/config to get I/O
    inputs, outputs, cfg = get_io_names_and_shapes(client, model_name, model_version)
    in0 = inputs[0]
    input_name = in0["name"]
    input_dtype = in0["datatype"]
    input_shape = in0["shape"]  # 例如 [1,3,640,640] 或 [-1,3,-1,-1]

    output_names = [o["name"] for o in outputs]

    print(f"\n[Model] {model_name}")
    print(f"  Input : name={input_name}, dtype={input_dtype}, shape={input_shape}")
    print(f"  Output: {output_names}")
    # print("  Config:", cfg)  # 需要时打开

    # 3) prepare input
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # 对 dynamic shape 做一个可运行的兜底：把 -1 替换为 1/640/640（你可按模型实际改）
    shape_for_pre = list(input_shape)
    # 常见：[-1,3,-1,-1] 或 [1,3,-1,-1]
    if len(shape_for_pre) == 4:
        if shape_for_pre[0] == -1:
            shape_for_pre[0] = 1
        if shape_for_pre[2] == -1:
            shape_for_pre[2] = 640
        if shape_for_pre[3] == -1:
            shape_for_pre[3] = 640

    x = simple_image_to_nchw_float32(img, shape_for_pre).astype(np.float32)

    # Triton InferInput：name/shape/datatype
    infer_in = httpclient.InferInput(input_name, x.shape, input_dtype)
    infer_in.set_data_from_numpy(x)

    # 要哪些输出：全要
    infer_outs = [httpclient.InferRequestedOutput(n) for n in output_names]

    # 4) infer
    resp = client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=[infer_in],
        outputs=infer_outs,
    )

    # 5) fetch outputs
    results = {}
    for n in output_names:
        arr = resp.as_numpy(n)
        results[n] = arr
        print(f"  -> output {n}: shape={None if arr is None else arr.shape}, dtype={None if arr is None else arr.dtype}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("--url", default="127.0.0.1:6000", help="Triton HTTP url host:port")
    ap.add_argument("--url", default="192.168.90.156:6000", help="Triton HTTP url host:port")
    ap.add_argument("--model", required=True, help="Model name in Triton, e.g. PP-OCRv5_server_det")
    ap.add_argument("--image", required=True, help="Path to image")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    infer_one_model_http(args.url, args.model, args.image, verbose=args.verbose)
