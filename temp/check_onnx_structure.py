import onnxruntime as ort

# session = ort.InferenceSession("C:\\Users\zmn\Desktop\\fsdownload\model_onnx\PP-OCRv5_server_rec.onnx")
session = ort.InferenceSession("C:\\Users\zmn\.cache\modelscope\hub\models\RapidAI\RapidOCR\onnx\PP-OCRv5\\rec\ch_PP-OCRv5_rec_server_infer.onnx")

print("模型输入信息:")
for inp in session.get_inputs():
    print(f" • Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

print("模型输出信息:")
for out in session.get_outputs():
    print(f" • Name: {out.name}, Shape: {out.shape}, Type: {out.type}")
