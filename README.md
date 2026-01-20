## 部署
```cmd
conda create -n docling_env python=3.10
conda create -p /root/autodl-tmp/miniconda3/docling_env python=3.12
conda create -p /root/autodl-tmp/miniconda3/paddleocr python=3.12
conda activate docling_env
docling-tools models download
python anylaze_output.py --input ./static/文档解析_输出结果对比_20251230_151152.xlsx --output report.xlsx
```

## 批量解析文档
`run_batch`
Docling-default + PP-OCRv4
### 本地docling运行方法
`local_function`

## 图片理解用法
`picture_description`


## VLM文档解析
`docling_with_vlm`


```batch
export DOCLING_SERVE_ARTIFACTS_PATH="/a/domains/docling/docling_models/"
```