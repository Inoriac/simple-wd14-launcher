from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import onnxruntime as ort
import pandas as pd
import numpy as np

app = FastAPI(title="WD14 Tagger API")

# 加载标签
tags_df = pd.read_csv("selected_tags.csv")
tag_names = tags_df["name"].tolist()

# 加载模型
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
print("Using providers:", session.get_providers())

def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((448, 448)) # 图像预处理
    arr = np.array(image).astype(np.float32) / 255.0    # 归一化
    arr = arr.transpose(2, 0, 1)    # CHW
    arr = np.expand_dims(arr, 0)    # NCHW
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.35):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    inputs = preprocess_image(image)
    ort_inputs = {session.get_inputs()[0].name: inputs}
    preds = session.run(None, ort_inputs)[0][0]

    results = {tag: float(score) for tag, score in zip(tag_names, preds) if score >= threshold}

    return {"tags": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

