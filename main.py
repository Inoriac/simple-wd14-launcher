from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import onnxruntime as ort
import pandas as pd
import numpy as np
import cv2

app = FastAPI(title="WD14 Tagger API")

# 加载标签 - 按照参考代码的正确方式
try:
    tags_df = pd.read_csv("selected_tags.csv")

    # 调试：打印CSV文件信息
    print(f"CSV columns: {tags_df.columns.tolist()}")
    print(f"CSV shape: {tags_df.shape}")
    print(f"Category column unique values: {tags_df['category'].unique()}")
    print(f"Category column data type: {tags_df['category'].dtype}")

    # 尝试不同的category匹配方式
    # 方法1：字符串匹配
    rating_tags = tags_df[tags_df["category"] == "9"]["name"].tolist()
    general_tags = tags_df[tags_df["category"] == "0"]["name"].tolist()
    character_tags = tags_df[tags_df["category"] == "4"]["name"].tolist()

    # 如果上面没有找到，尝试数字匹配
    if len(rating_tags) == 0 and len(general_tags) == 0 and len(character_tags) == 0:
        print("Trying numeric category matching...")
        rating_tags = tags_df[tags_df["category"] == 9]["name"].tolist()
        general_tags = tags_df[tags_df["category"] == 0]["name"].tolist()
        character_tags = tags_df[tags_df["category"] == 4]["name"].tolist()

    # 如果还是没有找到，尝试浮点数匹配
    if len(rating_tags) == 0 and len(general_tags) == 0 and len(character_tags) == 0:
        print("Trying float category matching...")
        rating_tags = tags_df[tags_df["category"] == 9.0]["name"].tolist()
        general_tags = tags_df[tags_df["category"] == 0.0]["name"].tolist()
        character_tags = tags_df[tags_df["category"] == 4.0]["name"].tolist()

    # 合并所有标签，按照模型输出顺序：rating + general + character
    all_tag_names = rating_tags + general_tags + character_tags

    print(
        f"Loaded {len(rating_tags)} rating tags, {len(general_tags)} general tags, {len(character_tags)} character tags")
    print(f"Total tags: {len(all_tag_names)}")

    if len(all_tag_names) == 0:
        print("ERROR: No tags loaded! Please check CSV file format.")
        print("Expected CSV format: tag_id,name,category,count")
        print("Category should be: 0=general, 4=character, 9=rating")
        raise Exception("No tags loaded from CSV file")

except Exception as e:
    print(f"Error loading selected_tags.csv: {e}")
    raise

# 加载模型 - 优先使用CPU避免CUDA配置问题
try:
    session = ort.InferenceSession(
        "model.onnx",
        providers=['CPUExecutionProvider']  # 只使用CPU，避免CUDA配置问题
    )
    print("Using providers:", session.get_providers())
except Exception as e:
    print(f"Error loading model: {e}")
    raise


def preprocess_image(image: Image.Image):
    """
    正确的WD14图片预处理流程
    """
    # 转换为RGB（确保格式正确）
    image = image.convert("RGB")

    # 转换为numpy数组
    image = np.array(image)
    print(f"Original image shape: {image.shape}, dtype: {image.dtype}")

    # 关键步骤1：RGB -> BGR 转换
    image = image[:, :, ::-1]  # RGB->BGR
    print(f"After BGR conversion: {image.shape}")

    # 关键步骤2：pad到正方形，保持比例
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    print(f"Padding: size={size}, pad_x={pad_x}, pad_y={pad_y}, pad_l={pad_l}, pad_t={pad_t}")

    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
                   mode="constant", constant_values=255)
    print(f"After padding: {image.shape}")

    # 关键步骤3：resize到448x448
    if size > 448:
        image = cv2.resize(image, (448, 448), cv2.INTER_AREA)
    else:
        image = cv2.resize(image, (448, 448), cv2.INTER_LINEAR)
    print(f"After resize: {image.shape}")

    # 转换为float32（不归一化！）
    image = image.astype(np.float32)
    print(f"After type conversion: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")

    # 添加batch维度
    image = np.expand_dims(image, 0)  # (1, 448, 448, 3)
    print(f"Final shape: {image.shape}")

    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.35):
    try:
        # 读取图片
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        print(f"Uploaded file: {file.filename}, size: {len(image_bytes)} bytes")
        print(f"Image mode: {image.mode}, size: {image.size}")

        # 预处理图片
        inputs = preprocess_image(image)

        # 获取模型输入信息
        input_name = session.get_inputs()[0].name
        print(f"Model input name: {input_name}, shape: {session.get_inputs()[0].shape}")
        print(f"Preprocessed input shape: {inputs.shape}")

        # 调试：检查预处理后的数据
        print(f"Input data range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"Input data mean: {inputs.mean():.3f}")
        print(f"Input data sample (first 5x5 pixels):")
        print(inputs[0, :5, :5, 0])  # 显示第一个通道的前5x5像素

        # 运行推理
        ort_inputs = {input_name: inputs}
        preds = session.run(None, ort_inputs)[0][0]

        # 按照参考代码的方式处理结果
        results = {}

        # 处理general标签（从第5个开始，跳过前4个rating）
        for i, score in enumerate(preds[4:]):
            if i < len(general_tags) and score >= threshold:
                tag_name = general_tags[i]
                results[tag_name] = float(score)
            elif i >= len(general_tags) and score >= threshold:
                # 处理character标签
                char_index = i - len(general_tags)
                if char_index < len(character_tags):
                    tag_name = character_tags[char_index]
                    results[tag_name] = float(score)

        # 处理rating标签（前4个，使用argmax选择最高的）
        if len(preds) >= 4:
            ratings_probs = preds[:4]
            rating_index = ratings_probs.argmax()
            if rating_index < len(rating_tags):
                found_rating = rating_tags[rating_index]
                results[found_rating] = float(ratings_probs[rating_index])

        # 打印结果
        print(f"Found {len(results)} tags above threshold {threshold}")
        for tag, score in sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]:  # 只显示前10个
            print(f"  {tag}: {score:.3f}")

        return {"tags": results}

    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e), "tags": {}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
