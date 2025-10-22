from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
import base64
from PIL import Image
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=所有日志，1=INFO以下忽略，2=WARNING以下忽略
# app = Flask(__name__)
# -------------------------- 2. 配置静态文件服务（加载前端页面） --------------------------
# 挂载static目录，让后端能访问static下的index.html和favicon.ico
app = Flask(__name__, static_folder="templates")

# 加载模型
try:
    model = load_model('best_mnist_model.keras')  # 修改为新保存的模型
    # 加载测试数据
    (_, _), (X_test_image, y_test_label) = mnist.load_data()
    X_test = X_test_image.reshape(10000, 28, 28, 1) / 255.0  # 归一化
    y_test = to_categorical(y_test_label, 10)

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'模型加载成功，测试准确率: {accuracy:.4f}')
except Exception as e:
    print(f'模型加载失败: {e}')
    model = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/accuracy')
def get_accuracy():
    if model is None:
        return jsonify({'error': '模型未加载'}), 500
    print("X_test[0]均值:", np.mean(X_test[0]))  # 应接近0.5左右（0-1归一化）
    print("X_test[0]最大值:", np.max(X_test[0]))  # 应接近1.0
    return jsonify({'accuracy': accuracy})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        if data['mode'] == 'test':
            # 从测试集中选择图像
            index = int(data['index'])
            if index < 0 or index >= len(X_test):
                return jsonify({'success': False, 'error': '索引超出范围'}), 400

            image = X_test[index]
            # 转换图像为PNG格式的base64编码
            img_data = (image.reshape(28, 28) * 255).astype(np.uint8)
            img = Image.fromarray(img_data, 'L')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('ascii')

        elif data['mode'] == 'draw':
            # 处理手绘图像
            image_data = data['image_data']
            # 解码base64图像
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('L')  # 转换为灰度图
            # 调整图像大小为28x28
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            # 转换为numpy数组并归一化
            img_array = np.array(img) / 255.0
            # 反转颜色（MNIST图像背景为黑色，数字为白色）
            image = 1.0 - img_array.reshape(28, 28, 1)
            # 转换图像为PNG格式的base64编码，用于显示
            img_data = (image.reshape(28, 28) * 255).astype(np.uint8)
            img_processed = Image.fromarray(img_data, 'L')
            buffered = io.BytesIO()
            img_processed.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('ascii')

        else:
            return jsonify({'success': False, 'error': '未知模式'}), 400

        # 预测
        prediction = model.predict(image.reshape(1, 28, 28, 1), verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0].tolist()

        return jsonify({
            'success': True,
            'prediction': int(predicted_class),
            'confidence': confidence,
            'image_data': img_str,
            'index': index if data['mode'] == 'test' else None
        })

    except Exception as e:
        print(f'预测错误: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)