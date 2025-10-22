import numpy as np
from keras.datasets import mnist # 导入keras自带的mnist库
from tensorflow.keras.utils import to_categorical # 导入One-hot编码工具
from keras import models # 导入Keras模型，以及各种神经网络的层
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt # 导入绘图工具包
# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]

# 读入测试集与训练集
(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()

# ====================== 数据预处理 ======================
# 1. 数据归一化（关键修改点）
X_train = X_train_image.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0  # 归一化到 [0, 1]
X_test = X_test_image.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0    # 测试集同训练集处理方式

# 2. 标签one-hot编码
y_train = to_categorical(y_train_label, 10)
y_test = to_categorical(y_test_label, 10)

# ====================== 构建卷积神经网络模型 ======================
model = models.Sequential()

# 第1层：卷积层 + 最大池化层
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第2层：卷积层 + 最大池化层
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第3层：Dropout层 + 展平层
model.add(Dropout(0.25))
model.add(Flatten())

# 第4层：全连接层 + Dropout层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# 输出层：Softmax分类
model.add(Dense(10, activation='softmax'))

# ====================== 编译与训练模型 ======================
model.compile(
    optimizer='adam',       # 替换优化器为更鲁棒的Adam
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型（添加训练过程可视化）
history = model.fit(
    X_train, y_train,
    validation_split=0.2,   # 拆分20%训练数据为验证集
    epochs=10,              # 增加训练轮次提升收敛效果
    batch_size=256,         # 调整批量大小平衡内存与速度
    verbose=1
)

# ====================== 模型评估与预测 ======================
# 在测试集上评估
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集准确率: {test_accuracy:.4f}")  # 预期输出 > 0.985

# 预测第一个测试样本
pred = model.predict(X_test[0:1])  # 注意：保留批次维度
predicted_class = np.argmax(pred[0])
true_class = np.argmax(y_test[0])

print(f"真实标签: {true_class}, 预测标签: {predicted_class}")

# ====================== 可视化预测结果 ======================
plt.imshow(X_test_image[0], cmap='Greys')
plt.title(f"预测: {predicted_class}, 真实: {true_class}")
plt.axis('off')
plt.savefig('mnist_prediction.png', dpi=300, bbox_inches='tight')
print("预测结果已保存为 mnist_prediction.png")

# ====================== 保存模型 ======================
model.save('mnist_cnn_normalized.keras')  # （Keras标准模型格式）