
from keras.models import load_model
from tensorflow.keras.utils import to_categorical # 导入One-hot编码工具
m = int(input("请输入预测第几个数据："))
# 加载模型
model = load_model('mnist_cnn.keras')
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from keras.datasets import mnist
(_, _), (X_test_image, y_test_label) = mnist.load_data()
X_test = X_test_image.reshape(10000, 28, 28, 1)
y_test = to_categorical(y_test_label, 10)

# 在测试集上评估模型
score = model.evaluate(X_test, y_test)
print('测试集预测准确率', score[1])

# 预测第一个数据
pred = model.predict(X_test[m].reshape(1, 28, 28, 1))
print(pred[0], "转换一下格式得到：", pred.argmax())

# 输出图片（如果需要）
import matplotlib.pyplot as plt
plt.imshow(X_test[m].reshape(28, 28), cmap='Greys')
plt.axis('off')
plt.title('MNIST Test Image')
# plt.show()
# 如果需要保存图像（替代交互式显示）
plt.savefig(f'第{m}张.png')