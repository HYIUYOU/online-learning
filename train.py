import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 创建模拟数据集
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
print("数据集大小：", X.shape, y.shape)
#print("标签分布：", np.bincount(y))


# 初始化在线学习模型
model = SGDClassifier(max_iter=1, tol=None)

# 确保每批次至少有一个样本
batch_size = 10
n_batches = int(np.ceil(X.shape[0] / batch_size))

# # 在线学习：分批逐步训练模型
# for i in range(n_batches):
#     start_index = i * batch_size
#     end_index = start_index + batch_size
#     X_batch, y_batch = X[start_index:end_index], y[start_index:end_index]
#     model.partial_fit(X_batch, y_batch, classes=np.unique(y))

# # 使用模型进行预测
# sample_data = X[:10]  # 假设这是新的数据\
# print("真实结果：", y[:10])
# predictions = model.predict(sample_data)
# print("预测结果：", predictions)

def train():
    # 在线学习：分批逐步训练模型
    for i in range(n_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        X_batch, y_batch = X[start_index:end_index], y[start_index:end_index]
        model.partial_fit(X_batch, y_batch, classes=np.unique(y))

    # 使用模型进行预测
    sample_data = X[:10]  # 假设这是新的数据
    print("真实结果：", y[:10])
    predictions = model.predict(sample_data)

    print("预测结果：", predictions)

for i in range(10):
    train()
    model = SGDClassifier(max_iter=1, tol=None)
    print("第{}次训练".format(i+1))
