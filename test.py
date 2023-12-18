from sklearn.linear_model import SGDRegressor
import numpy as np

# 示例数据
# 假设我们有一些初始数据
X_initial = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
y_initial = np.array([3, 6, 9, 12])

# 创建一个SGDRegressor模型，这是一个支持增量学习的线性回归模型
model = SGDRegressor(max_iter=1000, tol=1e-3)

# 使用初始数据训练模型
model.fit(X_initial, y_initial)

# 输出预测结果
y_pred = model.predict(X_initial)
print(y_initial)
print(y_pred)

# 模拟新数据的到来
X_new = np.array([[5, 10], [6, 12]])
y_new = np.array([15, 18])

# 使用新数据更新模型
model.partial_fit(X_new, y_new)

# 输出更新后的预测结果
y_updated = model.predict(X_new)
print(y_new)
print(y_updated)



# 检查更新后模型的参数
# updated_coefficients = model.coef_
# updated_intercept = model.intercept_

# updated_coefficients, updated_intercept


