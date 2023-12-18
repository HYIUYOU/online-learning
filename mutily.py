from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import numpy as np

# 重新定义和训练多目标回归模型
# 示例多目标数据
X_initial_multi = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
y_initial_multi = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # 每个样本有两个目标值



# Create and train the model
model_multi = MultiOutputRegressor(SGDRegressor(max_iter=1000, tol=1e-3))
model_multi.fit(X_initial_multi, y_initial_multi)

# 输出模型参数
# print("model_multi.coef_:",model_multi.coef_)
# print("model_multi.intercept_:",model_multi.intercept_)


# 输出预测结果
y_initial_pred_multi = model_multi.predict(X_initial_multi)
print("y_initial_multi:",y_initial_multi)
print("y_initial_pred_multi:",y_initial_pred_multi)

# 模拟新数据的到来
X_new_multi = np.array([[5, 10], [6, 12]])
y_new_multi = np.array([[5, 6], [6, 7]])

# 使用新数据更新模型
model_multi.partial_fit(X_new_multi, y_new_multi)

# 输出模型参数
# print("model_multi.coef_:",model_multi.coef_)
# print("model_multi.intercept_:",model_multi.intercept_)

# 输出更新后的预测结果
y_new_pred_multi = model_multi.predict(X_new_multi)
print("y_new_multi",y_new_multi)
print("y_new_pred_multi",y_new_pred_multi)


# # 计算R²分数
# y_initial_pred_multi = model_multi.predict(X_initial_multi)
# y_new_pred_multi = model_multi.predict(X_new_multi)
# r2_initial_multi = r2_score(y_initial_multi, y_initial_pred_multi, multioutput='uniform_average')
# r2_updated_multi = r2_score(y_new_multi, y_new_pred_multi, multioutput='uniform_average')

# r2_initial_multi, r2_updated_multi

# # 输出分数
# print(r2_initial_multi)
# print(r2_updated_multi)
