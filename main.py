import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.compose import make_column_selector
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import expon, reciprocal
from sklearn.model_selection import GridSearchCV

# 加载Ames房价数据集
housing = fetch_openml(name="house_prices", as_frame=True)

# 获取特征矩阵和目标变量
X = housing.data
y = housing.target

# 选择需要可视化的特征列
feature_columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt']

# 将特征矩阵转换为 pandas DataFrame
df = pd.DataFrame(X, columns=housing.feature_names)

# 计算特征之间的相关系数矩阵
correlation_matrix = df[feature_columns].corr()

# 设置图形的大小
plt.figure(figsize=(12, 6))

# 绘制特征与目标变量之间的关系
for i, feature in enumerate(feature_columns, 1):
    plt.subplot(2, 3, i)
    plt.scatter(X[feature], y)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(f'{feature} vs SalePrice')

# 绘制热力图
plt.subplot(2, 3, 5)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='PuBu')
plt.title('Correlation Heatmap')

# 调整子图之间的间距
plt.tight_layout()

# 打印特征之间的相关系数矩阵
print("Correlation Matrix:")
print(correlation_matrix)

# 计算特征之间的共线性
vif = pd.DataFrame()
vif["Features"] = feature_columns
vif["VIF"] = [1 / (1 - sm.OLS(df[feature], sm.add_constant(df[feature_columns].drop(columns=feature))).fit().rsquared) for feature in feature_columns]

# 打印特征之间的共线性
print("\nVariance Inflation Factor (VIF):")
print(vif)

# 判断是否适合线性回归
if all(vif["VIF"] < 5):
    print("\nThe features are suitable for linear regression.")
else:
    print("\nThe features may have multicollinearity issues and may not be suitable for linear regression.")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)

# 创建预处理管道
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
    ('cat', cat_pipeline, make_column_selector(dtype_exclude=np.number)),
])

# 使用预处理管道对训练集和测试集进行预处理
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# 定义随机森林参数网格
param_grid = {
    'n_estimators': [100, 200, 300],  # 尝试不同的树的数量
    'max_features': ['sqrt'],  # 特征的选择方式
    'max_depth': [10, 20, 30],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶子节点最少样本数
    # 可以继续添加其他参数
}

# 创建随机森林回归模型对象
rf = RandomForestRegressor()

# 实例化网格搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# 使用训练集进行网格搜索
grid_search.fit(X_train_preprocessed, y_train)

# 打印最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数创建一个新的随机森林模型
rf_model_best = grid_search.best_estimator_

# 使用训练好的最佳模型进行预测
y_pred_rf_best = rf_model_best.predict(X_test_preprocessed)

# 计算最佳随机森林的MSE
mse_rf_best = mean_squared_error(y_test, y_pred_rf_best)
# 计算RMSE
rmse_rf_best = sqrt(mse_rf_best)
print("Random Forest RMSE with best parameters:", rmse_rf_best)


# 计算交叉验证的RMSE
scores = cross_val_score(rf_model_best, X_train_preprocessed, y_train,
                         scoring='neg_mean_squared_error', cv=10)
rf_rmse_scores = np.sqrt(-scores)
print("Random Forest RMSE (cross-validated):", rf_rmse_scores.mean())


# 创建支持向量机回归模型对象
svr = SVR()

# 定义随机搜索的参数分布
param_dist = {
    'kernel': ['linear', 'rbf'],        # 使用线性或径向基核函数
    'C': reciprocal(1, 100),            # 正则化参数
    'gamma': expon(scale=1.0),          # 核函数的参数
}

# 实例化随机搜索
random_search = RandomizedSearchCV(estimator=svr, param_distributions=param_dist,
                                   n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1,
                                   scoring='neg_mean_squared_error')

# 对预处理过的训练数据执行随机搜索
random_search.fit(X_train_preprocessed, y_train)

# 打印随机搜索找到的最佳参数
print("Best parameters found: ", random_search.best_params_)

# 使用找到的最佳参数创建一个新的支持向量机模型
svr_best = random_search.best_estimator_

# 使用最佳模型对预处理过的测试数据进行预测
y_pred_svr_best = svr_best.predict(X_test_preprocessed)

# 计算最佳支持向量机模型的均方误差(MSE)
mse_svr_best = mean_squared_error(y_test, y_pred_svr_best)

# 计算均方根误差(Root Mean Squared Error, RMSE)
rmse_svr_best = np.sqrt(mse_svr_best)
print("最佳支持向量机模型的RMSE:", rmse_svr_best)





# 创建线性回归模型对象
linear_model = LinearRegression()

# 使用训练集进行模型训练
linear_model.fit(X_train_preprocessed, y_train)

# 使用训练好的模型进行预测
y_pred_linear = linear_model.predict(X_test_preprocessed)

mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = sqrt(mse_linear)
print("Linear Regression RMSE:", rmse_linear)

# 绘制预测结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_rf_best)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Random Forest Prediction Results')

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_svr_best)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Support Vector Machine Prediction Results')

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_linear)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Linear Regression Prediction Results')

plt.tight_layout()
plt.show()
