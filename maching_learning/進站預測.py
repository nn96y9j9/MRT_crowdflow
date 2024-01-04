'''
非疫情年進站預測
'''

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

# 訓練模型時優先排除疫情時間
years = [2017, 2018, 2019, 2023]
dataframeList = {}
all_data = pd.DataFrame()

start_time = time.time()

for index, year in enumerate(years):
    df = pd.read_csv(f"main_{year}.csv")
    df = df.groupby(['日期', '時段', '進站']).sum().reset_index()
    df.drop('出站', axis=1, inplace=True)

    # 非疫情年的總資料
    all_data = pd.concat([all_data, df])
    all_data.rename(columns={'日期': 'date', '時段': 'hour',
                    '進站': 'station', '人次': 'crowdflow'}, inplace=True)
end_time = time.time()
execution_time = end_time - start_time
print(f"讀取資料時間為: {execution_time} 秒")

# 檢查車站資料
print(all_data['station'].unique())


# label encoder
all_data['date'] = pd.to_datetime(all_data['date'])

# 插入星期，一~日對應的標籤為1~7
all_data['date'] = pd.to_datetime(all_data['date'])
all_data['weekday'] = all_data['date'].dt.weekday + 1


# 插入假日及所屬月份
hoilday = pd.read_excel('holiday_Calendar.xlsx')
hoilday["date"] = pd.to_datetime(hoilday["年"].astype(
    str) + "-" + hoilday["月"].astype(str) + "-" + hoilday["日"].astype(str))
hoildays = hoilday["date"]
all_data['hoilday'] = all_data['date'].isin(hoildays)
all_data[all_data['hoilday'] == False]
all_data['month'] = all_data['date'].dt.month


print(all_data)

# feature engineering
# 以日期作為切割依據

# 將日期切為三段 2017~2019 與 2023 共 1305天 60%=783天
# 指定要切割的日期範圍 910+180+210天， 比例 0.69 0.137 0.160
date_ranges = [
    (pd.to_datetime('2017-01-01'), pd.to_datetime('2019-05-31'), 0.69),
    (pd.to_datetime('2019-06-01'), pd.to_datetime('2019-12-31'), 0.137),
    (pd.to_datetime('2023-01-01'), pd.to_datetime('2023-06-30'), 0.160)
]

split_data = []

for start_date, end_date, ratio in date_ranges:
    subset = all_data[(all_data['date'] >= start_date)
                      & (all_data['date'] <= end_date)]
    split_data.append((subset, ratio))

train_data, train_ratio = split_data[0]
val_data, val_ratio = split_data[1]
test_data, test_ratio = split_data[2]

# finaly check ratio of data
train_size = int(len(train_data))
val_size = int(len(val_data))
test_size = int(len(test_data))
print(f'測試集比例{test_size / (test_size + val_size + train_size)}')
print(f'驗證集比例{val_size / (test_size + val_size + train_size)}')

# 對車站進行代碼轉換
# 量體越大運量越大
# 以測試集當作標準

train_crowdflow = train_data.groupby(['station']).sum().reset_index()
train_data['station'] = round(train_crowdflow['crowdflow']/910)
val_data['station'] = round(train_crowdflow['crowdflow']/910)
test_data['station'] = round(train_crowdflow['crowdflow']/910)

# 劃分特徵矩陣和目標矩陣
X_train = train_data.drop(['crowdflow', 'date'], axis=1)
y_train = train_data['crowdflow']

X_val = val_data.drop(['crowdflow', 'date'], axis=1)
y_val = val_data['crowdflow']

X_test = test_data.drop(['crowdflow', 'date'], axis=1)
y_test = test_data['crowdflow']


X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print(X_train)
print(y_train)

# RandomForestRegressor

# 超參選擇
# 訓練集選擇

start_time = time.time()

rforest = RandomForestRegressor(
    max_depth=20, n_estimators=20, min_samples_split=5, random_state=0)
rforest.fit(X_train, y_train)

end_time = time.time()

execution_time = end_time - start_time

print("程式執行時間：", execution_time, "秒")

'''
要判斷隨機森林回歸是否存在過度擬合（overfitting），你可以使用以下方法：

1.觀察訓練集和測試集的性能：首先，將數據集分為訓練集和測試集。然後，在訓練集上擬合隨機森林回歸模型，並在測試集上評估模型的性能。如果模型在訓練集上的表現非常好，但在測試集上表現不佳，這可能表示模型存在過度擬合。

2.查看學習曲線：繪製學習曲線可以幫助你更直觀地理解模型是否存在過度擬合。學習曲線是一個以訓練集大小為x軸、模型性能（如R平方）為y軸的曲線。如果模型在訓練集和測試集上的性能逐漸收斂並趨於穩定，則模型較不容易過度擬合。反之，如果模型在訓練集上的性能高於測試集，且兩者之間存在明顯差距，則可能存在過度擬合。

3.使用交叉驗證：交叉驗證是一種有效的方法來評估模型的泛化能力。通過使用交叉驗證來評估不同子集上的模型性能，你可以更全面地了解模型的表現。如果在交叉驗證中模型的平均性能顯著優於測試集的性能，這可能表明模型存在過度擬合。

4.調整超參數：過度擬合可能是由於模型的複雜度過高或某些超參數設置不當導致的。嘗試調整隨機森林回歸的超參數，例如樹的數量（n_estimators）、決策樹的深度（max_depth）、節點分裂所需的最小樣本數（min_samples_split）等，以找到適合的超參數組合。
'''

# 訓練集上的準確率計算
predictions = rforest.predict(X_train)
X_addC = sm.add_constant(predictions)
result = sm.OLS(y_train, X_addC).fit()
print(result.rsquared, result.rsquared_adj)
y_pred = rforest.predict(X_train)
r_squared = r2_score(y_train, y_pred)
print("R Square:", r_squared)
mse = mean_squared_error(y_train, y_pred)
print("MSE:", mse)
mae = mean_absolute_error(y_train, y_pred)
print("MAE:", mae)


# 測試集上的準確率計算
predictions = rforest.predict(X_test)
X_addC = sm.add_constant(predictions)
result = sm.OLS(y_test, X_addC).fit()
print(result.rsquared, result.rsquared_adj)
y_pred = rforest.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print("R Square:", r_squared)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# 畫出預測與實際值模型
true_data = pd.DataFrame(data=y_test)
predictions_data = pd.DataFrame(data=predictions)
combined = predictions_data
combined['Actual'] = true_data[0]
combined.rename(columns={0: 'Predicted'}, inplace=True)
random_combined = combined.sample(n=250, random_state=1)
print(random_combined.head())

# 觀察predictions_data[1000:1250]區間的差異
f, ax = plt.subplots(figsize=(20, 40))
plt.plot(true_data[1000:1250], 'b-', label='actual')
plt.plot(predictions_data[1000:1250], 'yo', label='prediction')
plt.xticks(rotation='vertical')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Traffic')
plt.title('Actual and Predicted Values')
plt.show()

# 使用網格搜尋法尋找最佳超參數
start_time = time.time()

# 創建Random Forest迴歸器
rf_model = RandomForestRegressor()

# 定義超參數的範圍
param_grid = {
    'n_estimators': [10, 20, 30],
    'min_samples_split': [5, 10, 15],
    'max_depth': [5, 10, 15]
}

# 使用Grid Search尋找最佳超參數組合，並以MAE為評估指標
scoring = ['neg_mean_absolute_error', 'r2']

for score in scoring:
    start_time = time.time()
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               scoring=f'{score}', cv=5, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 獲取Grid Search的結果
    results = grid_search.cv_results_

    # 輸出每個CV折疊的訓練MAE、驗證MAE和error bar
    for i in range(len(results[f'{score}'])):
        print(
            f"超參數組合：{results['params'][i]}, 訓練MAE：{-results['mean_train_score'][i]}, 驗證MAE：{-results['mean_test_score'][i]}")
        print(
            f"訓練MAE error bar：{results['std_train_score'][i]}, 驗證MAE error bar：{results['std_test_score'][i]}")

    best_params = grid_search.best_params_
    print("最佳超參數組合：", best_params)

    # 使用最佳超參數來建立最終模型
    best_rf_model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                          min_samples_split=best_params['min_samples_split'],
                                          max_depth=best_params['max_depth'])
    best_rf_model.fit(X_train, y_train)

    end_time = time.time()
    execution_time = end_time - start_time
    print("程式執行時間：", execution_time, "秒")

# 可得到最優參數解為{'max_depth': 15, 'min_samples_split': 5, 'n_estimators': 30}
# learning Curve


start_time = time.time()

# 載入資料並分割為訓練集和測試集
X = X_train
y = y_train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# 初始化模型
model = RandomForestRegressor(
    max_depth=15, min_samples_split=5, n_estimators=30, random_state=0)

# 定義不同的訓練集大小
train_sizes = np.linspace(0.1, 1.0, 10)

# 計算學習曲線
train_sizes_abs, train_scores, test_scores = learning_curve(
    model, X_train, y_train, train_sizes=train_sizes, cv=5)

# 計算性能指標的平均值
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# 繪製學習曲線
plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_scores_mean, label='Training Score')
plt.plot(train_sizes_abs, test_scores_mean, label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Performance')
plt.title('Learning Curve')
plt.legend()
plt.show()

end_time = time.time()
execution_time = end_time - start_time
print("程式執行時間：", execution_time, "秒")

# Variation Curve
start_time = time.time()

# 定義 max_depth 超參數範圍
max_depth_range = [None, 10, 20, 30]

# 初始化空列表，用於保存性能指標的值
train_scores, val_scores = [], []

# 遍歷 max_depth 超參數範圍
for max_depth_value in max_depth_range:
    # 初始化隨機森林回歸模型，並固定 n_estimators 為 100
    model = RandomForestRegressor(
        n_estimators=20, max_depth=max_depth_value, random_state=42)
    # 在訓練集上擬合模型
    model.fit(X_train, y_train)
    # 計算訓練集和驗證集上的均方誤差
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    # 將性能指標的值添加到列表中
    train_scores.append(train_mse)
    val_scores.append(val_mse)

# 繪製驗證曲線
plt.plot(max_depth_range, train_scores, label='Training MSE (max_depth)')
plt.plot(max_depth_range, val_scores, label='Validation MSE (max_depth)')
plt.xlabel('Max Depth')
plt.ylabel('Mean Squared Error')
plt.title('Validation Curve for Random Forest Regression')
plt.legend()
plt.show()

end_time = time.time()
execution_time = end_time - start_time
print("程式執行時間：", execution_time, "秒")
