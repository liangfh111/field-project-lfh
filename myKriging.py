from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D
from sklearn.svm import SVR
from pykrige.rk import RegressionKriging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import numpy as np

def training(df):
    coordinates = df[['x', 'y', 'z']].values
    target_values = df['target'].values
    # 我服了，用个勾巴普通kriging
    model = OrdinaryKriging3D(coordinates[:, 0].astype(float), coordinates[:, 1].astype(float), coordinates[:, 2].astype(float), target_values, variogram_model="linear", weight=True)
    
    # 对特征标准化
    # scaler = MinMaxScaler()
    # coordinates = scaler.fit_transform(coordinates)
    # SVR也不行
    # model = SVR(kernel='rbf', C=1.0, epsilon=0.1).fit(coordinates, target_values)
    
    # 直接用随机森林
    rf_model = RandomForestRegressor(n_estimators=1000)
    # rf_model.fit(coordinates, target_values)
    # model = rf_model
    
    # 线性回归
    # lr_model = LinearRegression(copy_X=False, fit_intercept=True)
    # lr_model.fit(coordinates, target_values)
    # model = lr_model
    #*********************************************************************************
    # 回归kriging(使用随机森林/线性回归)
    # model = RegressionKriging(method='ordinary3d', regression_model=rf_model, n_closest_points=50, weight=True)
    # model.fit(coordinates, coordinates, target_values)
    
    return model

def testing(df,model):
    coordinates_test = df[['x', 'y', 'z']].values.astype(float)
    target_values_test = df['target'].values.astype(float)
    # 对特征标准化
    # coordinates_test = scaler.transform(coordinates_test)
    
    # predict_values_test = model.predict(coordinates_test, coordinates_test)
    # predict_values_test = model.predict(coordinates_test)
    
    # 普通kriging
    predict_values_test, ss = model.execute('points', coordinates_test[:, 0].astype(float), coordinates_test[:, 1].astype(float), coordinates_test[:, 2].astype(float))
    
    
    # 机器学习模型直接评估
    print("模型的r2分数为",r2_score(target_values_test, predict_values_test))
    
    #**********************************************************************************
    # 回归kriging评估模型
    # print("Regression Score: ", model.regression_model.score(coordinates_test, target_values_test))
    # print("RK score: ", model.score(coordinates_test, coordinates_test, target_values_test))
    # print("模型的r2分数为",r2_score(target_values_test, predict_values_test))
    
    #**********************************************************************************
    # 计算平均相对误差
    print("模型的平均相对误差为",np.abs(mean_absolute_error(target_values_test,predict_values_test)/np.mean(target_values_test)))
    return predict_values_test,target_values_test
