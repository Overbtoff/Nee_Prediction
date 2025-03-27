import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
np.seterr(over='ignore')
# 画训练过程中的loss与指标图    
def plot_metric(dfhistory, metric, column=None):
    train_metrics = dfhistory["train_" + metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    if column is not None: 
        column = " " + column
    else:
        column = ""
    plt.title('Training and validation '+ metric + column)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()

def cal_tda(y_real, y_pred):
    """
    计算趋势方向准确性 (TDA Trend Directional Accuracy) 指标。
    TDA = 1/N * SUM(A_t)
    A_t = 1 if y_t+1 > y_t and y_t+1 - y_t > 0 else 0
    即预测值与真实值的t+1时刻-t时刻 符号相同  都是增加或者减少
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    tda_count = 0
    for i in range(len(y_real) - 1):
        actual_change = y_real[i + 1] - y_real[i]
        predicted_change = y_pred[i + 1] - y_real[i]

        if actual_change * predicted_change >= 0:
            tda_count += 1
    
    # calculate the tda value
    tda = tda_count / (len(y_real) - 1)
    
    return tda

def cal_smape(y_real, y_pred):
    """
    SMAPE 对称平均绝对百分比误差 (Symmetric Mean Absolute Percentage Error) 指标。
    SMAPE = 1/N * SUM(|y_t - ŷ_t|) / ((|y_t| + |ŷ_t|)/2 )
    """
    N = len(y_real)
    smape = 1/N * np.sum(np.abs(y_real - y_pred) / ((np.abs(y_real) + np.abs(y_pred))/2))
    return smape

# 计算点预测的评估指标
def cal_eval(y_real, y_pred):
    """
    **评估指标说明**:
    - R2 (决定系数, Coefficient of Determination):
        表示模型解释了数据变异性的比例。公式: R2 = 1 - Σ(yi - ŷi)^2 / Σ(yi - ȳ)^2
        
    - MSE (均方误差, Mean Squared Error):
        衡量预测值与实际值之间差异的平方的平均值，越大表示误差越大。公式: MSE = Σ(yi - ŷi)^2 / n

    - RMSE (均方根误差, Root Mean Square Error):
        是预测值与真实值之间差值的平方的平均数的平方根，衡量预测的平均误差大小。公式: RMSE = √[Σ(yi - ŷi)^2 / n]

    - MAE (平均绝对误差, Mean Absolute Error):
        是预测值与真实值之间差值的绝对值的平均数，反映预测的平均偏差。公式: MAE = Σ|yi - ŷi| / n

    - MedAE (中值绝对误差, Median Absolute Error):
        是预测值与真实值之间差值的绝对值的中位数，不受极端值的影响，更稳健。公式: MedAE = 中位数(|yi - ŷi|)

    - MAPE (平均绝对百分比误差, Mean Absolute Percentage Error):
        是预测值与真实值之间差值的绝对值占真实值的比例的平均数，通常用于无量纲比较。公式: MAPE = 100 * Σ(|yi - ŷi| / yi) / n
        注意：MAPE 不适用于 yi = 0 的情况，因为会得到无穷大。
    - MdAPE (中值 Median absolute percentage error)

    输入参数:
    y_real - numpy数组，表示测试集的真实目标值。
    y_pred - numpy数组，表示预测的结果。

    输出:
    df_eval - pandas DataFrame对象，包含了评估结果的表格，包括R2, RMSE, MAE, MAPE和MedAE等指标。

    """
    
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred, squared=True)
    rmse = mean_squared_error(y_real, y_pred, squared=False)  # RMSE and MAE are various on different scales
    mae = mean_absolute_error(y_real, y_pred)
    medae = median_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100  # Note that dataset cannot have any 0 value.
    epsilon = np.finfo(np.float64).eps  # to avoid division by zero
    mdape = np.median(np.abs(y_real - y_pred) / np.maximum(np.abs(y_real), epsilon)) * 100  # Note that dataset cannot have any 0 value.
    tda = cal_tda(y_real, y_pred) * 100
    # smape = cal_smape(y_real, y_pred) * 100
    
    df_eval = pd.DataFrame({'R2': r2, 
                            'MSE':mse, 'RMSE': rmse, 
                            'MAE': mae, 'MedAE':medae, 
                            'MAPE': mape, 'MdAPE':mdape,
                            # 'SMAPE': smape,
                            'TDA': tda}, index=['Eval'])

    return df_eval



# 计算区间预测的评估指标
def cal_interval_eval(y_real, pre_low, pre_up, mu=95, eta=50):
    """
    计算区间预测的评估标准
    
    Input and Parameters:
    ---------------------
    y_real           - 真实值
    pre_low          - 区间预测下界
    pre_up           - 区间预测上界
    
    n_samples        - 样本多长
    in_intervals     - 落在区间内/否，一个布尔类型的数组
    n_in_intervals   - 落在区间内的数量
    
    PIAW: prediction interval average width，PIAW，即让预测区间宽度越小越好
    PINAW: prediction interval average normalized width，PINAW，就是除个（y max - y min）
    PICP: prediction interval coverage probability，PICP，即让预测区间覆盖真实值概率越高越好
    ACE(ACPE): average coverage probability error，ACE，1-picp, 即让PICP区间覆盖率变成越小越好
    
    MPICD  
    CWC  覆盖宽度准则(Coverage Width-based Criterion, CWC)
    AWD  累积带宽偏差指标 (Accumulated Width Deviation，AWD)
    
    """
    n_samples = len(y_real)
    
    n_in_intervals = 0
    for i in range(n_samples):
        if pre_low[i]<= y_real[i] <= pre_up[i]:
            n_in_intervals = n_in_intervals+1
       
    
    PIAW = np.mean(pre_up - pre_low)
    PINAW = PIAW/(np.max(y_real)-np.min(y_real))
    
    PICP = n_in_intervals / n_samples * 100

    pre_mid = (pre_low + pre_up)/2
    MPICD = np.mean(np.abs(pre_mid - y_real))
    
    
    A_t_values = np.where(y_real < pre_low, 
                          (pre_low - y_real) / (pre_up - pre_low),
                          np.where(y_real > pre_up, (y_real - pre_up) / (pre_up - pre_low), 0)
                          )
    AWD = np.sum(A_t_values)

    CWC = cal_CWC(PINAW, PICP, mu, eta)
    
    df_interval_eval = pd.DataFrame({
        'PIAW': PIAW, 
        'PINAW': PINAW, 
        'PICP': PICP, 
        'MPICD': MPICD, 
        'AWD': AWD,
        'CWC': CWC
        }, 
        index=[mu]
        )

    return df_interval_eval.T


def cal_CWC(PINAW, PICP, mu, eta):
    # 计算 gamma
    gamma = 1 if PICP < mu else 0

    # 使用NumPy的元素级操作来避免警告
    CWC = PINAW * (1 + gamma * np.exp(-eta * (PICP - mu)))
    return CWC


def cal_multi_quantile_eval(levels, df_pred_true, mu=95, eta=50):
    """
    多个分位数时候计算区间评估指标
    按列拼接  列是不同的分位数
    行是不同分位数下的指标
    
    注意分位数与置信区间含义不同，仅仅此处值相似，mu用于计算CWC
    
    """
    result_dict = {}
    for quantile in levels:
        y_real, pre_low, pre_up = df_pred_true['Real'], df_pred_true[f'Predict-lo-{quantile}'], df_pred_true[f'Predict-hi-{quantile}']
        df_interval_eval = cal_interval_eval(y_real, pre_low, pre_up, mu=mu, eta=eta)
        result_dict[f'Quantile-{quantile}'] = df_interval_eval

    return pd.concat(result_dict.values(), axis=1, keys=result_dict.keys())