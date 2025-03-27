import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 归一化与反归一化（点）
class DataPreprocessing:
    """
    A class for preprocessing time series data, including normalization,
    applying sliding windows, and splitting data into training and testing sets.
    
    Args:
    window_size (int): The size of the sliding window.
    forecast_step (int): The number of steps ahead to predict.
    train_ratio (float): Ratio of data used for training (default: 0.8).
    """

    def __init__(self, window_size=30, forecast_step=1, train_ratio=0.6, val_ratio=None):
        self.window_size = window_size
        self.forecast_step = forecast_step
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def normalize_data(self, df, target_col='target'):
        """
        Normalize the input data using MinMaxScaler. If input is a Series, it's 
        treated as both features and target. If it's a DataFrame, the 'target_col'
        is normalized separately as the target variable.

        Args:
        df (pd.Series or pd.DataFrame): Input data.
        target_col (str): Name of the target column (default: 'target').

        Returns:
        tuple: (x_normalized, y_normalized, scaler)
        """
        if isinstance(df, pd.Series):
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_normal = scaler.fit_transform(df.values.reshape(-1, 1))
            # 即只有一列数据，既是x也是y
            return df_normal, df_normal, scaler
        
        elif isinstance(df, pd.DataFrame):
            # 存在多列数据，要求有一列叫做Target，也就是目标变量y
            if target_col not in df.columns:
                raise ValueError(f'Target column "{target_col}" not found in data')
            
            # Normalize target column
            scaler_Y = MinMaxScaler(feature_range=(0, 1))
            y = df[target_col].values.reshape(-1, 1)
            y_normal = scaler_Y.fit_transform(y)

            # Normalize feature columns
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            x = df.drop(target_col, axis=1)
            x_normal = scaler_X.fit_transform(x)

            return x_normal, y_normal, scaler_Y
        else:
            raise TypeError('Data must be a pandas Series or DataFrame')
    
    def sliding_window(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - self.window_size):
            if i + self.window_size + self.forecast_step - 1 == len(dataset):
                break
            
            dataX.append(dataset[i : i + self.window_size])
            dataY.append(dataset[:,0][i+self.window_size : i+self.window_size+self.forecast_step].tolist())
        return np.array(dataX), np.vstack(dataY)

    def split_data(self, data):
        """
        Split the data into training and testing sets according to the defined 
        train_ratio.

        Args:
        data (np.array or list): Full dataset.

        Returns:
        tuple: (train_data, test_data)
        """
        if self.val_ratio is None:
            train_size = int(len(data) * self.train_ratio)
            train_data = data[:train_size]
            test_data = data[train_size:]
            return train_data, test_data
            
        elif type(self.val_ratio) is float:
            train_size = int(len(data) * self.train_ratio)
            train_data = data[:train_size]
            val_size = int(len(data) * (self.val_ratio))
            val_data = data[train_size: val_size]
            test_data = data[val_size:]
            return train_data, val_data, test_data