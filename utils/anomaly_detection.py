"""异常检测模块"""
import pandas as pd
import numpy as np


ABSOLUTE_THRESHOLDS = {
    '温度': {'min': -10, 'max': 50},
    '湿度': {'min': 0, 'max': 100},
    '氨气浓度(ppm)': {'min': 0, 'max': 100},
    'CO2浓度(ppm)': {'min': 0, 'max': 10000},
    '光照强度(lux)': {'min': 0, 'max': 100000},
    '噪声(dB)': {'min': 0, 'max': 120},
}


MUTATION_THRESHOLDS = {
    '温度': 5,
    '氨气浓度(ppm)': 10,
    '湿度': 20,
    'CO2浓度(ppm)': 1000,
    '光照强度(lux)': 5000,
    '噪声(dB)': 20,
}


def detect_absolute_threshold(df, param, barn_id):
    """绝对阈值法检测异常"""
    barn_data = df[df['栋舍编号'] == barn_id].copy()
    if barn_data.empty:
        return barn_data
    
    thresholds = ABSOLUTE_THRESHOLDS.get(param, {'min': -float('inf'), 'max': float('inf')})
    barn_data[f'{param}_绝对阈值异常'] = (
        (barn_data[param] < thresholds['min']) | 
        (barn_data[param] > thresholds['max'])
    )
    return barn_data


def detect_statistical(df, param, barn_id, sigma=3):
    """统计法 (3倍标准差) 检测异常"""
    barn_data = df[df['栋舍编号'] == barn_id].copy()
    if barn_data.empty or len(barn_data) < 2:
        barn_data[f'{param}_统计异常'] = False
        return barn_data
    
    mean_val = barn_data[param].mean()
    std_val = barn_data[param].std()
    
    if std_val == 0:
        barn_data[f'{param}_统计异常'] = False
        return barn_data
    
    barn_data[f'{param}_统计异常'] = (
        (barn_data[param] < mean_val - sigma * std_val) | 
        (barn_data[param] > mean_val + sigma * std_val)
    )
    return barn_data


def detect_mutation(df, param, barn_id):
    """突变检测 (相邻时间点变化量超过合理范围)"""
    barn_data = df[df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
    if barn_data.empty or len(barn_data) < 2:
        barn_data[f'{param}_突变异常'] = False
        return barn_data
    
    threshold = MUTATION_THRESHOLDS.get(param, float('inf'))
    diffs = barn_data[param].diff().abs()
    barn_data[f'{param}_突变异常'] = diffs > threshold
    barn_data[f'{param}_突变异常'] = barn_data[f'{param}_突变异常'].fillna(False)
    return barn_data


def detect_anomalies_for_barn_param(df, param, barn_id):
    """对某个栋舍的某个参数执行三种异常检测"""
    barn_data = df[df['栋舍编号'] == barn_id].copy()
    if barn_data.empty:
        return barn_data
    
    result1 = detect_absolute_threshold(barn_data, param, barn_id)
    result2 = detect_statistical(result1, param, barn_id)
    result3 = detect_mutation(result2, param, barn_id)
    
    result3[f'{param}_异常'] = (
        result3[f'{param}_绝对阈值异常'] | 
        result3[f'{param}_统计异常'] | 
        result3[f'{param}_突变异常']
    )
    
    return result3


def detect_sensor_drift(df, param, barn_id, hours=24, std_threshold=0.1):
    """
    传感器漂移检测
    如果某个传感器连续24小时的读数标准差低于0.1, 判定为传感器可能故障
    """
    barn_data = df[df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
    if barn_data.empty:
        return False, None
    
    barn_data = barn_data.set_index('时间戳')
    rolling_std = barn_data[param].rolling(f'{hours}h').std()
    
    drift_detected = (rolling_std < std_threshold).any()
    drift_time = None
    if drift_detected:
        drift_idx = rolling_std[rolling_std < std_threshold].index[0]
        drift_time = drift_idx
    
    return drift_detected, drift_time


def get_all_anomalies(df):
    """获取所有栋舍所有参数的异常检测结果"""
    params = ['温度', '湿度', '氨气浓度(ppm)', 'CO2浓度(ppm)', '光照强度(lux)', '噪声(dB)']
    barns = df['栋舍编号'].unique()
    
    all_results = []
    
    for barn_id in barns:
        barn_data = df[df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
        
        for param in params:
            result = detect_anomalies_for_barn_param(barn_data, param, barn_id)
            barn_data[f'{param}_异常'] = result[f'{param}_异常'].values
            barn_data[f'{param}_绝对阈值异常'] = result[f'{param}_绝对阈值异常'].values
            barn_data[f'{param}_统计异常'] = result[f'{param}_统计异常'].values
            barn_data[f'{param}_突变异常'] = result[f'{param}_突变异常'].values
        
        barn_data['任一异常'] = barn_data[[f'{p}_异常' for p in params]].any(axis=1)
        all_results.append(barn_data)
    
    return pd.concat(all_results, ignore_index=True)


def summarize_anomalies(df):
    """汇总异常统计"""
    anomaly_df = get_all_anomalies(df)
    params = ['温度', '湿度', '氨气浓度(ppm)', 'CO2浓度(ppm)', '光照强度(lux)', '噪声(dB)']
    
    summary = []
    for barn_id in df['栋舍编号'].unique():
        barn_anomalies = anomaly_df[anomaly_df['栋舍编号'] == barn_id]
        total_count = len(barn_anomalies)
        anomaly_count = barn_anomalies['任一异常'].sum()
        
        param_stats = {}
        for param in params:
            param_stats[param] = barn_anomalies[f'{param}_异常'].sum()
        
        summary.append({
            '栋舍编号': barn_id,
            '采样点数': total_count,
            '异常点数': anomaly_count,
            '异常占比(%)': round(anomaly_count / total_count * 100, 2) if total_count > 0 else 0,
            **param_stats
        })
    
    return pd.DataFrame(summary)


def check_all_sensor_drift(df):
    """检查所有传感器漂移情况"""
    params = ['温度', '湿度', '氨气浓度(ppm)', 'CO2浓度(ppm)', '光照强度(lux)', '噪声(dB)']
    barns = df['栋舍编号'].unique()
    
    results = []
    for barn_id in barns:
        for param in params:
            drift_detected, drift_time = detect_sensor_drift(df, param, barn_id)
            if drift_detected:
                results.append({
                    '栋舍编号': barn_id,
                    '传感器': param,
                    '漂移检测时间': drift_time,
                    '状态': '疑似故障'
                })
    
    return pd.DataFrame(results)
