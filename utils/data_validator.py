"""数据校验模块"""
import pandas as pd
import numpy as np
from datetime import datetime


def validate_environment_data(df):
    """校验环境监测数据"""
    errors = []
    required_cols = ['时间戳', '栋舍编号', '温度', '湿度', '氨气浓度(ppm)', 'CO2浓度(ppm)', '光照强度(lux)', '噪声(dB)']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return df, [f"缺少必要列: {', '.join(missing_cols)}"], 100.0
    
    df = df.copy()
    df['_异常标记'] = False
    df['_异常原因'] = ''
    
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        row_errors = []
        
        try:
            ts = pd.to_datetime(row['时间戳'])
            if pd.isna(ts):
                row_errors.append('时间戳格式错误')
        except:
            row_errors.append('时间戳格式错误')
        
        numeric_cols = ['温度', '湿度', '氨气浓度(ppm)', 'CO2浓度(ppm)', '光照强度(lux)', '噪声(dB)']
        for col in numeric_cols:
            try:
                val = float(row[col])
                if val < 0:
                    row_errors.append(f'{col}为负数')
            except:
                row_errors.append(f'{col}非数值')
        
        try:
            barn_id = str(row['栋舍编号'])
            if not barn_id:
                row_errors.append('栋舍编号为空')
        except:
            row_errors.append('栋舍编号无效')
        
        if row_errors:
            df.at[idx, '_异常标记'] = True
            df.at[idx, '_异常原因'] = '; '.join(row_errors)
            errors.append(f"第{idx+1}行: {'; '.join(row_errors)}")
    
    error_count = df['_异常标记'].sum()
    error_rate = (error_count / total_rows * 100) if total_rows > 0 else 0
    
    return df, errors, error_rate


def validate_production_data(df):
    """校验生产数据"""
    errors = []
    required_cols = ['时间戳', '栋舍编号', '日采食量(kg)', '日饮水量(L)', '日死淘数(只)', '平均体重(kg)', '日龄']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return df, [f"缺少必要列: {', '.join(missing_cols)}"], 100.0
    
    df = df.copy()
    df['_异常标记'] = False
    df['_异常原因'] = ''
    
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        row_errors = []
        
        try:
            ts = pd.to_datetime(row['时间戳'])
            if pd.isna(ts):
                row_errors.append('时间戳格式错误')
        except:
            row_errors.append('时间戳格式错误')
        
        numeric_cols = ['日采食量(kg)', '日饮水量(L)', '日死淘数(只)', '平均体重(kg)', '日龄']
        for col in numeric_cols:
            try:
                val = float(row[col])
                if val < 0:
                    row_errors.append(f'{col}为负数')
            except:
                row_errors.append(f'{col}非数值')
        
        try:
            barn_id = str(row['栋舍编号'])
            if not barn_id:
                row_errors.append('栋舍编号为空')
        except:
            row_errors.append('栋舍编号无效')
        
        if row_errors:
            df.at[idx, '_异常标记'] = True
            df.at[idx, '_异常原因'] = '; '.join(row_errors)
            errors.append(f"第{idx+1}行: {'; '.join(row_errors)}")
    
    error_count = df['_异常标记'].sum()
    error_rate = (error_count / total_rows * 100) if total_rows > 0 else 0
    
    return df, errors, error_rate


def check_barn_continuity(df, barn_col='栋舍编号'):
    """
    检查栋舍编号是否连续
    返回: (排序列表, 是否连续, 缺失的编号列表)
    """
    import re
    
    barns = df[barn_col].unique()
    barns_sorted = sorted([str(b) for b in barns])
    
    barn_numbers = []
    for barn in barns_sorted:
        matches = re.findall(r'\d+', str(barn))
        if matches:
            barn_numbers.append(int(matches[-1]))
    
    if not barn_numbers:
        return barns_sorted, True, []
    
    barn_numbers_sorted = sorted(barn_numbers)
    min_num = min(barn_numbers_sorted)
    max_num = max(barn_numbers_sorted)
    
    expected_numbers = set(range(min_num, max_num + 1))
    actual_numbers = set(barn_numbers_sorted)
    
    missing_numbers = sorted(expected_numbers - actual_numbers)
    is_continuous = len(missing_numbers) == 0
    
    return barns_sorted, is_continuous, missing_numbers


def check_energy_columns(df):
    """检查是否包含能耗列"""
    energy_cols = ['通风能耗kWh', '加热能耗kWh', '降温能耗kWh']
    existing = [col for col in energy_cols if col in df.columns]
    return existing
