"""能耗关联分析模块"""
import pandas as pd
import numpy as np


def has_energy_data(env_df):
    """检查是否包含能耗数据"""
    energy_cols = ['通风能耗kWh', '加热能耗kWh', '降温能耗kWh']
    return any(col in env_df.columns for col in energy_cols)


def calculate_energy_co2_relation(env_df, barn_id=None):
    """
    能耗与环境控制效果的散点图数据
    横轴为通风能耗, 纵轴为平均CO2浓度
    """
    data = env_df.copy()
    if barn_id:
        data = data[data['栋舍编号'] == barn_id]
    
    if '通风能耗kWh' not in data.columns:
        return pd.DataFrame()
    
    data = data.dropna(subset=['通风能耗kWh', 'CO2浓度(ppm)'])
    
    if data.empty:
        return pd.DataFrame()
    
    result = data[['时间戳', '栋舍编号', '通风能耗kWh', 'CO2浓度(ppm)', '温度', '湿度']].copy()
    return result


def calculate_energy_water_ratio(env_df, prod_df, barn_id=None):
    """
    计算吨水能耗等效指标: 总能耗除以饮水总量
    """
    env_data = env_df.copy()
    prod_data = prod_df.copy()
    
    if barn_id:
        env_data = env_data[env_data['栋舍编号'] == barn_id]
        prod_data = prod_data[prod_data['栋舍编号'] == barn_id]
    
    energy_cols = ['通风能耗kWh', '加热能耗kWh', '降温能耗kWh']
    available_energy_cols = [col for col in energy_cols if col in env_data.columns]
    
    if not available_energy_cols:
        return None
    
    env_data['总能耗'] = env_data[available_energy_cols].sum(axis=1)
    
    total_energy = env_data['总能耗'].sum()
    total_water = prod_data['日饮水量(L)'].sum()
    
    if total_water == 0:
        return None
    
    energy_per_ton_water = (total_energy / total_water) * 1000  # kWh/吨水
    
    return {
        '总能耗(kWh)': round(total_energy, 2),
        '总饮水量(L)': round(total_water, 2),
        '吨水能耗(kWh/吨)': round(energy_per_ton_water, 4)
    }


def calculate_monthly_energy_trend(env_df, barn_id=None):
    """
    按月汇总能耗趋势
    """
    data = env_df.copy()
    if barn_id:
        data = data[data['栋舍编号'] == barn_id]
    
    energy_cols = ['通风能耗kWh', '加热能耗kWh', '降温能耗kWh']
    available_cols = [col for col in energy_cols if col in data.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    data['月份'] = pd.to_datetime(data['时间戳']).dt.to_period('M')
    
    monthly = data.groupby('月份')[available_cols].sum().reset_index()
    monthly['月份'] = monthly['月份'].astype(str)
    monthly['总能耗'] = monthly[available_cols].sum(axis=1)
    
    return monthly


def get_energy_summary(env_df, prod_df):
    """
    获取各栋舍能耗摘要
    """
    barns = env_df['栋舍编号'].unique()
    results = []
    
    for barn_id in barns:
        energy_info = calculate_energy_water_ratio(env_df, prod_df, barn_id)
        if energy_info:
            energy_info['栋舍编号'] = barn_id
            results.append(energy_info)
    
    return pd.DataFrame(results) if results else pd.DataFrame()
