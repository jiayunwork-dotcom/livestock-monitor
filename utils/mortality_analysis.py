"""死亡率分析模块"""
import pandas as pd
import numpy as np


def calculate_cumulative_mortality(prod_df, barn_id, total_livestock=10000):
    """
    计算日龄-累计死淘率曲线
    """
    barn_data = prod_df[prod_df['栋舍编号'] == barn_id].sort_values(['日龄', '时间戳']).copy()
    if barn_data.empty:
        return pd.DataFrame()
    
    barn_data = barn_data.groupby('日龄').agg({
        '日死淘数(只)': 'sum',
        '时间戳': 'first'
    }).reset_index()
    
    barn_data['累计死淘数'] = barn_data['日死淘数(只)'].cumsum()
    barn_data['累计死淘率(%)'] = (barn_data['累计死淘数'] / total_livestock) * 100
    barn_data['日死淘率(%)'] = (barn_data['日死淘数(只)'] / total_livestock) * 100
    
    return barn_data


def detect_mortality_inflection_points(prod_df, barn_id, total_livestock=10000, increase_threshold=1.0):
    """
    检测死淘率突增拐点
    日死淘率环比增加超过100%的日期用竖线标记
    """
    barn_data = calculate_cumulative_mortality(prod_df, barn_id, total_livestock)
    if barn_data.empty or len(barn_data) < 2:
        return barn_data, []
    
    barn_data['日死淘率环比'] = barn_data['日死淘率(%)'].pct_change()
    
    inflection_points = barn_data[
        (barn_data['日死淘率环比'] > increase_threshold) & 
        (barn_data['日死淘率(%)'] > 0.01)
    ].copy()
    
    inflection_points_list = []
    for _, row in inflection_points.iterrows():
        inflection_points_list.append({
            '日龄': row['日龄'],
            '日期': row['时间戳'],
            '日死淘率(%)': round(row['日死淘率(%)'], 4),
            '环比增长(%)': round(row['日死淘率环比'] * 100, 2),
            '累计死淘率(%)': round(row['累计死淘率(%)'], 4)
        })
    
    return barn_data, inflection_points_list


def get_env_summary_around_date(env_df, barn_id, target_date, days_before=1, days_after=1):
    """
    获取拐点日期前后的环境参数摘要
    """
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date)
    
    start_date = target_date - pd.Timedelta(days=days_before)
    end_date = target_date + pd.Timedelta(days=days_after)
    
    barn_env = env_df[env_df['栋舍编号'] == barn_id].copy()
    barn_env['时间戳'] = pd.to_datetime(barn_env['时间戳'])
    
    period_env = barn_env[
        (barn_env['时间戳'] >= start_date) & 
        (barn_env['时间戳'] <= end_date)
    ]
    
    if period_env.empty:
        return {}
    
    from utils.comfort_eval import calculate_thi
    period_env['THI'] = period_env.apply(
        lambda row: calculate_thi(row['温度'], row['湿度']), axis=1
    )
    
    summary = {
        '温度均值(℃)': round(period_env['温度'].mean(), 2),
        '温度峰值(℃)': round(period_env['温度'].max(), 2),
        '湿度均值(%)': round(period_env['湿度'].mean(), 2),
        '氨气峰值(ppm)': round(period_env['氨气浓度(ppm)'].max(), 2),
        '氨气均值(ppm)': round(period_env['氨气浓度(ppm)'].mean(), 2),
        'CO2峰值(ppm)': round(period_env['CO2浓度(ppm)'].max(), 2),
        'THI最大值': round(period_env['THI'].max(), 2),
        'THI均值': round(period_env['THI'].mean(), 2),
    }
    
    return summary


def compare_batches(prod_df, barn_ids, total_livestock=10000):
    """
    多批次/多栋舍叠加对比
    """
    all_data = []
    
    for barn_id in barn_ids:
        barn_data = calculate_cumulative_mortality(prod_df, barn_id, total_livestock)
        if not barn_data.empty:
            barn_data['栋舍编号'] = barn_id
            all_data.append(barn_data)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)
