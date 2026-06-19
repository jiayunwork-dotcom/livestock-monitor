"""多栋舍对比模块"""
import pandas as pd
import numpy as np
from utils.comfort_eval import calculate_thi


def calculate_env_means(env_df, barn_ids):
    """
    计算各栋舍环境参数均值 (用于雷达图)
    """
    results = []
    
    for barn_id in barn_ids:
        barn_data = env_df[env_df['栋舍编号'] == barn_id].copy()
        if barn_data.empty:
            continue
        
        barn_data['THI'] = barn_data.apply(
            lambda row: calculate_thi(row['温度'], row['湿度']), axis=1
        )
        
        means = {
            '栋舍编号': barn_id,
            '温度': barn_data['温度'].mean(),
            '湿度': barn_data['湿度'].mean(),
            'THI指数': barn_data['THI'].mean(),
            '氨气浓度(ppm)': barn_data['氨气浓度(ppm)'].mean(),
            'CO2浓度(ppm)': barn_data['CO2浓度(ppm)'].mean(),
            '光照强度(lux)': barn_data['光照强度(lux)'].mean(),
            '噪声(dB)': barn_data['噪声(dB)'].mean(),
        }
        results.append(means)
    
    return pd.DataFrame(results)


def calculate_production_comparison(prod_df, barn_ids):
    """
    生产指标对比
    """
    results = []
    
    for barn_id in barn_ids:
        barn_data = prod_df[prod_df['栋舍编号'] == barn_id].copy()
        if barn_data.empty:
            continue
        
        barn_data = barn_data.sort_values('时间戳')
        
        total_feed = barn_data['日采食量(kg)'].sum()
        total_water = barn_data['日饮水量(L)'].sum()
        total_mortality = barn_data['日死淘数(只)'].sum()
        
        first_weight = barn_data.iloc[0]['平均体重(kg)'] if len(barn_data) > 0 else 0
        last_weight = barn_data.iloc[-1]['平均体重(kg)'] if len(barn_data) > 0 else 0
        weight_gain = last_weight - first_weight
        
        total_livestock = 10000
        survival_rate = (total_livestock - total_mortality) / total_livestock * 100
        
        feed_conversion_ratio = total_feed / (total_livestock * weight_gain) if weight_gain > 0 else 0
        
        results.append({
            '栋舍编号': barn_id,
            '总采食量(kg)': round(total_feed, 2),
            '总饮水量(L)': round(total_water, 2),
            '总死淘数(只)': int(total_mortality),
            '存活率(%)': round(survival_rate, 2),
            '体重增长(kg)': round(weight_gain, 2),
            '料肉比': round(feed_conversion_ratio, 3),
            '日均采食量(kg)': round(total_feed / len(barn_data), 2),
        })
    
    return pd.DataFrame(results)


def find_largest_difference(env_means_df, prod_df, barn_ids):
    """
    找出差异最大的指标项并给出建议
    """
    suggestions = []
    
    if env_means_df.empty or len(env_means_df) < 2:
        return suggestions
    
    env_params = ['温度', '湿度', '氨气浓度(ppm)', 'CO2浓度(ppm)', 'THI指数', '噪声(dB)']
    
    for param in env_params:
        if param not in env_means_df.columns:
            continue
        
        values = env_means_df[param].values
        if len(values) < 2:
            continue
        
        max_val = max(values)
        min_val = min(values)
        max_idx = env_means_df[param].idxmax()
        min_idx = env_means_df[param].idxmin()
        
        if min_val == 0:
            continue
        
        diff_ratio = (max_val - min_val) / min_val
        
        if diff_ratio > 0.2:
            high_barn = env_means_df.loc[max_idx, '栋舍编号']
            low_barn = env_means_df.loc[min_idx, '栋舍编号']
            
            if '氨气' in param:
                suggestions.append(
                    f"{high_barn}栋{param}显著高于其他栋舍(较{low_barn}高{round(diff_ratio*100, 1)}%), 建议检查通风系统"
                )
            elif '温度' in param or 'THI' in param:
                suggestions.append(
                    f"{high_barn}栋{param}显著高于其他栋舍(较{low_barn}高{round(diff_ratio*100, 1)}%), 建议检查降温设备"
                )
            elif 'CO2' in param:
                suggestions.append(
                    f"{high_barn}栋{param}显著高于其他栋舍(较{low_barn}高{round(diff_ratio*100, 1)}%), 建议加强通风"
                )
            elif '湿度' in param:
                suggestions.append(
                    f"{high_barn}栋{param}显著高于其他栋舍(较{low_barn}高{round(diff_ratio*100, 1)}%), 建议检查通风和湿度控制"
                )
            else:
                suggestions.append(
                    f"{high_barn}栋{param}与{low_barn}栋差异较大(差值{round(diff_ratio*100, 1)}%), 建议排查原因"
                )
    
    return suggestions
