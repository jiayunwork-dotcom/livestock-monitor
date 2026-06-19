"""环境舒适度评价模块"""
import pandas as pd
import numpy as np


THI_THRESHOLDS = {
    '肉鸡': {'正常': 70, '轻度热应激': 78, '中度热应激': 85, '重度热应激': float('inf')},
    '蛋鸡': {'正常': 68, '轻度热应激': 75, '中度热应激': 82, '重度热应激': float('inf')},
    '生猪': {'正常': 72, '轻度热应激': 79, '中度热应激': 86, '重度热应激': float('inf')},
    '奶牛': {'正常': 70, '轻度热应激': 77, '中度热应激': 84, '重度热应激': float('inf')},
}


def calculate_thi(temperature, humidity):
    """计算温湿度指数 THI = 0.8*温度 + 湿度/100*(温度-14.4) + 46.4"""
    thi = 0.8 * temperature + (humidity / 100) * (temperature - 14.4) + 46.4
    return thi


def classify_thi(thi_value, livestock_type='肉鸡'):
    """THI分级"""
    thresholds = THI_THRESHOLDS.get(livestock_type, THI_THRESHOLDS['肉鸡'])
    
    if thi_value < thresholds['正常']:
        return '正常', 'green'
    elif thi_value < thresholds['轻度热应激']:
        return '轻度热应激', 'yellow'
    elif thi_value < thresholds['中度热应激']:
        return '中度热应激', 'orange'
    else:
        return '重度热应激', 'red'


def classify_ammonia(ammonia_level):
    """氨气分级: 超过20ppm为一级告警, 超过35ppm为二级告警"""
    if ammonia_level <= 20:
        return '正常', 'green'
    elif ammonia_level <= 35:
        return '一级告警', 'yellow'
    else:
        return '二级告警', 'red'


def classify_co2(co2_level):
    """CO2分级: 超过3000ppm为一级告警, 超过5000ppm为二级告警"""
    if co2_level <= 3000:
        return '正常', 'green'
    elif co2_level <= 5000:
        return '一级告警', 'yellow'
    else:
        return '二级告警', 'red'


def check_ventilation_efficiency(df, barn_id):
    """
    通风效率评估
    如果同一栋舍连续3个采样点的CO2浓度持续上升且增幅每次超过200ppm, 则判定通风不良
    """
    barn_data = df[df['栋舍编号'] == barn_id].sort_values('时间戳').reset_index(drop=True)
    
    if len(barn_data) < 3:
        return False, []
    
    co2_levels = barn_data['CO2浓度(ppm)'].values
    timestamps = barn_data['时间戳'].values
    bad_ventilation_periods = []
    
    for i in range(len(co2_levels) - 2):
        diff1 = co2_levels[i+1] - co2_levels[i]
        diff2 = co2_levels[i+2] - co2_levels[i+1]
        
        if diff1 > 200 and diff2 > 200:
            bad_ventilation_periods.append({
                '开始时间': timestamps[i],
                '结束时间': timestamps[i+2],
                'CO2增量1': diff1,
                'CO2增量2': diff2
            })
    
    has_bad_ventilation = len(bad_ventilation_periods) > 0
    return has_bad_ventilation, bad_ventilation_periods


def evaluate_barn_status(df, livestock_type='肉鸡'):
    """评估每个栋舍的当前状态 (取最新数据)"""
    results = []
    barns = df['栋舍编号'].unique()
    
    for barn_id in barns:
        barn_data = df[df['栋舍编号'] == barn_id].sort_values('时间戳')
        if barn_data.empty:
            continue
        
        latest = barn_data.iloc[-1]
        
        thi = calculate_thi(latest['温度'], latest['湿度'])
        thi_status, thi_color = classify_thi(thi, livestock_type)
        
        ammonia_status, ammonia_color = classify_ammonia(latest['氨气浓度(ppm)'])
        co2_status, co2_color = classify_co2(latest['CO2浓度(ppm)'])
        
        has_bad_vent, _ = check_ventilation_efficiency(df, barn_id)
        vent_status = '通风不良' if has_bad_vent else '正常'
        vent_color = 'red' if has_bad_vent else 'green'
        
        overall_level = max(
            [thi_color, ammonia_color, co2_color, vent_color],
            key=lambda x: {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}.get(x, 0)
        )
        overall_status = {
            'green': '良好',
            'yellow': '轻度告警',
            'orange': '中度告警',
            'red': '重度告警'
        }[overall_level]
        
        results.append({
            '栋舍编号': barn_id,
            '更新时间': latest['时间戳'],
            '温度(℃)': round(latest['温度'], 1),
            '湿度(%)': round(latest['湿度'], 1),
            'THI指数': round(thi, 1),
            'THI状态': thi_status,
            '氨气(ppm)': round(latest['氨气浓度(ppm)'], 1),
            '氨气状态': ammonia_status,
            'CO2(ppm)': round(latest['CO2浓度(ppm)'], 1),
            'CO2状态': co2_status,
            '通风状态': vent_status,
            '综合状态': overall_status,
            '综合等级颜色': overall_level
        })
    
    return pd.DataFrame(results)
