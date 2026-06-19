"""疾病预警模型 - 多因子融合风险评分机制"""
import pandas as pd
import numpy as np
from utils.comfort_eval import calculate_thi, classify_thi


WEIGHTS = {
    'feed_intake': 0.30,    # 采食量异常
    'water_intake': 0.20,   # 饮水量异常
    'mortality': 0.25,      # 死淘异常
    'environment': 0.15,    # 环境恶化
    'air_quality': 0.10,    # 空气质量
}


def check_feed_anomaly(prod_df, barn_id, drop_threshold=0.10, days=2):
    """
    采食异常信号
    采食量连续2天下降超过10%
    """
    barn_data = prod_df[prod_df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
    if len(barn_data) < days + 1:
        return False, []
    
    barn_data = barn_data.sort_values('时间戳')
    feed_vals = barn_data['日采食量(kg)'].values
    dates = barn_data['时间戳'].values
    
    anomaly_dates = []
    for i in range(days, len(feed_vals)):
        is_anomaly = True
        for j in range(days):
            if feed_vals[i - j - 1] <= 0:
                is_anomaly = False
                break
            drop_rate = (feed_vals[i - j - 1] - feed_vals[i - j]) / feed_vals[i - j - 1]
            if drop_rate < drop_threshold:
                is_anomaly = False
                break
        if is_anomaly:
            anomaly_dates.append(dates[i])
    
    return len(anomaly_dates) > 0, anomaly_dates


def check_water_anomaly(prod_df, barn_id, deviation_threshold=0.20, days=2):
    """
    饮水异常信号
    饮水量连续2天偏离历史均值超过20%
    """
    barn_data = prod_df[prod_df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
    if len(barn_data) < days + 5:
        return False, []
    
    water_vals = barn_data['日饮水量(L)'].values
    dates = barn_data['时间戳'].values
    
    anomaly_dates = []
    for i in range(days - 1, len(water_vals)):
        historical_mean = np.mean(water_vals[:i]) if i > 0 else water_vals[0]
        if historical_mean == 0:
            continue
        
        consecutive_deviation = True
        for j in range(days):
            if i - j < 0:
                consecutive_deviation = False
                break
            deviation = abs(water_vals[i - j] - historical_mean) / historical_mean
            if deviation < deviation_threshold:
                consecutive_deviation = False
                break
        
        if consecutive_deviation:
            anomaly_dates.append(dates[i])
    
    return len(anomaly_dates) > 0, anomaly_dates


def check_mortality_anomaly(prod_df, barn_id, total_livestock=None, threshold=0.005):
    """
    死淘异常信号
    日死淘率超过0.5%
    """
    barn_data = prod_df[prod_df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
    if barn_data.empty:
        return False, []
    
    dates = barn_data['时间戳'].values
    mortality_counts = barn_data['日死淘数(只)'].values
    
    if total_livestock is None:
        total_livestock = 10000
    
    anomaly_dates = []
    for i in range(len(mortality_counts)):
        mortality_rate = mortality_counts[i] / total_livestock
        if mortality_rate > threshold:
            anomaly_dates.append(dates[i])
    
    return len(anomaly_dates) > 0, anomaly_dates


def check_environment_deterioration(env_df, barn_id, livestock_type='肉鸡', hours=6):
    """
    环境恶化信号
    环境舒适度连续处于中度以上热应激超过6小时
    """
    barn_data = env_df[env_df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
    if barn_data.empty:
        return False, []
    
    barn_data['THI'] = barn_data.apply(
        lambda row: calculate_thi(row['温度'], row['湿度']), axis=1
    )
    
    barn_data['中度以上应激'] = barn_data['THI'].apply(
        lambda thi: classify_thi(thi, livestock_type)[0] in ['中度热应激', '重度热应激']
    )
    
    barn_data = barn_data.set_index('时间戳')
    stress_series = barn_data['中度以上应激'].astype(int)
    
    rolling_sum = stress_series.rolling(f'{hours}h').sum()
    rolling_count = stress_series.rolling(f'{hours}h').count()
    
    continuous_stress = (rolling_sum == rolling_count) & (rolling_count > 0)
    
    anomaly_dates = []
    if continuous_stress.any():
        stress_times = continuous_stress[continuous_stress].index
        if len(stress_times) > 0:
            anomaly_dates = [stress_times[0]]
    
    return len(anomaly_dates) > 0, anomaly_dates


def check_air_quality_anomaly(env_df, barn_id, threshold=25, hours=4):
    """
    空气质量信号
    氨气浓度连续超过25ppm超过4小时
    """
    barn_data = env_df[env_df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
    if barn_data.empty:
        return False, []
    
    barn_data = barn_data.set_index('时间戳')
    ammonia_series = barn_data['氨气浓度(ppm)']
    
    above_threshold = (ammonia_series > threshold).astype(int)
    rolling_sum = above_threshold.rolling(f'{hours}h').sum()
    rolling_count = above_threshold.rolling(f'{hours}h').count()
    
    continuous_high = (rolling_sum == rolling_count) & (rolling_count > 0)
    
    anomaly_dates = []
    if continuous_high.any():
        high_times = continuous_high[continuous_high].index
        if len(high_times) > 0:
            anomaly_dates = [high_times[0]]
    
    return len(anomaly_dates) > 0, anomaly_dates


def calculate_disease_risk(env_df, prod_df, barn_id, livestock_type='肉鸡', total_livestock=10000):
    """
    计算疾病风险评分
    满分1.0，超过0.4为低风险(黄色), 超过0.6为中风险(橙色), 超过0.8为高风险(红色)
    """
    feed_anomaly, feed_dates = check_feed_anomaly(prod_df, barn_id)
    water_anomaly, water_dates = check_water_anomaly(prod_df, barn_id)
    mortality_anomaly, mortality_dates = check_mortality_anomaly(prod_df, barn_id, total_livestock)
    env_anomaly, env_dates = check_environment_deterioration(env_df, barn_id, livestock_type)
    air_anomaly, air_dates = check_air_quality_anomaly(env_df, barn_id)
    
    risk_score = 0.0
    triggered_signals = []
    
    if feed_anomaly:
        risk_score += WEIGHTS['feed_intake']
        triggered_signals.append('采食量异常')
    
    if water_anomaly:
        risk_score += WEIGHTS['water_intake']
        triggered_signals.append('饮水量异常')
    
    if mortality_anomaly:
        risk_score += WEIGHTS['mortality']
        triggered_signals.append('死淘异常')
    
    if env_anomaly:
        risk_score += WEIGHTS['environment']
        triggered_signals.append('环境恶化')
    
    if air_anomaly:
        risk_score += WEIGHTS['air_quality']
        triggered_signals.append('空气质量异常')
    
    if risk_score >= 0.8:
        risk_level = '高风险'
        risk_color = 'red'
    elif risk_score >= 0.6:
        risk_level = '中风险'
        risk_color = 'orange'
    elif risk_score >= 0.4:
        risk_level = '低风险'
        risk_color = 'yellow'
    else:
        risk_level = '正常'
        risk_color = 'green'
    
    return {
        '栋舍编号': barn_id,
        '风险评分': round(risk_score, 3),
        '风险等级': risk_level,
        '风险颜色': risk_color,
        '触发信号': triggered_signals,
        '采食异常': feed_anomaly,
        '饮水异常': water_anomaly,
        '死淘异常': mortality_anomaly,
        '环境恶化': env_anomaly,
        '空气质量异常': air_anomaly,
    }


def calculate_daily_risk_timeline(env_df, prod_df, barn_id, livestock_type='肉鸡', total_livestock=10000):
    """
    按日期展示风险评分时间线
    """
    barn_prod = prod_df[prod_df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
    if barn_prod.empty:
        return pd.DataFrame()
    
    barn_env = env_df[env_df['栋舍编号'] == barn_id].copy()
    
    dates = barn_prod['时间戳'].unique()
    results = []
    
    for date in dates:
        date_env = barn_env[barn_env['时间戳'].dt.date == pd.Timestamp(date).date()]
        date_prod = barn_prod[barn_prod['时间戳'].dt.date == pd.Timestamp(date).date()]
        
        if date_prod.empty:
            continue
        
        risk_data = calculate_disease_risk(date_env, date_prod, barn_id, livestock_type, total_livestock)
        risk_data['日期'] = date
        results.append(risk_data)
    
    return pd.DataFrame(results)


def get_all_barns_risk(env_df, prod_df, livestock_type='肉鸡', total_livestock=10000):
    """获取所有栋舍的疾病风险评分"""
    barns = prod_df['栋舍编号'].unique()
    results = []
    
    for barn_id in barns:
        risk = calculate_disease_risk(env_df, prod_df, barn_id, livestock_type, total_livestock)
        results.append(risk)
    
    return pd.DataFrame(results)
