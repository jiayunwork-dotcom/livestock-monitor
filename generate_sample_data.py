"""生成示例数据，用于测试应用"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_environment_data(days=7, barns=['1号栋', '2号栋', '3号栋'], interval_minutes=15):
    """生成环境监测示例数据"""
    np.random.seed(42)
    
    all_data = []
    start_date = datetime.now() - timedelta(days=days)
    total_intervals = int(days * 24 * 60 / interval_minutes)
    
    for barn in barns:
        base_temp = np.random.uniform(22, 26)
        base_humidity = np.random.uniform(55, 70)
        base_ammonia = np.random.uniform(10, 20)
        base_co2 = np.random.uniform(1500, 2500)
        base_light = np.random.uniform(5000, 10000)
        base_noise = np.random.uniform(50, 70)
        
        for i in range(total_intervals):
            timestamp = start_date + timedelta(minutes=i * interval_minutes)
            
            hour = timestamp.hour
            temp_variation = 2 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            temp = base_temp + temp_variation + np.random.normal(0, 0.5)
            humidity = base_humidity + np.random.normal(0, 2)
            ammonia = base_ammonia + np.random.normal(0, 2)
            co2 = base_co2 + np.random.normal(0, 200)
            light = base_light * (0.8 + 0.4 * np.sin(2 * np.pi * (hour - 6) / 24)) + np.random.normal(0, 500)
            noise = base_noise + np.random.normal(0, 2)
            
            if barn == '3号栋' and i > total_intervals * 0.6 and i < total_intervals * 0.7:
                ammonia += 15
                co2 += 1500
            
            if barn == '2号栋' and i % 100 == 0:
                temp += 8
            
            temp = max(0, min(50, temp))
            humidity = max(0, min(100, humidity))
            ammonia = max(0, min(100, ammonia))
            co2 = max(0, min(10000, co2))
            light = max(0, min(100000, light))
            noise = max(0, min(120, noise))
            
            all_data.append({
                '时间戳': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                '栋舍编号': barn,
                '温度': round(temp, 2),
                '湿度': round(humidity, 2),
                '氨气浓度(ppm)': round(ammonia, 2),
                'CO2浓度(ppm)': round(co2, 2),
                '光照强度(lux)': round(light, 2),
                '噪声(dB)': round(noise, 2),
                '通风能耗kWh': round(abs(np.random.normal(50, 10)) + temp * 0.5, 2),
                '加热能耗kWh': round(abs(np.random.normal(20, 5)), 2) if temp < 20 else 0,
                '降温能耗kWh': round(abs(np.random.normal(30, 8)) + max(0, temp - 25) * 2, 2),
            })
    
    df = pd.DataFrame(all_data)
    return df


def generate_production_data(days=30, barns=['1号栋', '2号栋', '3号栋'], total_livestock=10000):
    """生成生产数据示例"""
    np.random.seed(123)
    
    all_data = []
    start_date = datetime.now() - timedelta(days=days)
    
    for barn in barns:
        base_feed = np.random.uniform(900, 1100)
        base_water = np.random.uniform(1800, 2200)
        base_weight = 0.05
        
        for day in range(days):
            date = start_date + timedelta(days=day)
            
            growth_rate = 1 + 0.02 * day
            feed = base_feed * growth_rate + np.random.normal(0, 20)
            water = base_water * growth_rate + np.random.normal(0, 50)
            
            avg_weight = base_weight + day * 0.05 + np.random.normal(0, 0.02)
            
            mortality_base = total_livestock * 0.0005
            mortality = max(0, int(mortality_base + np.random.normal(0, 5)))
            
            if barn == '3号栋' and day > 20 and day < 25:
                feed *= 0.85
                mortality += 20
            
            if barn == '2号栋' and day == 15:
                mortality += 30
            
            all_data.append({
                '时间戳': date.strftime('%Y-%m-%d'),
                '栋舍编号': barn,
                '日采食量(kg)': round(feed, 2),
                '日饮水量(L)': round(water, 2),
                '日死淘数(只)': mortality,
                '平均体重(kg)': round(max(0.05, avg_weight), 3),
                '日龄': day + 1,
            })
    
    df = pd.DataFrame(all_data)
    return df


if __name__ == '__main__':
    output_dir = 'sample_data'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成环境监测示例数据...")
    env_df = generate_environment_data(days=7, barns=['1号栋', '2号栋', '3号栋'])
    env_file = os.path.join(output_dir, '环境监测数据_示例.csv')
    env_df.to_csv(env_file, index=False, encoding='utf-8-sig')
    print(f"已生成: {env_file} ({len(env_df)} 条记录)")
    
    print("\n正在生成生产数据示例...")
    prod_df = generate_production_data(days=30, barns=['1号栋', '2号栋', '3号栋'])
    prod_file = os.path.join(output_dir, '生产数据_示例.csv')
    prod_df.to_csv(prod_file, index=False, encoding='utf-8-sig')
    print(f"已生成: {prod_file} ({len(prod_df)} 条记录)")
    
    print("\n✅ 示例数据生成完成!")
    print(f"数据保存在: {os.path.abspath(output_dir)}/")
