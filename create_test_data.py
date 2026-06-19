"""生成测试数据：1号栋和3号栋（缺失2号栋），含能耗列"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(99)

output_dir = 'test_data'
os.makedirs(output_dir, exist_ok=True)

start_date = datetime.now() - timedelta(days=3)
barns = ['1号栋', '3号栋']  # 故意缺失2号栋
interval_minutes = 60
total_intervals = int(3 * 24 * 60 / interval_minutes)

all_env_data = []
for barn in barns:
    for i in range(total_intervals):
        timestamp = start_date + timedelta(minutes=i * interval_minutes)
        temp = 24 + np.random.normal(0, 1)
        humidity = 60 + np.random.normal(0, 3)
        ammonia = 15 + np.random.normal(0, 2)
        co2 = 2000 + np.random.normal(0, 200)
        light = 6000 if 6 <= timestamp.hour <= 20 else 50
        noise = 60 + np.random.normal(0, 3)
        all_env_data.append({
            '时间戳': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            '栋舍编号': barn,
            '温度': round(temp, 2),
            '湿度': round(humidity, 2),
            '氨气浓度(ppm)': round(ammonia, 2),
            'CO2浓度(ppm)': round(co2, 2),
            '光照强度(lux)': round(light, 2),
            '噪声(dB)': round(noise, 2),
            '通风能耗kWh': round(50 + temp * 0.5, 2),
            '加热能耗kWh': round(20 if temp < 22 else 0, 2),
            '降温能耗kWh': round(30 + max(0, temp - 25) * 2, 2),
        })

env_df = pd.DataFrame(all_env_data)
env_file = os.path.join(output_dir, '测试_环境_1号栋和3号栋_含能耗.csv')
env_df.to_csv(env_file, index=False, encoding='utf-8-sig')
print(f'已生成: {env_file} ({len(env_df)}条记录)')
print(f'栋舍: {sorted(env_df["栋舍编号"].unique().tolist())}')
print(f'能耗列: 通风能耗kWh, 加热能耗kWh, 降温能耗kWh')

all_prod_data = []
start_date_prod = datetime.now() - timedelta(days=15)
for barn in barns:
    for day in range(15):
        date = start_date_prod + timedelta(days=day)
        all_prod_data.append({
            '时间戳': date.strftime('%Y-%m-%d'),
            '栋舍编号': barn,
            '日采食量(kg)': round(1000 + day * 20 + np.random.normal(0, 20), 2),
            '日饮水量(L)': round(2000 + day * 40 + np.random.normal(0, 40), 2),
            '日死淘数(只)': max(0, int(5 + np.random.normal(0, 2))),
            '平均体重(kg)': round(0.1 + day * 0.05, 3),
            '日龄': day + 1,
        })

prod_df = pd.DataFrame(all_prod_data)
prod_file = os.path.join(output_dir, '测试_生产_1号栋和3号栋.csv')
prod_df.to_csv(prod_file, index=False, encoding='utf-8-sig')
print(f'\n已生成: {prod_file} ({len(prod_df)}条记录)')
print(f'栋舍: {sorted(prod_df["栋舍编号"].unique().tolist())}')

print('\n✅ 测试数据生成完毕!')
print('测试要点:')
print('  1. 只有1号栋和3号栋，缺失2号栋 -> 应显示栋舍不连续告警')
print('  2. 包含能耗列 -> 侧边栏应显示8项菜单（含能耗分析）')
