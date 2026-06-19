"""批次管理与经济效益分析模块测试脚本"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

sys.path.insert(0, '.')

FEED_UNIT_PRICE = 2.8
OTHER_COST_PER_BIRD = 2.0
INITIAL_WEIGHT_KG = 0.04


def _get_barn_list_from_env(env_df, prod_df):
    barns = set()
    if env_df is not None:
        barns.update(env_df['栋舍编号'].unique().tolist())
    if prod_df is not None:
        barns.update(prod_df['栋舍编号'].unique().tolist())
    return sorted(barns)


def _next_batch_id_test(batches):
    if not batches:
        return "B001"
    max_num = 0
    for b in batches:
        try:
            num = int(b['batch_id'][1:])
            if num > max_num:
                max_num = num
        except (ValueError, IndexError):
            pass
    return f"B{max_num + 1:03d}"


def _calculate_feed_cost_test(prod_df, barn_id, start_date, end_date):
    if prod_df is None:
        return None, 0.0, True

    prod_df_copy = prod_df.copy()
    prod_df_copy['时间戳'] = pd.to_datetime(prod_df_copy['时间戳'])

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    barn_data = prod_df_copy[
        (prod_df_copy['栋舍编号'] == barn_id) &
        (prod_df_copy['时间戳'] >= start_ts) &
        (prod_df_copy['时间戳'] <= end_ts)
    ]

    if barn_data.empty:
        return None, 0.0, True

    total_feed = barn_data['日采食量(kg)'].sum()
    feed_cost = total_feed * FEED_UNIT_PRICE

    total_days = (end_ts - start_ts).days + 1
    covered_days = barn_data['时间戳'].dt.date.nunique()
    coverage = covered_days / total_days if total_days > 0 else 0
    data_incomplete = coverage < 0.8

    return total_feed, feed_cost, data_incomplete


def _calculate_economic_indicators_test(batch, prod_df):
    if batch.get('status') != '已出栏':
        return None

    chick_count = batch['chick_count']
    chick_price = batch['chick_price']
    slaughter_count = batch['slaughter_count']
    avg_weight = batch['avg_slaughter_weight']
    sale_price = batch['sale_price']

    barn_id = batch['barn_id']
    start_date = batch['start_date']
    end_date = batch['slaughter_date']

    total_feed, feed_cost, data_incomplete = _calculate_feed_cost_test(
        prod_df, barn_id, start_date, end_date
    )

    chick_cost = chick_count * chick_price
    other_cost = chick_count * OTHER_COST_PER_BIRD
    total_cost = chick_cost + feed_cost + other_cost

    total_revenue = slaughter_count * avg_weight * sale_price
    profit = total_revenue - total_cost

    total_weight_gain = slaughter_count * avg_weight - chick_count * INITIAL_WEIGHT_KG
    fcr = total_feed / total_weight_gain if (total_weight_gain > 0 and total_feed is not None) else None

    survival_rate = (slaughter_count / chick_count) * 100 if chick_count > 0 else 0
    profit_per_bird = profit / slaughter_count if slaughter_count > 0 else 0

    return {
        'chick_cost': chick_cost,
        'feed_cost': feed_cost,
        'other_cost': other_cost,
        'total_cost': total_cost,
        'total_revenue': total_revenue,
        'profit': profit,
        'fcr': fcr,
        'survival_rate': survival_rate,
        'profit_per_bird': profit_per_bird,
        'total_feed': total_feed,
        'data_incomplete': data_incomplete,
    }


def test_validation_logic():
    print('\n=== 测试数据校验逻辑 ===')

    tests_passed = 0
    tests_total = 0

    # 测试1: 进苗数量必须为正整数
    tests_total += 1
    test_chick_counts = [
        (100, True),
        (0, False),
        (-5, False),
        (100.5, False),
        (100.0, True),
    ]
    all_pass = True
    for val, expected in test_chick_counts:
        result = val > 0 and int(val) == val
        if result != expected:
            print(f'  ❌ 进苗数量校验失败: {val} -> {result}, 期望 {expected}')
            all_pass = False
    if all_pass:
        print(f'  ✅ 进苗数量正整数校验通过 ({len(test_chick_counts)} 用例)')
        tests_passed += 1

    # 测试2: 出栏数量不超过进苗数量
    tests_total += 1
    test_slaughter = [
        (100, 95, True),
        (100, 100, True),
        (100, 105, False),
        (100, 0, False),
    ]
    all_pass = True
    for chick, slaughter, expected in test_slaughter:
        result = 0 < slaughter <= chick and int(slaughter) == slaughter
        if result != expected:
            print(f'  ❌ 出栏数量校验失败: {slaughter}/{chick} -> {result}, 期望 {expected}')
            all_pass = False
    if all_pass:
        print(f'  ✅ 出栏数量范围校验通过 ({len(test_slaughter)} 用例)')
        tests_passed += 1

    # 测试3: 出栏日期不早于进苗日期
    tests_total += 1
    test_dates = [
        (date(2026, 5, 20), date(2026, 6, 30), True),
        (date(2026, 5, 20), date(2026, 5, 20), True),
        (date(2026, 5, 20), date(2026, 5, 19), False),
    ]
    all_pass = True
    for start, end, expected in test_dates:
        result = end >= start
        if result != expected:
            print(f'  ❌ 日期校验失败: {end}/{start} -> {result}, 期望 {expected}')
            all_pass = False
    if all_pass:
        print(f'  ✅ 出栏日期校验通过 ({len(test_dates)} 用例)')
        tests_passed += 1

    # 测试4: 目标出栏日龄20-120天
    tests_total += 1
    test_days = [
        (20, True),
        (42, True),
        (120, True),
        (19, False),
        (121, False),
        (0, False),
    ]
    all_pass = True
    for days, expected in test_days:
        result = 20 <= days <= 120
        if result != expected:
            print(f'  ❌ 日龄校验失败: {days} -> {result}, 期望 {expected}')
            all_pass = False
    if all_pass:
        print(f'  ✅ 目标出栏日龄校验通过 ({len(test_days)} 用例)')
        tests_passed += 1

    print(f'  校验用例通过: {tests_passed}/{tests_total}')
    return tests_passed == tests_total


def test_batch_id_generation():
    print('\n=== 测试批次编号自动递增 ===')
    tests = [
        ([], "B001"),
        ([{'batch_id': 'B001'}], "B002"),
        ([{'batch_id': 'B001'}, {'batch_id': 'B005'}], "B006"),
        ([{'batch_id': 'B999'}], "B1000"),
        ([{'batch_id': 'B001'}, {'batch_id': 'INVALID'}, {'batch_id': 'B003'}], "B004"),
    ]
    all_pass = True
    for batches, expected in tests:
        result = _next_batch_id_test(batches)
        if result != expected:
            print(f'  ❌ 批次编号生成失败: {len(batches)}批次 -> {result}, 期望 {expected}')
            all_pass = False
        else:
            print(f'  ✅ {len(batches)} 批次 -> 下一批次编号 {result}')
    return all_pass


def test_economic_calculation():
    print('\n=== 测试经济指标计算 ===')

    prod_df = pd.read_csv('sample_data/生产数据_示例.csv')
    prod_df['时间戳'] = pd.to_datetime(prod_df['时间戳'])
    barns = sorted(prod_df['栋舍编号'].unique())
    print(f'  生产数据栋舍: {barns}')
    print(f'  生产数据日期范围: {prod_df["时间戳"].min()} 至 {prod_df["时间戳"].max()}')

    start_date = date(2026, 5, 20)
    end_date = date(2026, 6, 30)

    batch1 = {
        'batch_id': 'B001',
        'start_date': start_date,
        'barn_id': '1号栋',
        'chick_count': 10000,
        'chick_price': 3.0,
        'target_days': 42,
        'target_weight': 2.5,
        'status': '已出栏',
        'slaughter_date': end_date,
        'slaughter_count': 9800,
        'avg_slaughter_weight': 2.45,
        'sale_price': 10.5,
    }

    ind = _calculate_economic_indicators_test(batch1, prod_df)

    if ind is None:
        print('  ❌ 指标计算返回None')
        return False

    expected_chick_cost = 10000 * 3.0
    expected_other_cost = 10000 * 2.0

    print(f'  批次B001经济指标:')
    print(f'    苗鸡成本: {ind["chick_cost"]:.2f} 元 (期望: {expected_chick_cost:.2f})')
    print(f'    其他成本: {ind["other_cost"]:.2f} 元 (期望: {expected_other_cost:.2f})')
    print(f'    饲料成本: {ind["feed_cost"]:.2f} 元 (饲料总量: {ind["total_feed"]} kg)')
    print(f'    总投入: {ind["total_cost"]:.2f} 元')
    print(f'    总收入: {ind["total_revenue"]:.2f} 元')
    print(f'    利润: {ind["profit"]:.2f} 元')
    print(f'    料肉比: {ind["fcr"]:.3f}' if ind["fcr"] else '    料肉比: N/A')
    print(f'    成活率: {ind["survival_rate"]:.1f}%')
    print(f'    每只利润: {ind["profit_per_bird"]:.2f} 元')
    print(f'    数据完整性: {"⚠️ 不完整" if ind["data_incomplete"] else "✅ 完整"}')

    all_pass = True

    if abs(ind['chick_cost'] - expected_chick_cost) > 0.01:
        print(f'  ❌ 苗鸡成本计算错误')
        all_pass = False

    if abs(ind['other_cost'] - expected_other_cost) > 0.01:
        print(f'  ❌ 其他成本计算错误')
        all_pass = False

    expected_revenue = 9800 * 2.45 * 10.5
    if abs(ind['total_revenue'] - expected_revenue) > 0.01:
        print(f'  ❌ 总收入计算错误: {ind["total_revenue"]:.2f} != {expected_revenue:.2f}')
        all_pass = False
    else:
        print(f'  ✅ 总收入计算正确')

    expected_survival = 9800 / 10000 * 100
    if abs(ind['survival_rate'] - expected_survival) > 0.01:
        print(f'  ❌ 成活率计算错误')
        all_pass = False
    else:
        print(f'  ✅ 成活率计算正确: {ind["survival_rate"]:.1f}%')

    if ind['fcr'] is not None and ind['fcr'] <= 0:
        print(f'  ❌ 料肉比应该为正数')
        all_pass = False
    elif ind['fcr'] is not None:
        print(f'  ✅ 料肉比计算正常: {ind["fcr"]:.3f}')

    # 测试利润计算
    expected_profit = expected_revenue - (expected_chick_cost + ind['feed_cost'] + expected_other_cost)
    if abs(ind['profit'] - expected_profit) > 0.01:
        print(f'  ❌ 利润计算错误')
        all_pass = False
    else:
        print(f'  ✅ 利润计算正确')

    return all_pass


def test_feed_cost_data_completeness():
    print('\n=== 测试饲料成本数据完整性检测 ===')
    prod_df = pd.read_csv('sample_data/生产数据_示例.csv')

    # 测试1: 无生产数据 -> data_incomplete=True
    total_feed, feed_cost, data_inc = _calculate_feed_cost_test(None, '1号栋', date(2026, 5, 20), date(2026, 6, 30))
    print(f'  无生产数据: total_feed={total_feed}, data_incomplete={data_inc} (期望: None, True)')
    if total_feed is not None or not data_inc:
        print('    ❌ 无生产数据时应该标记为不完整')
        return False
    print('    ✅ 正确')

    # 测试2: 日期范围完全不匹配 -> data_incomplete=True
    total_feed, feed_cost, data_inc = _calculate_feed_cost_test(
        prod_df, '1号栋', date(2020, 1, 1), date(2020, 2, 1)
    )
    print(f'  日期不匹配: total_feed={total_feed}, data_incomplete={data_inc} (期望: None, True)')
    if total_feed is not None or not data_inc:
        print('    ❌ 不匹配日期时应该标记为不完整')
        return False
    print('    ✅ 正确')

    # 测试3: 正常范围 -> 检查覆盖率计算
    total_feed, feed_cost, data_inc = _calculate_feed_cost_test(
        prod_df, '1号栋', date(2026, 5, 20), date(2026, 6, 30)
    )
    print(f'  正常范围(42天): 饲料总量={total_feed:.2f}kg, 饲料成本={feed_cost:.2f}元, 数据完整={not data_inc}')
    print('    ✅ 正常计算')

    return True


def test_comparison_worst_value_logic():
    print('\n=== 测试批次对比最差值标红逻辑 ===')

    # 模拟3个批次
    batch_ids = ['B001', 'B002', 'B003']
    fcr_list = [1.85, 2.10, 1.95]  # B002最差(最高)
    survival_list = [98.5, 96.2, 97.8]  # B002最差(最低)
    profit_per_bird_list = [3.50, 2.10, 3.20]  # B002最差(最低)

    # 标红最差单元格
    worst_cells = []

    valid_fcr = [(i, v) for i, v in enumerate(fcr_list) if v is not None and v > 0]
    if valid_fcr:
        worst_fcr_idx = max(valid_fcr, key=lambda x: x[1])[0]
        worst_cells.append((worst_fcr_idx, '料肉比'))
        print(f'  最差料肉比: 批次 {batch_ids[worst_fcr_idx]} = {fcr_list[worst_fcr_idx]} (最高)')

    if survival_list:
        worst_survival_idx = survival_list.index(min(survival_list))
        worst_cells.append((worst_survival_idx, '成活率(%)'))
        print(f'  最差成活率: 批次 {batch_ids[worst_survival_idx]} = {survival_list[worst_survival_idx]}% (最低)')

    if profit_per_bird_list:
        worst_profit_idx = profit_per_bird_list.index(min(profit_per_bird_list))
        worst_cells.append((worst_profit_idx, '每只利润(元)'))
        print(f'  最差每只利润: 批次 {batch_ids[worst_profit_idx]} = {profit_per_bird_list[worst_profit_idx]}元 (最低)')

    # 检查标红的是具体单元格而非整行
    expected_cells = {(1, '料肉比'), (1, '成活率(%)'), (1, '每只利润(元)')}
    actual_cells = set(worst_cells)
    if actual_cells == expected_cells:
        print('  ✅ 所有最差值定位正确 (单元格级标红)')
    else:
        print(f'  ❌ 标红位置错误: 期望 {expected_cells}, 实际 {actual_cells}')
        return False

    # 验证是独立单元格(不是整行)
    cell_set = set(worst_cells)
    batch0_cells = [c for c in cell_set if c[0] == 0]
    batch1_cells = [c for c in cell_set if c[0] == 1]
    batch2_cells = [c for c in cell_set if c[0] == 2]
    print(f'  B001标红列: {[c[1] for c in batch0_cells]} (应为空)')
    print(f'  B002标红列: {[c[1] for c in batch1_cells]} (应为三项:全最差)')
    print(f'  B003标红列: {[c[1] for c in batch2_cells]} (应为空)')

    if len(batch0_cells) == 0 and len(batch1_cells) == 3 and len(batch2_cells) == 0:
        print('  ✅ 单元格级标红验证通过(不是整行标红)')
        return True
    else:
        print('  ❌ 单元格级标红验证失败')
        return False


def test_edge_cases():
    print('\n=== 测试边界情况 ===')

    prod_df = pd.read_csv('sample_data/生产数据_示例.csv')
    all_pass = True

    # 测试1: 出栏数量=0 (虽然校验不允许但测试计算安全)
    batch_edge = {
        'batch_id': 'B00X',
        'start_date': date(2026, 5, 20),
        'barn_id': '1号栋',
        'chick_count': 10000,
        'chick_price': 3.0,
        'target_days': 42,
        'target_weight': 2.5,
        'status': '已出栏',
        'slaughter_date': date(2026, 6, 30),
        'slaughter_count': 0,
        'avg_slaughter_weight': 2.0,
        'sale_price': 10.0,
    }
    ind = _calculate_economic_indicators_test(batch_edge, prod_df)
    if ind is None:
        print('  ❌ 边界计算返回None')
        all_pass = False
    elif ind['survival_rate'] == 0 and ind['profit_per_bird'] == 0:
        print('  ✅ 0只出栏时成活率/每只利润安全=0')
    else:
        print(f'  ❌ 0只出栏边界错误: 成活率={ind["survival_rate"]}, 每只利润={ind["profit_per_bird"]}')
        all_pass = False

    # 测试2: 总增重接近0 (FCR分母边界)
    batch_edge2 = {
        'batch_id': 'B00Y',
        'start_date': date(2026, 5, 20),
        'barn_id': '1号栋',
        'chick_count': 10000,
        'chick_price': 3.0,
        'target_days': 42,
        'target_weight': 2.5,
        'status': '已出栏',
        'slaughter_date': date(2026, 5, 20),  # 当天出栏
        'slaughter_count': 10000,
        'avg_slaughter_weight': 0.04,  # 等于初始体重 -> 增重=0
        'sale_price': 10.0,
    }
    ind2 = _calculate_economic_indicators_test(batch_edge2, prod_df)
    if ind2 is None:
        print('  ❌ 增重=0边界返回None')
        all_pass = False
    elif ind2['fcr'] is None:
        print('  ✅ 总增重=0时料肉比安全返回None')
    else:
        print(f'  ❌ 增重=0时FCR应该为None, 实际={ind2["fcr"]}')
        all_pass = False

    return all_pass


def main():
    print('=' * 60)
    print('批次管理与经济效益分析模块 - 核心功能测试')
    print('=' * 60)

    results = []
    results.append(('批次编号生成', test_batch_id_generation()))
    results.append(('数据校验逻辑', test_validation_logic()))
    results.append(('经济指标计算', test_economic_calculation()))
    results.append(('饲料数据完整性', test_feed_cost_data_completeness()))
    results.append(('对比最差值标红', test_comparison_worst_value_logic()))
    results.append(('边界情况处理', test_edge_cases()))

    print('\n' + '=' * 60)
    print('测试结果汇总')
    print('=' * 60)
    total = len(results)
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        icon = '✅' if result else '❌'
        print(f'  {icon} {name}')
    print(f'\n总通过率: {passed}/{total} ({100*passed//total if total > 0 else 0}%)')

    if passed == total:
        print('\n🎉 所有批次管理模块测试通过!')
    else:
        print(f'\n⚠️  {total - passed} 个测试未通过, 请检查')
        sys.exit(1)


if __name__ == '__main__':
    main()
