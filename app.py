"""规模化养殖场环境监测与动物疾病预警系统"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
from scipy.optimize import linprog

from utils.data_validator import validate_environment_data, validate_production_data, check_barn_continuity, check_energy_columns
from utils.comfort_eval import (
    calculate_thi, classify_thi, classify_ammonia, classify_co2,
    check_ventilation_efficiency, evaluate_barn_status, THI_THRESHOLDS
)
from utils.anomaly_detection import (
    get_all_anomalies, summarize_anomalies, check_all_sensor_drift,
    detect_anomalies_for_barn_param
)
from utils.disease_warning import (
    calculate_disease_risk, get_all_barns_risk, calculate_daily_risk_timeline, WEIGHTS
)
from utils.mortality_analysis import (
    calculate_cumulative_mortality, detect_mortality_inflection_points,
    get_env_summary_around_date, compare_batches
)
from utils.energy_analysis import (
    has_energy_data, calculate_energy_co2_relation,
    calculate_energy_water_ratio, calculate_monthly_energy_trend, get_energy_summary
)
from utils.barn_comparison import (
    calculate_env_means, calculate_production_comparison, find_largest_difference
)
from utils.report_generator import generate_daily_report

try:
    _ver_parts = tuple(int(x) for x in st.__version__.split('.'))
    ST_VERSION = _ver_parts if len(_ver_parts) >= 2 else (1, 28, 0)
except Exception:
    ST_VERSION = (1, 28, 0)


def _st_btn_kwargs(full_width=True):
    """Return button kwargs compatible with current Streamlit version."""
    if ST_VERSION >= (1, 30, 0) and full_width:
        return {'use_container_width': True}
    return {}


st.set_page_config(
    page_title="规模化养殖场环境监测与疾病预警系统",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)


def color_status_cell(val, color_col=None):
    """状态单元格颜色标记"""
    if color_col:
        color_map = {
            'green': 'background-color: #C6EFCE; color: #006100',
            'yellow': 'background-color: #FFEB9C; color: #9C5700',
            'orange': 'background-color: #FFC000; color: #806000',
            'red': 'background-color: #FFC7CE; color: #9C0006',
        }
        return color_map.get(val, '')
    return ''


def style_dataframe(df, status_cols=None):
    """给DataFrame添加样式"""
    if status_cols is None:
        status_cols = ['综合状态', 'THI状态', '氨气状态', 'CO2状态', '通风状态', '风险等级']
    
    def highlight_status(s):
        color_map = {
            '良好': 'background-color: #C6EFCE; color: #006100',
            '正常': 'background-color: #C6EFCE; color: #006100',
            '轻度告警': 'background-color: #FFEB9C; color: #9C5700',
            '轻度热应激': 'background-color: #FFEB9C; color: #9C5700',
            '一级告警': 'background-color: #FFEB9C; color: #9C5700',
            '低风险': 'background-color: #FFEB9C; color: #9C5700',
            '中度告警': 'background-color: #FFC000; color: #806000',
            '中度热应激': 'background-color: #FFC000; color: #806000',
            '中风险': 'background-color: #FFC000; color: #806000',
            '重度告警': 'background-color: #FFC7CE; color: #9C0006',
            '重度热应激': 'background-color: #FFC7CE; color: #9C0006',
            '二级告警': 'background-color: #FFC7CE; color: #9C0006',
            '高风险': 'background-color: #FFC7CE; color: #9C0006',
            '通风不良': 'background-color: #FFC7CE; color: #9C0006',
            '疑似故障': 'background-color: #FFC7CE; color: #9C0006',
        }
        return [color_map.get(v, '') for v in s]
    
    styled = df.style
    for col in status_cols:
        if col in df.columns:
            styled = styled.apply(highlight_status, subset=[col])
    
    return styled


def init_session_state():
    """初始化session_state，必须在访问任何session_state键之前调用"""
    defaults = {
        'env_data': None,
        'prod_data': None,
        'env_errors': [],
        'prod_errors': [],
        'env_error_rate': 0,
        'prod_error_rate': 0,
        'current_page': "📥 数据导入",
        'warning_tickets': [],
        'batches': [],
        'global_feed_unit_price': 2.8,
        'feed_ingredients': None,
        'feed_constraints': None,
        'feed_formula_result': None,
        'saved_formulas': [],
        'sensitivity_result': None,
        'ingredient_price_history': {},
        'predicted_prices': {},
        'predicted_feed_formula_result': None,
        'global_feed_unit_price_predicted': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """主应用函数"""
    
    init_session_state()
    
    st.sidebar.title("🐔 养殖场环境监测系统")
    
    livestock_type = st.sidebar.selectbox(
        "选择畜种",
        ["肉鸡", "蛋鸡", "生猪", "奶牛"],
        index=0,
        help="不同畜种的THI分级阈值不同"
    )
    
    total_livestock = st.sidebar.number_input(
        "单栋养殖数量(只/头)",
        min_value=100,
        max_value=100000,
        value=10000,
        step=100
    )
    
    st.sidebar.markdown("---")
    
    has_energy = False
    if st.session_state.env_data is not None:
        has_energy = has_energy_data(st.session_state.env_data)
    
    pages_list = [
        "📥 数据导入",
        "🌡️ 环境舒适度评价",
        "⚠️ 异常检测",
        "🏥 疾病预警",
        "💀 死亡率分析",
        "📊 多栋舍对比",
        "📋 养殖批次管理与经济效益分析",
        "🥗 饲料配方模拟与成本优化",
        "📄 报告导出"
    ]
    
    if has_energy:
        pages_list.insert(5, "⚡ 能耗分析")
    
    saved_page = st.session_state.current_page
    default_index = pages_list.index(saved_page) if saved_page in pages_list else 0
    
    nav_key = f"nav_radio_energy_{has_energy}"
    
    page = st.sidebar.radio(
        "功能模块",
        pages_list,
        index=default_index,
        key=nav_key
    )
    
    st.session_state.current_page = page
    
    if page == "📥 数据导入":
        data_import_page()
    elif page == "🌡️ 环境舒适度评价":
        comfort_page(livestock_type)
    elif page == "⚠️ 异常检测":
        anomaly_page()
    elif page == "🏥 疾病预警":
        disease_warning_page(livestock_type, total_livestock)
    elif page == "💀 死亡率分析":
        mortality_page(total_livestock)
    elif page == "⚡ 能耗分析":
        energy_page()
    elif page == "📊 多栋舍对比":
        comparison_page(livestock_type, total_livestock)
    elif page == "📋 养殖批次管理与经济效益分析":
        batch_management_page()
    elif page == "🥗 饲料配方模拟与成本优化":
        feed_formula_page()
    elif page == "📄 报告导出":
        report_page(livestock_type, total_livestock)


def data_import_page():
    """数据导入页面"""
    st.header("📥 数据导入")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("环境监测数据")
        st.info("CSV格式要求列: 时间戳、栋舍编号、温度、湿度、氨气浓度(ppm)、CO2浓度(ppm)、光照强度(lux)、噪声(dB)")
        
        env_file = st.file_uploader("上传环境监测CSV", type=["csv"], key="env_upload")
        
        if env_file is not None:
            try:
                df = pd.read_csv(env_file)
                validated_df, errors, error_rate = validate_environment_data(df)
                
                st.session_state.env_data = validated_df
                st.session_state.env_errors = errors
                st.session_state.env_error_rate = error_rate
                
                if error_rate > 0:
                    st.warning(f"⚠️ 发现 {int(validated_df['_异常标记'].sum())} 行异常数据, 占比 {error_rate:.2f}%")
                    with st.expander("查看异常详情"):
                        if len(errors) > 20:
                            st.write("... (仅显示前20条)")
                            for err in errors[:20]:
                                st.text(err)
                        else:
                            for err in errors:
                                st.text(err)
                else:
                    st.success(f"✅ 数据校验通过, 共 {len(df)} 条记录")
                
                st.dataframe(validated_df.head(10), use_container_width=True)
                st.caption(f"共 {len(validated_df)} 条记录")
                
                barns_sorted, is_continuous, missing = check_barn_continuity(validated_df)
                st.write(f"栋舍列表: {', '.join(map(str, barns_sorted))}")
                
                if not is_continuous:
                    st.warning(f"⚠️ 栋舍编号不连续, 缺失编号: {', '.join(map(str, missing))}")
                
                energy_cols = check_energy_columns(validated_df)
                if energy_cols:
                    st.success(f"检测到能耗列: {', '.join(energy_cols)}, 将启用能耗分析模块")
                
            except Exception as e:
                st.error(f"文件解析失败: {str(e)}")
    
    with col2:
        st.subheader("生产数据")
        st.info("CSV格式要求列: 时间戳、栋舍编号、日采食量(kg)、日饮水量(L)、日死淘数(只)、平均体重(kg)、日龄")
        
        prod_file = st.file_uploader("上传生产数据CSV", type=["csv"], key="prod_upload")
        
        if prod_file is not None:
            try:
                df = pd.read_csv(prod_file)
                validated_df, errors, error_rate = validate_production_data(df)
                
                st.session_state.prod_data = validated_df
                st.session_state.prod_errors = errors
                st.session_state.prod_error_rate = error_rate
                
                if error_rate > 0:
                    st.warning(f"⚠️ 发现 {int(validated_df['_异常标记'].sum())} 行异常数据, 占比 {error_rate:.2f}%")
                    with st.expander("查看异常详情"):
                        if len(errors) > 20:
                            st.write("... (仅显示前20条)")
                            for err in errors[:20]:
                                st.text(err)
                        else:
                            for err in errors:
                                st.text(err)
                else:
                    st.success(f"✅ 数据校验通过, 共 {len(df)} 条记录")
                
                st.dataframe(validated_df.head(10), use_container_width=True)
                st.caption(f"共 {len(validated_df)} 条记录")
                
                barns_sorted, is_continuous, missing = check_barn_continuity(validated_df)
                st.write(f"栋舍列表: {', '.join(map(str, barns_sorted))}")
                
                if not is_continuous:
                    st.warning(f"⚠️ 栋舍编号不连续, 缺失编号: {', '.join(map(str, missing))}")
                
            except Exception as e:
                st.error(f"文件解析失败: {str(e)}")
    
    st.markdown("---")
    st.subheader("📋 数据概览")
    
    if st.session_state.env_data is not None and st.session_state.prod_data is not None:
        env_df = st.session_state.env_data
        prod_df = st.session_state.prod_data
        
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            st.metric("环境监测记录数", len(env_df))
        with col4:
            st.metric("生产数据记录数", len(prod_df))
        with col5:
            env_barns = env_df['栋舍编号'].nunique()
            st.metric("环境监测栋舍数", env_barns)
        with col6:
            prod_barns = prod_df['栋舍编号'].nunique()
            st.metric("生产数据栋舍数", prod_barns)
        
        st.success("✅ 数据导入完成, 可以开始使用各功能模块")
    else:
        st.info("请上传环境监测数据和生产数据以开始使用")


def comfort_page(livestock_type):
    """环境舒适度评价页面"""
    st.header("🌡️ 环境舒适度评价")
    st.markdown("---")
    
    if st.session_state.env_data is None:
        st.warning("请先导入环境监测数据")
        return
    
    env_df = st.session_state.env_data.copy()
    env_df['时间戳'] = pd.to_datetime(env_df['时间戳'])
    
    status_df = evaluate_barn_status(env_df, livestock_type)
    
    st.subheader("📊 各栋舍当前状态")
    
    display_cols = ['栋舍编号', '更新时间', '温度(℃)', '湿度(%)', 'THI指数', 'THI状态', 
                    '氨气(ppm)', '氨气状态', 'CO2(ppm)', 'CO2状态', '通风状态', '综合状态']
    display_df = status_df[display_cols].copy()
    
    st.dataframe(
        style_dataframe(display_df),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("📈 THI趋势分析")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        barns = sorted(status_df['栋舍编号'].unique().tolist())
        selected_barn = st.selectbox("选择栋舍", barns, key="thi_barn")
    
    with col2:
        barn_data = env_df[env_df['栋舍编号'] == selected_barn].sort_values('时间戳').copy()
        barn_data['THI'] = barn_data.apply(
            lambda row: calculate_thi(row['温度'], row['湿度']), axis=1
        )
        
        thresholds = THI_THRESHOLDS.get(livestock_type, THI_THRESHOLDS['肉鸡'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=barn_data['时间戳'],
            y=barn_data['THI'],
            mode='lines',
            name='THI指数',
            line=dict(color='#4472C4', width=2)
        ))
        
        fig.add_hline(y=thresholds['正常'], line_dash="dash", line_color="green", 
                      annotation_text="正常阈值", annotation_position="right")
        fig.add_hline(y=thresholds['轻度热应激'], line_dash="dash", line_color="orange",
                      annotation_text="轻度应激", annotation_position="right")
        fig.add_hline(y=thresholds['中度热应激'], line_dash="dash", line_color="red",
                      annotation_text="中度应激", annotation_position="right")
        
        fig.update_layout(
            title=f'{selected_barn} THI趋势图',
            xaxis_title='时间',
            yaxis_title='THI指数',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("💨 通风效率评估")
    
    vent_results = []
    for barn_id in sorted(env_df['栋舍编号'].unique()):
        has_bad, periods = check_ventilation_efficiency(env_df, barn_id)
        vent_results.append({
            '栋舍编号': barn_id,
            '通风状态': '通风不良' if has_bad else '正常',
            '异常时段数': len(periods)
        })
    
    vent_df = pd.DataFrame(vent_results)
    st.dataframe(style_dataframe(vent_df, status_cols=['通风状态']), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("🚨 有害气体分级")
    
    gas_col1, gas_col2 = st.columns(2)
    
    with gas_col1:
        st.markdown("**氨气分级标准**")
        st.info("正常: ≤20ppm\n\n一级告警: >20ppm 且 ≤35ppm\n\n二级告警: >35ppm")
    
    with gas_col2:
        st.markdown("**CO2分级标准**")
        st.info("正常: ≤3000ppm\n\n一级告警: >3000ppm 且 ≤5000ppm\n\n二级告警: >5000ppm")


def anomaly_page():
    """异常检测页面"""
    st.header("⚠️ 异常检测")
    st.markdown("---")
    
    if st.session_state.env_data is None:
        st.warning("请先导入环境监测数据")
        return
    
    env_df = st.session_state.env_data.copy()
    env_df['时间戳'] = pd.to_datetime(env_df['时间戳'])
    
    with st.spinner("正在进行异常检测..."):
        anomaly_summary = summarize_anomalies(env_df)
        sensor_drift_df = check_all_sensor_drift(env_df)
    
    st.subheader("📊 异常统计汇总")
    
    display_summary = anomaly_summary.copy()
    st.dataframe(display_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📈 时序异常检测")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        barns = sorted(env_df['栋舍编号'].unique().tolist())
        selected_barn = st.selectbox("选择栋舍", barns, key="anomaly_barn")
    
    with col2:
        params = ['温度', '湿度', '氨气浓度(ppm)', 'CO2浓度(ppm)', '光照强度(lux)', '噪声(dB)']
        selected_param = st.selectbox("选择参数", params, key="anomaly_param")
    
    with col3:
        st.write("检测方法")
        show_absolute = st.checkbox("绝对阈值", value=True)
        show_statistical = st.checkbox("统计法(3σ)", value=True)
        show_mutation = st.checkbox("突变检测", value=True)
    
    barn_anomaly = detect_anomalies_for_barn_param(env_df, selected_param, selected_barn)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=barn_anomaly['时间戳'],
        y=barn_anomaly[selected_param],
        mode='lines',
        name=selected_param,
        line=dict(color='#4472C4', width=1.5)
    ))
    
    anomaly_mask = pd.Series([False] * len(barn_anomaly))
    if show_absolute:
        anomaly_mask = anomaly_mask | barn_anomaly[f'{selected_param}_绝对阈值异常']
    if show_statistical:
        anomaly_mask = anomaly_mask | barn_anomaly[f'{selected_param}_统计异常']
    if show_mutation:
        anomaly_mask = anomaly_mask | barn_anomaly[f'{selected_param}_突变异常']
    
    anomaly_points = barn_anomaly[anomaly_mask]
    if not anomaly_points.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_points['时间戳'],
            y=anomaly_points[selected_param],
            mode='markers',
            name='异常点',
            marker=dict(color='red', size=8, symbol='circle'),
            hovertemplate='%{x}<br>%{y}'
        ))
    
    fig.update_layout(
        title=f'{selected_barn} {selected_param} 异常检测',
        xaxis_title='时间',
        yaxis_title=selected_param,
        height=450,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("绝对阈值异常", int(barn_anomaly[f'{selected_param}_绝对阈值异常'].sum()))
    with col5:
        st.metric("统计法异常(3σ)", int(barn_anomaly[f'{selected_param}_统计异常'].sum()))
    with col6:
        st.metric("突变检测异常", int(barn_anomaly[f'{selected_param}_突变异常'].sum()))
    
    st.markdown("---")
    st.subheader("🔧 传感器漂移检测")
    
    if not sensor_drift_df.empty:
        st.warning("⚠️ 检测到疑似传感器故障")
        st.dataframe(style_dataframe(sensor_drift_df, status_cols=['状态']), use_container_width=True, hide_index=True)
    else:
        st.success("✅ 所有传感器运行正常，未检测到漂移")


def generate_suggestions(risk_level, triggered_signals):
    """根据风险等级和触发信号生成建议措施"""
    suggestions = []
    if risk_level in ['低风险', '中风险', '高风险']:
        suggestions.append('加强栋舍巡查频率，密切观察动物状态')
    if '采食量异常' in triggered_signals:
        suggestions.append('检查饲料质量和投喂系统，排查疾病诱因')
    if '饮水量异常' in triggered_signals:
        suggestions.append('检查供水系统，确保饮水清洁充足')
    if '死淘异常' in triggered_signals:
        suggestions.append('对死淘动物进行剖检，排查传染病风险')
    if '环境恶化' in triggered_signals:
        suggestions.append('立即改善通风降温措施，降低热应激')
    if '空气质量异常' in triggered_signals:
        suggestions.append('加强通风换气，清理粪便，降低氨气浓度')
    if risk_level == '中风险':
        suggestions.append('咨询兽医，考虑预防性投药')
    if risk_level == '高风险':
        suggestions.append('隔离观察疑似患病动物，启动应急预案')
        suggestions.append('对栋舍进行全面消毒')
    return '；'.join(suggestions) if suggestions else '正常巡查'


def auto_generate_warning_tickets(risk_df):
    """根据风险评估结果自动生成预警工单（含去重）"""
    tickets = st.session_state.warning_tickets
    today = datetime.now().date()
    
    existing_keys = set()
    for t in tickets:
        existing_keys.add((t['栋舍编号'], pd.Timestamp(t['生成时间']).date()))
    
    for _, row in risk_df.iterrows():
        barn_id = row['栋舍编号']
        risk_level = row['风险等级']
        risk_score = row['风险评分']
        triggered_signals = row['触发信号']
        
        if risk_level in ['低风险', '中风险', '高风险']:
            key = (barn_id, today)
            if key not in existing_keys:
                new_ticket = {
                    '工单ID': f"WARN{int(datetime.now().timestamp() * 1000)}",
                    '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    '栋舍编号': barn_id,
                    '风险等级': risk_level,
                    '风险评分': risk_score,
                    '触发信号': ', '.join(triggered_signals) if triggered_signals else '无',
                    '建议措施': generate_suggestions(risk_level, triggered_signals),
                    '处置状态': '待处理',
                    '处置备注': '',
                    '关闭时间': None,
                }
                tickets.append(new_ticket)
    
    st.session_state.warning_tickets = tickets


def calculate_warning_statistics():
    """计算预警统计指标"""
    tickets = st.session_state.warning_tickets
    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    week_new = 0
    pending_count = 0
    total_closed_hours = 0
    closed_count = 0
    week_total = 0
    week_closed = 0
    
    for t in tickets:
        create_time = pd.Timestamp(t['生成时间'])
        if create_time >= week_start:
            week_total += 1
            week_new += 1
        
        if t['处置状态'] == '待处理':
            pending_count += 1
        
        if t['处置状态'] == '已关闭' and t['关闭时间']:
            close_time = pd.Timestamp(t['关闭时间'])
            duration = (close_time - create_time).total_seconds() / 3600
            total_closed_hours += duration
            closed_count += 1
            if create_time >= week_start:
                week_closed += 1
    
    avg_duration = round(total_closed_hours / closed_count, 1) if closed_count > 0 else 0
    week_close_rate = round((week_closed / week_total) * 100, 1) if week_total > 0 else 0
    
    return {
        'week_new': week_new,
        'pending_count': pending_count,
        'avg_duration': avg_duration,
        'week_close_rate': week_close_rate,
    }


def style_ticket_row(row):
    """根据工单状态设置行底色"""
    status = row['处置状态']
    if status == '待处理':
        return ['background-color: #FFC7CE'] * len(row)
    elif status == '处理中':
        return ['background-color: #FFEB9C'] * len(row)
    elif status == '已关闭':
        return ['background-color: #C6EFCE'] * len(row)
    else:
        return [''] * len(row)


def export_tickets_csv():
    """导出工单数据为CSV"""
    tickets = st.session_state.warning_tickets
    if not tickets:
        return None
    
    df = pd.DataFrame(tickets)
    display_cols = ['工单ID', '生成时间', '栋舍编号', '风险等级', '风险评分', 
                    '触发信号', '建议措施', '处置状态', '处置备注', '关闭时间']
    df = df[display_cols]
    
    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()


def disease_warning_page(livestock_type, total_livestock):
    """疾病预警页面"""
    st.header("🏥 疾病预警模型")
    st.markdown("---")
    
    if st.session_state.env_data is None or st.session_state.prod_data is None:
        st.warning("请先导入环境监测数据和生产数据")
        return
    
    env_df = st.session_state.env_data.copy()
    prod_df = st.session_state.prod_data.copy()
    env_df['时间戳'] = pd.to_datetime(env_df['时间戳'])
    prod_df['时间戳'] = pd.to_datetime(prod_df['时间戳'])
    
    risk_df = get_all_barns_risk(env_df, prod_df, livestock_type, total_livestock)
    
    st.subheader("📊 各栋舍疾病风险评分")
    
    display_risk = risk_df[['栋舍编号', '风险评分', '风险等级', '触发信号', '采食异常', '饮水异常', '死淘异常', '环境恶化', '空气质量异常']].copy()
    display_risk['触发信号'] = display_risk['触发信号'].apply(lambda x: ', '.join(x) if x else '无')
    
    st.dataframe(
        style_dataframe(display_risk, status_cols=['风险等级']),
        use_container_width=True,
        hide_index=True
    )
    
    auto_generate_warning_tickets(risk_df)
    
    st.markdown("---")
    st.subheader("📈 风险评分时间线")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        barns = sorted(risk_df['栋舍编号'].unique().tolist())
        selected_barn = st.selectbox("选择栋舍", barns, key="risk_barn")
    
    with col2:
        timeline_df = calculate_daily_risk_timeline(env_df, prod_df, selected_barn, livestock_type, total_livestock)
        
        if not timeline_df.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timeline_df['日期'],
                y=timeline_df['风险评分'],
                mode='lines+markers',
                name='风险评分',
                line=dict(color='#4472C4', width=2),
                marker=dict(size=8)
            ))
            
            fig.add_hline(y=0.4, line_dash="dash", line_color="yellow", 
                          annotation_text="低风险阈值(0.4)", annotation_position="right")
            fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                          annotation_text="中风险阈值(0.6)", annotation_position="right")
            fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                          annotation_text="高风险阈值(0.8)", annotation_position="right")
            
            fig.update_layout(
                title=f'{selected_barn} 疾病风险评分时间线',
                xaxis_title='日期',
                yaxis_title='风险评分',
                yaxis_range=[0, 1],
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无足够数据绘制时间线")
    
    st.markdown("---")
    st.subheader("📋 风险因子权重说明")
    
    weight_df = pd.DataFrame([
        {'信号': '采食量异常(连续2天下降>10%)', '权重': WEIGHTS['feed_intake']},
        {'信号': '饮水量异常(连续2天偏离均值>20%)', '权重': WEIGHTS['water_intake']},
        {'信号': '日死淘率异常(>0.5%)', '权重': WEIGHTS['mortality']},
        {'信号': '环境恶化(中度热应激>6小时)', '权重': WEIGHTS['environment']},
        {'信号': '空气质量异常(氨气>25ppm超过4小时)', '权重': WEIGHTS['air_quality']},
    ])
    
    st.table(weight_df)
    
    st.markdown("---")
    st.header("🚨 预警处置")
    
    stats = calculate_warning_statistics()
    
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns([1, 1, 1, 1, 1])
    with stat_col1:
        st.metric("本周新增预警数", stats['week_new'])
    with stat_col2:
        st.metric("待处理工单数", stats['pending_count'])
    with stat_col3:
        st.metric("平均处置时长(小时)", stats['avg_duration'])
    with stat_col4:
        st.metric("本周关闭率(%)", stats['week_close_rate'])
    with stat_col5:
        csv_data = export_tickets_csv()
        if csv_data:
            filename = f"预警工单_{datetime.now().strftime('%Y%m%d')}.csv"
            st.download_button(
                label="📥 导出工单CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                **_st_btn_kwargs()
            )
        else:
            st.button("📥 导出工单CSV", disabled=True, **_st_btn_kwargs())
    
    st.markdown("---")
    st.subheader("📋 预警工单列表")
    
    tickets = st.session_state.warning_tickets
    if not tickets:
        st.info("暂无预警工单")
        return
    
    sorted_tickets = sorted(tickets, key=lambda x: x['生成时间'], reverse=True)
    
    display_cols = ['工单ID', '生成时间', '栋舍编号', '风险等级', '风险评分', 
                    '触发信号', '建议措施', '处置状态', '处置备注', '关闭时间']
    
    for idx, ticket in enumerate(sorted_tickets):
        with st.container():
            status = ticket['处置状态']
            if status == '待处理':
                bg_color = '#FFC7CE'
            elif status == '处理中':
                bg_color = '#FFEB9C'
            elif status == '已关闭':
                bg_color = '#C6EFCE'
            else:
                bg_color = '#FFFFFF'
            
            st.markdown(
                f"""
                <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: {bg_color};">
                </div>
                """,
                unsafe_allow_html=True
            )
            
            col_id, col_time, col_barn, col_level, col_score, col_signals = st.columns([2, 2, 1, 1, 1, 3])
            with col_id:
                st.markdown(f"**工单ID**: {ticket['工单ID']}")
            with col_time:
                st.markdown(f"**生成时间**: {ticket['生成时间']}")
            with col_barn:
                st.markdown(f"**栋舍**: {ticket['栋舍编号']}")
            with col_level:
                st.markdown(f"**等级**: {ticket['风险等级']}")
            with col_score:
                st.markdown(f"**评分**: {ticket['风险评分']}")
            with col_signals:
                st.markdown(f"**触发信号**: {ticket['触发信号']}")
            
            st.markdown(f"**建议措施**: {ticket['建议措施']}")
            
            edit_col1, edit_col2, edit_col3, edit_col4 = st.columns([1, 3, 1, 1])
            with edit_col1:
                new_status = st.selectbox(
                    "处置状态",
                    ["待处理", "处理中", "已关闭"],
                    index=["待处理", "处理中", "已关闭"].index(ticket['处置状态']),
                    key=f"status_{ticket['工单ID']}"
                )
            with edit_col2:
                new_remark = st.text_input(
                    "处置备注",
                    value=ticket['处置备注'],
                    placeholder="请记录实际采取的措施...",
                    key=f"remark_{ticket['工单ID']}"
                )
            with edit_col3:
                if ticket['关闭时间']:
                    st.markdown(f"**关闭时间**: {ticket['关闭时间']}")
                else:
                    st.markdown("**关闭时间**: 未关闭")
            with edit_col4:
                if st.button("更新状态", key=f"update_{ticket['工单ID']}", type="primary"):
                    for t in st.session_state.warning_tickets:
                        if t['工单ID'] == ticket['工单ID']:
                            t['处置状态'] = new_status
                            t['处置备注'] = new_remark
                            if new_status == '已关闭' and not t['关闭时间']:
                                t['关闭时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            elif new_status != '已关闭':
                                t['关闭时间'] = None
                            break
                    st.rerun()


def mortality_page(total_livestock):
    """死亡率分析页面"""
    st.header("💀 死亡率分析")
    st.markdown("---")
    
    if st.session_state.prod_data is None:
        st.warning("请先导入生产数据")
        return
    
    prod_df = st.session_state.prod_data.copy()
    prod_df['时间戳'] = pd.to_datetime(prod_df['时间戳'])
    
    env_df = st.session_state.env_data.copy() if st.session_state.env_data is not None else None
    if env_df is not None:
        env_df['时间戳'] = pd.to_datetime(env_df['时间戳'])
    
    st.subheader("📈 日龄-累计死淘率曲线")
    
    barns = sorted(prod_df['栋舍编号'].unique().tolist())
    selected_barns = st.multiselect("选择栋舍(支持多批次叠加对比)", barns, default=barns[:2] if len(barns) >= 2 else barns)
    
    if selected_barns:
        fig = go.Figure()
        
        colors = px.colors.qualitative.Plotly
        
        for i, barn_id in enumerate(selected_barns):
            barn_data = calculate_cumulative_mortality(prod_df, barn_id, total_livestock)
            
            if not barn_data.empty:
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter(
                    x=barn_data['日龄'],
                    y=barn_data['累计死淘率(%)'],
                    mode='lines',
                    name=f'{barn_id} 累计',
                    line=dict(color=color, width=2),
                    yaxis='y'
                ))
                
                fig.add_trace(go.Bar(
                    x=barn_data['日龄'],
                    y=barn_data['日死淘率(%)'],
                    name=f'{barn_id} 日死淘',
                    opacity=0.5,
                    marker_color=color,
                    yaxis='y2',
                    width=0.8
                ))
                
                _, inflection_points = detect_mortality_inflection_points(prod_df, barn_id, total_livestock)
                
                for point in inflection_points[:5]:
                    fig.add_vline(
                        x=point['日龄'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"拐点(日龄{point['日龄']})",
                        annotation_position="top"
                    )
        
        fig.update_layout(
            title='日龄-死淘率曲线对比',
            xaxis_title='日龄',
            yaxis=dict(
                title='累计死淘率(%)',
                side='left'
            ),
            yaxis2=dict(
                title='日死淘率(%)',
                side='right',
                overlaying='y',
                showgrid=False
            ),
            height=500,
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 拐点归因分析")
    
    if selected_barns and env_df is not None:
        for barn_id in selected_barns[:2]:
            _, inflection_points = detect_mortality_inflection_points(prod_df, barn_id, total_livestock)
            
            if inflection_points:
                st.markdown(f"**{barn_id} 拐点分析**")
                
                for point in inflection_points[:3]:
                    with st.expander(f"日龄 {point['日龄']} - 环比增长 {point['环比增长(%)']}%"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"日死淘率: {point['日死淘率(%)']}%")
                            st.write(f"累计死淘率: {point['累计死淘率(%)']}%")
                        
                        with col2:
                            env_summary = get_env_summary_around_date(
                                env_df, barn_id, point['日期'], days_before=1, days_after=1
                            )
                            if env_summary:
                                st.write("同期环境摘要:")
                                for k, v in env_summary.items():
                                    st.write(f"- {k}: {v}")
                            else:
                                st.write("无同期环境数据")
            else:
                st.info(f"{barn_id} 未检测到明显拐点")


def energy_page():
    """能耗分析页面"""
    st.header("⚡ 能耗关联分析")
    st.markdown("---")
    
    if st.session_state.env_data is None:
        st.warning("请先导入环境监测数据(包含能耗列)")
        return
    
    env_df = st.session_state.env_data.copy()
    prod_df = st.session_state.prod_data.copy() if st.session_state.prod_data is not None else None
    
    env_df['时间戳'] = pd.to_datetime(env_df['时间戳'])
    if prod_df is not None:
        prod_df['时间戳'] = pd.to_datetime(prod_df['时间戳'])
    
    if not has_energy_data(env_df):
        st.warning("数据中未包含能耗列(通风能耗kWh/加热能耗kWh/降温能耗kWh)")
        return
    
    st.subheader("📊 能耗与CO2散点图")
    
    barns = sorted(env_df['栋舍编号'].unique().tolist())
    selected_barn = st.selectbox("选择栋舍", barns, key="energy_barn")
    
    energy_co2_df = calculate_energy_co2_relation(env_df, selected_barn)
    
    if not energy_co2_df.empty:
        fig = px.scatter(
            energy_co2_df,
            x='通风能耗kWh',
            y='CO2浓度(ppm)',
            size='温度' if '温度' in energy_co2_df.columns else None,
            color='湿度' if '湿度' in energy_co2_df.columns else None,
            title=f'{selected_barn} 通风能耗 vs CO2浓度',
            labels={'通风能耗kWh': '通风能耗 (kWh)', 'CO2浓度(ppm)': 'CO2浓度 (ppm)'}
        )
        
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        corr = energy_co2_df['通风能耗kWh'].corr(energy_co2_df['CO2浓度(ppm)'])
        st.info(f"通风能耗与CO2浓度相关系数: {corr:.3f}")
        if corr < -0.5:
            st.success("✅ 能耗增加有效降低了CO2浓度，通风系统运行正常")
        elif corr > 0:
            st.warning("⚠️ 能耗增加未有效降低CO2，建议检查通风系统效率")
    else:
        st.info("暂无足够数据")
    
    st.markdown("---")
    st.subheader("💧 吨水能耗等效指标")
    
    if prod_df is not None:
        energy_summary = get_energy_summary(env_df, prod_df)
        if not energy_summary.empty:
            st.dataframe(energy_summary, use_container_width=True, hide_index=True)
        else:
            st.info("暂无可对比数据")
    else:
        st.info("请导入生产数据以计算吨水能耗")
    
    st.markdown("---")
    st.subheader("📅 月度能耗趋势")
    
    monthly_energy = calculate_monthly_energy_trend(env_df, selected_barn)
    
    if not monthly_energy.empty:
        fig = go.Figure()
        
        energy_cols = [col for col in ['通风能耗kWh', '加热能耗kWh', '降温能耗kWh'] if col in monthly_energy.columns]
        
        for col in energy_cols:
            fig.add_trace(go.Bar(
                x=monthly_energy['月份'],
                y=monthly_energy[col],
                name=col
            ))
        
        fig.update_layout(
            title=f'{selected_barn} 月度能耗趋势',
            xaxis_title='月份',
            yaxis_title='能耗 (kWh)',
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def comparison_page(livestock_type, total_livestock):
    """多栋舍对比页面"""
    st.header("📊 多栋舍对比")
    st.markdown("---")
    
    if st.session_state.env_data is None or st.session_state.prod_data is None:
        st.warning("请先导入环境监测数据和生产数据")
        return
    
    env_df = st.session_state.env_data.copy()
    prod_df = st.session_state.prod_data.copy()
    env_df['时间戳'] = pd.to_datetime(env_df['时间戳'])
    prod_df['时间戳'] = pd.to_datetime(prod_df['时间戳'])
    
    barns = sorted(list(set(env_df['栋舍编号'].unique()) | set(prod_df['栋舍编号'].unique())))
    
    st.subheader("🔧 选择对比栋舍")
    selected_barns = st.multiselect(
        "选择2-4个栋舍进行对比",
        barns,
        default=barns[:2] if len(barns) >= 2 else barns,
        max_selections=4
    )
    
    if len(selected_barns) < 2:
        st.info("请至少选择2个栋舍进行对比")
        return
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["环境参数对比", "疾病风险对比", "生产指标对比"])
    
    with tab1:
        st.subheader("🌡️ 环境参数均值对比 (雷达图)")
        
        env_means = calculate_env_means(env_df, selected_barns)
        
        if not env_means.empty:
            radar_params = ['温度', '湿度', '氨气浓度(ppm)', 'CO2浓度(ppm)', '噪声(dB)']
            
            radar_data = []
            for barn_id in selected_barns:
                barn_row = env_means[env_means['栋舍编号'] == barn_id]
                if not barn_row.empty:
                    radar_data.append(barn_row.iloc[0])
            
            fig = go.Figure()
            
            for data_row in radar_data:
                values = []
                for param in radar_params:
                    if param in data_row:
                        values.append(data_row[param])
                    else:
                        values.append(0)
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=radar_params,
                    fill='toself',
                    name=str(data_row['栋舍编号']) + '栋'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )
                ),
                showlegend=True,
                height=500,
                title='环境参数雷达图对比'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 环境参数详细对比")
            st.dataframe(env_means, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("🏥 疾病风险评分对比")
        
        from utils.disease_warning import calculate_daily_risk_timeline
        
        fig = go.Figure()
        
        for barn_id in selected_barns:
            timeline_df = calculate_daily_risk_timeline(env_df, prod_df, barn_id, livestock_type, total_livestock)
            
            if not timeline_df.empty:
                fig.add_trace(go.Scatter(
                    x=timeline_df['日期'],
                    y=timeline_df['风险评分'],
                    mode='lines+markers',
                    name=f'{barn_id}',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        fig.add_hline(y=0.4, line_dash="dash", line_color="yellow", 
                      annotation_text="低风险", annotation_position="right")
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                      annotation_text="中风险", annotation_position="right")
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                      annotation_text="高风险", annotation_position="right")
        
        fig.update_layout(
            title='疾病风险评分对比',
            xaxis_title='日期',
            yaxis_title='风险评分',
            yaxis_range=[0, 1],
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🐔 生产指标对比")
        
        prod_comp = calculate_production_comparison(prod_df, selected_barns)
        
        if not prod_comp.empty:
            st.dataframe(prod_comp, use_container_width=True, hide_index=True)
            
            metrics = ['总采食量(kg)', '总饮水量(L)', '总死淘数(只)', '存活率(%)', '体重增长(kg)', '料肉比']
            selected_metric = st.selectbox("选择对比指标", metrics)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[str(b) + '栋' for b in prod_comp['栋舍编号']],
                y=prod_comp[selected_metric],
                marker_color=px.colors.qualitative.Plotly[:len(prod_comp)],
                text=prod_comp[selected_metric],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f'各栋舍 {selected_metric} 对比',
                yaxis_title=selected_metric,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("💡 差异分析与建议")
    
    env_means = calculate_env_means(env_df, selected_barns)
    suggestions = find_largest_difference(env_means, prod_df, selected_barns)
    
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            st.info(f"💡 建议 {i}: {suggestion}")
    else:
        st.success("各栋舍环境参数差异较小，整体运行良好")


DEFAULT_FEED_UNIT_PRICE = 2.8
OTHER_COST_PER_BIRD = 2.0
INITIAL_WEIGHT_KG = 0.04


def _get_feed_unit_price():
    """获取饲料单价，优先使用session_state中的全局值"""
    return st.session_state.get('global_feed_unit_price', DEFAULT_FEED_UNIT_PRICE)


def _get_barn_list():
    barns = set()
    if st.session_state.env_data is not None:
        barns.update(st.session_state.env_data['栋舍编号'].unique().tolist())
    if st.session_state.prod_data is not None:
        barns.update(st.session_state.prod_data['栋舍编号'].unique().tolist())
    return sorted(barns)


def _next_batch_id():
    batches = st.session_state.batches
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


def _calculate_feed_cost(barn_id, start_date, end_date, use_predicted=False):
    prod_df = st.session_state.prod_data
    if prod_df is None:
        return None, 0.0, True, None

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
        return None, 0.0, True, None

    total_feed = barn_data['日采食量(kg)'].sum()
    feed_unit_price = _get_feed_unit_price()
    feed_cost = total_feed * feed_unit_price

    predicted_feed_cost = None
    predicted_unit_price = st.session_state.get('global_feed_unit_price_predicted', None)
    if use_predicted and predicted_unit_price is not None:
        predicted_feed_cost = total_feed * predicted_unit_price
    elif predicted_unit_price is not None:
        predicted_feed_cost = total_feed * predicted_unit_price

    total_days = (end_ts - start_ts).days + 1
    covered_days = barn_data['时间戳'].dt.date.nunique()
    coverage = covered_days / total_days if total_days > 0 else 0
    data_incomplete = coverage < 0.8

    return total_feed, feed_cost, data_incomplete, predicted_feed_cost


def _calculate_batch_health_score(batch, livestock_type='肉鸡', total_livestock=10000):
    if batch.get('status') != '已出栏':
        return None

    barn_id = batch['barn_id']
    start_date = batch['start_date']
    end_date = batch['slaughter_date']

    env_df = st.session_state.env_data
    prod_df = st.session_state.prod_data

    if env_df is None or prod_df is None:
        return None

    env_df_copy = env_df.copy()
    prod_df_copy = prod_df.copy()
    env_df_copy['时间戳'] = pd.to_datetime(env_df_copy['时间戳'])
    prod_df_copy['时间戳'] = pd.to_datetime(prod_df_copy['时间戳'])

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    env_filtered = env_df_copy[
        (env_df_copy['栋舍编号'] == barn_id) &
        (env_df_copy['时间戳'] >= start_ts) &
        (env_df_copy['时间戳'] <= end_ts)
    ]
    prod_filtered = prod_df_copy[
        (prod_df_copy['栋舍编号'] == barn_id) &
        (prod_df_copy['时间戳'] >= start_ts) &
        (prod_df_copy['时间戳'] <= end_ts)
    ]

    if env_filtered.empty or prod_filtered.empty:
        return None

    from utils.disease_warning import calculate_daily_risk_timeline

    timeline_df = calculate_daily_risk_timeline(
        env_filtered, prod_filtered, barn_id, livestock_type, total_livestock
    )

    if timeline_df.empty:
        return None

    risk_scores = timeline_df['风险评分'].values
    avg_risk = np.mean(risk_scores)
    peak_risk = np.max(risk_scores)

    return {
        'avg_risk': round(avg_risk, 2),
        'peak_risk': round(peak_risk, 2),
    }


def _calculate_economic_indicators(batch):
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

    total_feed, feed_cost, data_incomplete, predicted_feed_cost = _calculate_feed_cost(barn_id, start_date, end_date)

    chick_cost = chick_count * chick_price
    other_cost = chick_count * OTHER_COST_PER_BIRD
    total_cost = chick_cost + feed_cost + other_cost

    total_revenue = slaughter_count * avg_weight * sale_price

    profit = total_revenue - total_cost

    total_weight_gain = slaughter_count * avg_weight - chick_count * INITIAL_WEIGHT_KG
    fcr = total_feed / total_weight_gain if total_weight_gain > 0 and total_feed is not None else None

    survival_rate = (slaughter_count / chick_count) * 100 if chick_count > 0 else 0
    profit_per_bird = profit / slaughter_count if slaughter_count > 0 else 0

    profit_margin_pct = (profit / total_revenue * 100) if total_revenue > 0 else 0
    predicted_profit_margin_pct = None
    predicted_profit = None
    profit_margin_change_pct = None

    if predicted_feed_cost is not None:
        predicted_total_cost = chick_cost + predicted_feed_cost + other_cost
        predicted_profit = total_revenue - predicted_total_cost
        predicted_profit_margin_pct = (predicted_profit / total_revenue * 100) if total_revenue > 0 else 0
        profit_margin_change_pct = predicted_profit_margin_pct - profit_margin_pct

    return {
        'chick_cost': chick_cost,
        'feed_cost': feed_cost,
        'predicted_feed_cost': predicted_feed_cost,
        'other_cost': other_cost,
        'total_cost': total_cost,
        'total_revenue': total_revenue,
        'profit': profit,
        'predicted_profit': predicted_profit,
        'profit_margin_pct': profit_margin_pct,
        'predicted_profit_margin_pct': predicted_profit_margin_pct,
        'profit_margin_change_pct': profit_margin_change_pct,
        'fcr': fcr,
        'survival_rate': survival_rate,
        'profit_per_bird': profit_per_bird,
        'predicted_profit_per_bird': (predicted_profit / slaughter_count) if (predicted_profit is not None and slaughter_count > 0) else None,
        'total_feed': total_feed,
        'data_incomplete': data_incomplete,
    }


def _get_sort_key(batch, sort_by):
    if sort_by == "添加时间":
        return 0
    elif sort_by == "成活率":
        if batch['status'] != '已出栏':
            return -1
        ind = _calculate_economic_indicators(batch)
        return ind['survival_rate'] if ind else -1
    elif sort_by == "料肉比":
        if batch['status'] != '已出栏':
            return 999
        ind = _calculate_economic_indicators(batch)
        return ind['fcr'] if ind and ind['fcr'] is not None else 999
    elif sort_by == "每只利润":
        if batch['status'] != '已出栏':
            return -999999
        ind = _calculate_economic_indicators(batch)
        return ind['profit_per_bird'] if ind else -999999
    return 0


def _export_batches_csv():
    batches = st.session_state.batches
    completed_batches = [b for b in batches if b['status'] == '已出栏']
    if not completed_batches:
        return None

    export_data = []
    for b in completed_batches:
        ind = _calculate_economic_indicators(b)
        health = _calculate_batch_health_score(b)
        row = {
            '批次编号': b['batch_id'],
            '栋舍': b['barn_id'],
            '进苗数量': b['chick_count'],
            '出栏数量': b['slaughter_count'],
            '苗鸡成本': round(ind['chick_cost'], 2) if ind else '',
            '饲料成本': round(ind['feed_cost'], 2) if ind else '',
            '其他成本': round(ind['other_cost'], 2) if ind else '',
            '总投入': round(ind['total_cost'], 2) if ind else '',
            '总收入': round(ind['total_revenue'], 2) if ind else '',
            '利润': round(ind['profit'], 2) if ind else '',
            '料肉比': round(ind['fcr'], 3) if ind and ind['fcr'] is not None else '',
            '成活率': round(ind['survival_rate'], 1) if ind else '',
            '每只利润': round(ind['profit_per_bird'], 2) if ind else '',
            '日均风险评分': health['avg_risk'] if health else '',
            '峰值风险评分': health['peak_risk'] if health else '',
        }
        export_data.append(row)

    df = pd.DataFrame(export_data)
    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()


def batch_management_page():
    st.header("📋 养殖批次管理与经济效益分析")
    st.markdown("---")

    current_feed_price = _get_feed_unit_price()
    predicted_feed_price = st.session_state.get('global_feed_unit_price_predicted', None)
    if current_feed_price != DEFAULT_FEED_UNIT_PRICE:
        info_text = f"💡 当前饲料单价: **{current_feed_price:.2f} 元/kg** (由「饲料配方模拟与成本优化」模块提供)"
        if predicted_feed_price is not None:
            info_text += f"\n\n🔮 预测下月饲料单价: **{predicted_feed_price:.2f} 元/kg**"
        st.info(info_text)
    else:
        info_text = f"💡 当前饲料单价: **{current_feed_price:.2f} 元/kg** (默认值，可在「饲料配方模拟与成本优化」模块中调整)"
        if predicted_feed_price is not None:
            info_text += f"\n\n🔮 预测下月饲料单价: **{predicted_feed_price:.2f} 元/kg**"
        st.info(info_text)
    
    st.markdown("---")

    barns = _get_barn_list()

    st.subheader("📝 批次信息录入")
    next_id = _next_batch_id()

    with st.form("add_batch_form"):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.text_input("批次编号", value=next_id, disabled=True)
            start_date = st.date_input("进苗日期", value=datetime.now(), key="batch_start_date")

        with col_f2:
            barn_options = barns if barns else ["暂无栋舍数据"]
            barn_id = st.selectbox("栋舍编号", barn_options, key="batch_barn_id")
            chick_count = st.number_input("进苗数量(只)", min_value=1, step=100, key="batch_chick_count")

        with col_f3:
            chick_price = st.number_input("苗鸡单价(元/只)", min_value=0.0, value=3.0, step=0.1, format="%.2f", key="batch_chick_price")
            target_days = st.number_input("目标出栏日龄", min_value=20, max_value=120, value=42, step=1, key="batch_target_days")
            target_weight = st.number_input("目标出栏体重(kg)", min_value=0.0, value=2.5, step=0.1, format="%.2f", key="batch_target_weight")

        submitted = st.form_submit_button("添加批次", type="primary")

        form_errors = []
        if submitted:
            if chick_count <= 0 or int(chick_count) != chick_count:
                form_errors.append("进苗数量必须为正整数")
            if target_days < 20 or target_days > 120:
                form_errors.append("目标出栏日龄必须在20到120天之间")
            if barn_id == "暂无栋舍数据":
                form_errors.append("请先上传包含栋舍信息的数据")

        if form_errors:
            for err in form_errors:
                st.error(f"❌ {err}")

        if submitted and not form_errors:
            new_batch = {
                'batch_id': next_id,
                'start_date': start_date,
                'barn_id': barn_id,
                'chick_count': int(chick_count),
                'chick_price': chick_price,
                'target_days': target_days,
                'target_weight': target_weight,
                'status': '养殖中',
                'slaughter_date': None,
                'slaughter_count': None,
                'avg_slaughter_weight': None,
                'sale_price': None,
                'created_at': datetime.now(),
            }
            st.session_state.batches.append(new_batch)
            st.success(f"✅ 批次 {next_id} 添加成功!")
            st.rerun()

    st.markdown("---")
    st.subheader("📦 批次列表")

    batches = st.session_state.batches
    if not batches:
        st.info("暂无批次信息，请先添加批次")
        return

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        status_filter = st.selectbox(
            "状态筛选",
            ["全部", "养殖中", "已出栏"],
            key="batch_status_filter"
        )
    with col_filter2:
        sort_by = st.selectbox(
            "排序方式",
            ["添加时间", "成活率", "料肉比", "每只利润"],
            key="batch_sort_by"
        )

    if status_filter != "全部":
        display_batches = [b for b in batches if b['status'] == status_filter]
    else:
        display_batches = batches.copy()

    if sort_by != "添加时间":
        if sort_by == "料肉比":
            display_batches.sort(key=lambda b: _get_sort_key(b, sort_by), reverse=False)
        else:
            display_batches.sort(key=lambda b: _get_sort_key(b, sort_by), reverse=True)

    for i, batch in enumerate(display_batches):
        original_idx = next((idx for idx, b in enumerate(batches) if b['batch_id'] == batch['batch_id']), i)

        with st.container():
            status = batch['status']
            if status == '养殖中':
                border_color = '#4472C4'
                status_badge = '🔵 养殖中'
            else:
                border_color = '#70AD47'
                status_badge = '🟢 已出栏'

            st.markdown(
                f'<div style="padding:12px;border:2px solid {border_color};border-radius:8px;margin-bottom:10px;">'
                f'<span style="font-size:18px;font-weight:bold;">{batch["batch_id"]}</span>'
                f'&nbsp;&nbsp;<span style="font-size:14px;color:#666;">{status_badge}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.write(f"**栋舍**: {batch['barn_id']}")
                st.write(f"**进苗日期**: {batch['start_date']}")
            with col_b2:
                st.write(f"**进苗数量**: {batch['chick_count']} 只")
                st.write(f"**苗鸡单价**: {batch['chick_price']:.2f} 元/只")
            with col_b3:
                st.write(f"**目标出栏日龄**: {batch['target_days']} 天")
                st.write(f"**目标出栏体重**: {batch['target_weight']:.2f} kg")
            with col_b4:
                if status == '已出栏':
                    st.write(f"**出栏日期**: {batch['slaughter_date']}")
                    st.write(f"**出栏数量**: {batch['slaughter_count']} 只")

            if status == '养殖中':
                with st.expander(f"📝 登记出栏 [{batch['batch_id']}]"):
                    with st.form(f"slaughter_form_{batch['batch_id']}"):
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            s_date = st.date_input(
                                "实际出栏日期",
                                value=datetime.now(),
                                key=f"s_date_{batch['batch_id']}"
                            )
                            s_count = st.number_input(
                                "实际出栏数量(只)",
                                min_value=1,
                                step=10,
                                key=f"s_count_{batch['batch_id']}"
                            )
                        with sc2:
                            s_weight = st.number_input(
                                "实际平均出栏体重(kg)",
                                min_value=0.0,
                                value=2.5,
                                step=0.1,
                                format="%.2f",
                                key=f"s_weight_{batch['batch_id']}"
                            )
                            s_price = st.number_input(
                                "毛鸡销售单价(元/kg)",
                                min_value=0.0,
                                value=10.0,
                                step=0.1,
                                format="%.2f",
                                key=f"s_price_{batch['batch_id']}"
                            )

                        s_submitted = st.form_submit_button("确认出栏", type="primary")

                        s_errors = []
                        if s_submitted:
                            if s_date < batch['start_date']:
                                s_errors.append("出栏日期不能早于进苗日期")
                            if s_count > batch['chick_count']:
                                s_errors.append("出栏数量不能超过进苗数量")
                            if s_count <= 0 or int(s_count) != s_count:
                                s_errors.append("出栏数量必须为正整数")

                        if s_errors:
                            for err in s_errors:
                                st.error(f"❌ {err}")

                        if s_submitted and not s_errors:
                            st.session_state.batches[original_idx]['status'] = '已出栏'
                            st.session_state.batches[original_idx]['slaughter_date'] = s_date
                            st.session_state.batches[original_idx]['slaughter_count'] = int(s_count)
                            st.session_state.batches[original_idx]['avg_slaughter_weight'] = s_weight
                            st.session_state.batches[original_idx]['sale_price'] = s_price
                            st.success(f"✅ 批次 {batch['batch_id']} 出栏登记成功!")
                            st.rerun()

            if status == '已出栏':
                indicators = _calculate_economic_indicators(batch)
                if indicators:
                    st.markdown("**💰 经济指标**")
                    ic1, ic2, ic3 = st.columns(3)
                    incomplete_suffix = " (数据不完整,仅供参考)" if indicators['data_incomplete'] else ""

                    total_cost = indicators['total_cost']
                    chick_cost = indicators['chick_cost']
                    feed_cost = indicators['feed_cost']
                    other_cost = indicators['other_cost']

                    chick_pct = (chick_cost / total_cost * 100) if total_cost > 0 else 0
                    feed_pct = (feed_cost / total_cost * 100) if total_cost > 0 else 0
                    other_pct = (other_cost / total_cost * 100) if total_cost > 0 else 0

                    with ic1:
                        st.metric("总收入", f"{indicators['total_revenue']:.2f} 元")
                        st.markdown(f"**苗鸡成本**: {chick_cost:.2f} 元, 占比{chick_pct:.1f}%")
                        st.markdown(f"**饲料成本**{incomplete_suffix}: {feed_cost:.2f} 元, 占比{feed_pct:.1f}%")
                        if indicators['predicted_feed_cost'] is not None:
                            pred_feed = indicators['predicted_feed_cost']
                            pred_pct = (pred_feed / (chick_cost + pred_feed + other_cost) * 100) if (chick_cost + pred_feed + other_cost) > 0 else 0
                            feed_diff_pct = ((pred_feed - feed_cost) / feed_cost * 100) if feed_cost > 0 else 0
                            arrow = "⬆️" if feed_diff_pct > 0 else "⬇️"
                            st.markdown(f"**预测饲料成本**{incomplete_suffix}: {pred_feed:.2f} 元 {arrow}{abs(feed_diff_pct):.1f}%, 占比{pred_pct:.1f}%")
                        st.markdown(f"**其他成本**: {other_cost:.2f} 元, 占比{other_pct:.1f}%")
                    with ic2:
                        total_cost_label = f"**总投入**{incomplete_suffix}"
                        st.markdown(f"{total_cost_label}: {total_cost:.2f} 元")
                        profit_val = indicators['profit']
                        profit_label = f"**利润**{incomplete_suffix}"
                        st.markdown(f"{profit_label}: {profit_val:.2f} 元 ({'盈利' if profit_val >= 0 else '亏损'})")
                        if indicators['predicted_profit'] is not None:
                            pred_profit = indicators['predicted_profit']
                            profit_diff = pred_profit - profit_val
                            arrow = "⬆️" if profit_diff >= 0 else "⬇️"
                            st.markdown(f"**预测利润**{incomplete_suffix}: {pred_profit:.2f} 元 {arrow}{abs(profit_diff):.2f}元")
                        if indicators['fcr'] is not None:
                            fcr_label = f"**料肉比**{incomplete_suffix}"
                            st.markdown(f"{fcr_label}: {indicators['fcr']:.3f}")
                        else:
                            st.markdown("**料肉比**: N/A")
                    with ic3:
                        st.markdown(f"**成活率**: {indicators['survival_rate']:.1f}%")
                        ppb_label = f"**每只利润**{incomplete_suffix}"
                        st.markdown(f"{ppb_label}: {indicators['profit_per_bird']:.2f} 元")
                        if indicators['predicted_profit_per_bird'] is not None:
                            pred_ppb = indicators['predicted_profit_per_bird']
                            ppb_diff = pred_ppb - indicators['profit_per_bird']
                            arrow = "⬆️" if ppb_diff >= 0 else "⬇️"
                            st.markdown(f"**预测每只利润**{incomplete_suffix}: {pred_ppb:.2f} 元 {arrow}{abs(ppb_diff):.2f}元")

                        if indicators['profit_margin_change_pct'] is not None:
                            margin_change = indicators['profit_margin_change_pct']
                            arrow = "⬆️" if margin_change >= 0 else "⬇️"
                            margin_color = "#006100" if margin_change >= 0 else "#9C0006"
                            st.markdown(f"**利润率影响**: <span style='color:{margin_color}; font-weight:bold;'>{arrow}{abs(margin_change):.2f}个百分点</span> (当前{indicators['profit_margin_pct']:.1f}% → 预测{indicators['predicted_profit_margin_pct']:.1f}%)", unsafe_allow_html=True)

                        health_score = _calculate_batch_health_score(batch)
                        if health_score:
                            avg_risk = health_score['avg_risk']
                            peak_risk = health_score['peak_risk']
                            warning_icon = " ⚠️" if avg_risk > 0.5 else ""
                            st.markdown(f"**健康评分**: {avg_risk:.2f}/{peak_risk:.2f}{warning_icon}")
                        else:
                            st.markdown("**健康评分**: 暂无数据")

                    if indicators['data_incomplete']:
                        st.warning("⚠️ 饲料数据不完整(覆盖天数不足养殖周期80%)，以上含\"数据不完整,仅供参考\"标注的指标仅供参考")

            st.markdown("---")

    completed_batches = [b for b in batches if b['status'] == '已出栏']
    if len(completed_batches) >= 2:
        st.subheader("📊 批次对比分析")

        batch_ids = [b['batch_id'] for b in completed_batches]
        fcr_list = []
        survival_list = []
        profit_per_bird_list = []
        avg_risk_list = []
        peak_risk_list = []
        details = []

        for b in completed_batches:
            ind = _calculate_economic_indicators(b)
            health = _calculate_batch_health_score(b)
            if ind:
                fcr_list.append(ind['fcr'])
                survival_list.append(ind['survival_rate'])
                profit_per_bird_list.append(ind['profit_per_bird'])

                avg_risk = health['avg_risk'] if health else 0
                peak_risk = health['peak_risk'] if health else 0
                avg_risk_list.append(avg_risk)
                peak_risk_list.append(peak_risk)

                total_cost = ind['total_cost']
                chick_cost = ind['chick_cost']
                feed_cost = ind['feed_cost']
                other_cost = ind['other_cost']
                chick_pct = (chick_cost / total_cost * 100) if total_cost > 0 else 0
                feed_pct = (feed_cost / total_cost * 100) if total_cost > 0 else 0
                other_pct = (other_cost / total_cost * 100) if total_cost > 0 else 0

                detail_row = {
                    '批次编号': b['batch_id'],
                    '栋舍': b['barn_id'],
                    '进苗数量': b['chick_count'],
                    '出栏数量': b['slaughter_count'],
                    '苗鸡成本(元)': f"{chick_cost:.2f} ({chick_pct:.1f}%)",
                    '饲料成本(元)': f"{feed_cost:.2f} ({feed_pct:.1f}%)",
                    '其他成本(元)': f"{other_cost:.2f} ({other_pct:.1f}%)",
                    '总投入(元)': round(ind['total_cost'], 2),
                    '总收入(元)': round(ind['total_revenue'], 2),
                    '利润(元)': round(ind['profit'], 2),
                    '料肉比': round(ind['fcr'], 3) if ind['fcr'] is not None else 'N/A',
                    '成活率(%)': round(ind['survival_rate'], 1),
                    '每只利润(元)': round(ind['profit_per_bird'], 2),
                    '日均风险评分': round(avg_risk, 2),
                    '峰值风险评分': round(peak_risk, 2),
                }
                if ind['predicted_feed_cost'] is not None:
                    detail_row['预测饲料成本(元)'] = round(ind['predicted_feed_cost'], 2)
                    if ind['profit_margin_change_pct'] is not None:
                        margin_change = ind['profit_margin_change_pct']
                        arrow = "↑" if margin_change >= 0 else "↓"
                        detail_row['利润率影响(百分点)'] = f"{arrow}{abs(margin_change):.2f}"
                    else:
                        detail_row['利润率影响(百分点)'] = 'N/A'
                details.append(detail_row)

        if details:
            fig = go.Figure()

            fcr_display = [f'{v:.3f}' if v is not None else 'N/A' for v in fcr_list]
            fcr_plot = [v if v is not None else 0 for v in fcr_list]

            fig.add_trace(go.Bar(
                name='料肉比',
                x=batch_ids,
                y=fcr_plot,
                marker_color='#4472C4',
                text=fcr_display,
                textposition='auto',
            ))

            fig.add_trace(go.Bar(
                name='成活率(%)',
                x=batch_ids,
                y=survival_list,
                marker_color='#70AD47',
                text=[f'{v:.1f}%' for v in survival_list],
                textposition='auto',
            ))

            fig.add_trace(go.Bar(
                name='每只利润(元)',
                x=batch_ids,
                y=profit_per_bird_list,
                marker_color='#ED7D31',
                text=[f'{v:.2f}' for v in profit_per_bird_list],
                textposition='auto',
            ))

            fig.add_trace(go.Bar(
                name='日均风险评分',
                x=batch_ids,
                y=avg_risk_list,
                marker_color='#808080',
                text=[f'{v:.2f}' for v in avg_risk_list],
                textposition='auto',
            ))

            fig.update_layout(
                title='各批次核心指标对比',
                xaxis_title='批次编号',
                yaxis_title='数值',
                barmode='group',
                height=450,
            )

            st.plotly_chart(fig, use_container_width=True)

            detail_df = pd.DataFrame(details)

            worst_fcr_idx = None
            worst_survival_idx = None
            worst_profit_idx = None
            worst_risk_idx = None

            valid_fcr = [(i, v) for i, v in enumerate(fcr_list) if v is not None and v > 0]
            if valid_fcr:
                worst_fcr_idx = max(valid_fcr, key=lambda x: x[1])[0]

            if survival_list:
                worst_survival_idx = survival_list.index(min(survival_list))

            if profit_per_bird_list:
                worst_profit_idx = profit_per_bird_list.index(min(profit_per_bird_list))

            if avg_risk_list:
                worst_risk_idx = avg_risk_list.index(max(avg_risk_list))

            worst_cells = []
            if worst_fcr_idx is not None:
                worst_cells.append((worst_fcr_idx, '料肉比'))
            if worst_survival_idx is not None:
                worst_cells.append((worst_survival_idx, '成活率(%)'))
            if worst_profit_idx is not None:
                worst_cells.append((worst_profit_idx, '每只利润(元)'))
            if worst_risk_idx is not None:
                worst_cells.append((worst_risk_idx, '日均风险评分'))

            def apply_cell_highlight(df):
                result = pd.DataFrame('', index=df.index, columns=df.columns)
                for row_idx, col_name in worst_cells:
                    if col_name in df.columns and row_idx < len(df):
                        result.loc[row_idx, col_name] = 'background-color: #FFC7CE; color: #9C0006'
                return result

            styled_df = detail_df.style.apply(apply_cell_highlight, axis=None)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        col_export1, col_export2 = st.columns([3, 1])
        with col_export2:
            csv_data = _export_batches_csv()
            if csv_data:
                filename = f"批次报表_{datetime.now().strftime('%Y%m%d')}.csv"
                st.download_button(
                    label="📥 导出批次报表",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    type="primary",
                    **_st_btn_kwargs()
                )
            else:
                st.button("📥 导出批次报表", disabled=True, **_st_btn_kwargs())


DEFAULT_INGREDIENTS = [
    {'原料名称': '玉米', '单价(元/kg)': 2.8, '粗蛋白(%)': 8.5, '粗脂肪(%)': 3.5, '粗纤维(%)': 2.0, '钙(%)': 0.02, '磷(%)': 0.27, '代谢能(kcal/kg)': 3200},
    {'原料名称': '豆粕', '单价(元/kg)': 4.2, '粗蛋白(%)': 43.0, '粗脂肪(%)': 1.0, '粗纤维(%)': 5.0, '钙(%)': 0.32, '磷(%)': 0.62, '代谢能(kcal/kg)': 2450},
    {'原料名称': '鱼粉', '单价(元/kg)': 12.0, '粗蛋白(%)': 62.0, '粗脂肪(%)': 10.0, '粗纤维(%)': 1.0, '钙(%)': 3.96, '磷(%)': 2.85, '代谢能(kcal/kg)': 2900},
    {'原料名称': '麦麸', '单价(元/kg)': 1.6, '粗蛋白(%)': 15.0, '粗脂肪(%)': 4.0, '粗纤维(%)': 9.0, '钙(%)': 0.18, '磷(%)': 0.78, '代谢能(kcal/kg)': 1650},
    {'原料名称': '石粉', '单价(元/kg)': 0.5, '粗蛋白(%)': 0.0, '粗脂肪(%)': 0.0, '粗纤维(%)': 0.0, '钙(%)': 38.0, '磷(%)': 0.0, '代谢能(kcal/kg)': 0},
]

DEFAULT_CONSTRAINTS = {
    '粗蛋白(%)': {'min': 18.0, 'max': 23.0},
    '粗脂肪(%)': {'min': 3.0, 'max': 8.0},
    '粗纤维(%)': {'min': 2.0, 'max': 5.0},
    '钙(%)': {'min': 0.8, 'max': 1.2},
    '磷(%)': {'min': 0.4, 'max': 0.7},
    '代谢能(kcal/kg)': {'min': 2800.0, 'max': 3200.0},
}

NUTRIENT_COLS = ['粗蛋白(%)', '粗脂肪(%)', '粗纤维(%)', '钙(%)', '磷(%)', '代谢能(kcal/kg)']


def _get_default_ingredients():
    if st.session_state.feed_ingredients is None:
        st.session_state.feed_ingredients = pd.DataFrame(DEFAULT_INGREDIENTS)
    return st.session_state.feed_ingredients


def _get_default_constraints():
    if st.session_state.feed_constraints is None:
        st.session_state.feed_constraints = DEFAULT_CONSTRAINTS.copy()
    return st.session_state.feed_constraints


def _solve_feed_formula(ingredients_df, constraints):
    n = len(ingredients_df)
    if n == 0:
        return None, "请先添加至少一种原料"

    prices = ingredients_df['单价(元/kg)'].values
    nutrient_matrix = ingredients_df[NUTRIENT_COLS].values

    c = prices

    A_ub = []
    b_ub = []

    for i, col in enumerate(NUTRIENT_COLS):
        min_val = constraints[col]['min']
        max_val = constraints[col]['max']

        if max_val is not None:
            A_ub.append(nutrient_matrix[:, i])
            b_ub.append(max_val * 100.0)

        if min_val is not None:
            A_ub.append(-nutrient_matrix[:, i])
            b_ub.append(-min_val * 100.0)

    A_eq = [np.ones(n)]
    b_eq = [100.0]

    bounds = [(0.0, 80.0) for _ in range(n)]

    try:
        result = linprog(
            c,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            A_eq=np.array(A_eq),
            b_eq=np.array(b_eq),
            bounds=bounds,
            method='highs'
        )

        if result.success:
            return result.x, None
        else:
            conflict_info = _analyze_constraint_conflict(ingredients_df, constraints)
            return None, f"求解失败: {result.message}\n{conflict_info}"
    except Exception as e:
        return None, f"求解出错: {str(e)}"


def _analyze_constraint_conflict(ingredients_df, constraints):
    conflict_info = []
    for col in NUTRIENT_COLS:
        min_val = constraints[col]['min']
        max_val = constraints[col]['max']
        if min_val > max_val:
            conflict_info.append(f"- {col}: 最小值({min_val}) > 最大值({max_val})")
    
    max_nutrients = ingredients_df[NUTRIENT_COLS].max()
    min_nutrients = ingredients_df[NUTRIENT_COLS].min()
    
    for col in NUTRIENT_COLS:
        min_val = constraints[col]['min']
        max_val = constraints[col]['max']
        max_possible = max_nutrients[col]
        min_possible = min_nutrients[col]
        if min_val > max_possible:
            conflict_info.append(f"- {col}: 约束最小值({min_val}) > 所有原料最高含量({max_possible})")
        if max_val < min_possible:
            conflict_info.append(f"- {col}: 约束最大值({max_val}) < 所有原料最低含量({min_possible})")
    
    if conflict_info:
        return "可能的约束冲突:\n" + "\n".join(conflict_info)
    return "请尝试调整约束范围或原料配方"


def _calculate_nutrient_values(ingredients_df, amounts):
    result = {}
    total = sum(amounts)
    if total <= 0:
        return result

    for col in NUTRIENT_COLS:
        values = ingredients_df[col].values
        weighted = np.sum(values * amounts) / total
        result[col] = round(weighted, 4)
    return result


def _check_constraints(nutrient_values, constraints):
    results = {}
    for col in NUTRIENT_COLS:
        actual = nutrient_values.get(col, 0)
        min_val = constraints[col]['min']
        max_val = constraints[col]['max']
        is_met = (actual >= min_val - 1e-6) and (actual <= max_val + 1e-6)
        deviation = 0
        if actual < min_val:
            deviation = actual - min_val
        elif actual > max_val:
            deviation = actual - max_val
        results[col] = {
            'met': is_met,
            'actual': actual,
            'min': min_val,
            'max': max_val,
            'deviation': deviation
        }
    return results


def _parse_price_history_csv(file_content):
    """解析价格历史CSV文件"""
    try:
        df = pd.read_csv(file_content)
        if df.shape[1] < 2:
            return None, "CSV文件格式错误，需要至少两列：月份,单价"
        df.columns = [str(c).strip() for c in df.columns]
        month_col = None
        price_col = None
        for col in df.columns:
            if '月' in col or '时间' in col or 'date' in col.lower():
                month_col = col
            elif '价' in col or 'price' in col.lower():
                price_col = col
        if month_col is None or price_col is None:
            month_col = df.columns[0]
            price_col = df.columns[1]
        result_df = pd.DataFrame({
            '月份': df[month_col].astype(str).str.strip(),
            '单价': pd.to_numeric(df[price_col], errors='coerce')
        })
        result_df = result_df.dropna(subset=['单价'])
        if len(result_df) == 0:
            return None, "未解析到有效的价格数据"
        return result_df, None
    except Exception as e:
        return None, f"CSV解析失败: {str(e)}"


def _predict_next_month_price(history_df):
    """使用线性回归(numpy.polyfit一次拟合)预测下一个月价格"""
    if history_df is None or len(history_df) < 3:
        return None, False
    try:
        months = list(range(len(history_df)))
        prices = history_df['单价'].values.astype(float)
        coeffs = np.polyfit(months, prices, 1)
        next_month_idx = len(history_df)
        predicted = np.polyval(coeffs, next_month_idx)
        return float(predicted), True
    except Exception:
        return None, False


def _get_price_warnings(ingredients_df, predicted_prices):
    """检测价格涨幅超过15%的原料"""
    warnings = []
    if ingredients_df is None or len(ingredients_df) == 0:
        return warnings
    for _, row in ingredients_df.iterrows():
        name = row['原料名称']
        current_price = row['单价(元/kg)']
        if name in predicted_prices and predicted_prices[name] is not None and current_price > 0:
            predicted_price = predicted_prices[name]
            increase_pct = ((predicted_price - current_price) / current_price) * 100
            if increase_pct > 15:
                warnings.append({
                    'name': name,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'increase_pct': increase_pct
                })
    return warnings


def _analyze_prediction_conflict(ingredients_df, current_prices, predicted_prices, constraints):
    """分析预测价格导致约束冲突的原因"""
    info_parts = []
    price_increases = []
    for i, (_, row) in enumerate(ingredients_df.iterrows()):
        name = row['原料名称']
        current = current_prices[i]
        if name in predicted_prices and predicted_prices[name] is not None:
            predicted = predicted_prices[name]
            increase_pct = ((predicted - current) / current * 100) if current > 0 else 0
            if increase_pct > 10:
                price_increases.append((name, increase_pct, current, predicted))
    if price_increases:
        price_increases.sort(key=lambda x: x[1], reverse=True)
        info_parts.append("涨价幅度较大的原料:")
        for name, pct, cur, pred in price_increases:
            info_parts.append(f"  - {name}: +{pct:.1f}% (当前 {cur:.2f} → 预测 {pred:.2f})")
    for col in NUTRIENT_COLS:
        min_val = constraints[col]['min']
        max_val = constraints[col]['max']
        if min_val > max_val:
            info_parts.append(f"约束冲突: {col} 最小值({min_val}) > 最大值({max_val})")
    if info_parts:
        info_parts.append("\n建议: 尝试放宽以下约束之一")
        for col in NUTRIENT_COLS:
            min_val = constraints[col]['min']
            max_val = constraints[col]['max']
            info_parts.append(f"  - {col}: 当前范围 [{min_val}, {max_val}]")
        return "\n".join(info_parts)
    return "建议尝试放宽营养约束范围，或调整原料库"


def feed_formula_page():
    """饲料配方模拟与成本优化页面"""
    st.header("🥗 饲料配方模拟与成本优化")
    st.markdown("---")

    ingredients_df = _get_default_ingredients()
    constraints = _get_default_constraints()

    predicted_prices = st.session_state.predicted_prices
    price_warnings = _get_price_warnings(ingredients_df, predicted_prices)
    if price_warnings:
        warning_html_parts = [
            '<div style="background-color: #FFA500; padding: 12px; border-radius: 6px; margin-bottom: 16px;">',
            '<div style="font-weight: bold; font-size: 16px; color: #000; margin-bottom: 8px;">⚠️ 价格波动预警：以下原料预测涨幅超过15%</div>',
            '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
        ]
        for w in price_warnings:
            warning_html_parts.append(
                f'<button onclick="document.querySelector(\'section[data-testid=stSidebar] ~ div [data-testid=stVerticalBlock] [data-testid=stExpanderDetails] [data-testid=stSelectbox] select option:contains({w["name"]})\').selected = true" '
                f'style="background-color: #FFF; border: 1px solid #CC8400; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; color: #CC0000;">'
                f'{w["name"]} +{w["increase_pct"]:.1f}%</button>'
            )
        warning_html_parts.append('</div></div>')
        st.markdown(''.join(warning_html_parts), unsafe_allow_html=True)
        for w in price_warnings:
            if st.button(f"🔍 查看{w['name']}敏感性分析", key=f"warn_jump_{w['name']}"):
                st.session_state['sen_ingredient_default'] = w['name']
                st.session_state['sen_range_default'] = (-max(30, int(w['increase_pct'])), max(30, int(w['increase_pct'])))
                st.rerun()

    st.subheader("📦 原料库管理")

    edited_df = st.data_editor(
        ingredients_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "原料名称": st.column_config.TextColumn("原料名称", required=True),
            "单价(元/kg)": st.column_config.NumberColumn("单价(元/kg)", min_value=0.0, step=0.1, format="%.2f"),
            "粗蛋白(%)": st.column_config.NumberColumn("粗蛋白(%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
            "粗脂肪(%)": st.column_config.NumberColumn("粗脂肪(%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
            "粗纤维(%)": st.column_config.NumberColumn("粗纤维(%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
            "钙(%)": st.column_config.NumberColumn("钙(%)", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "磷(%)": st.column_config.NumberColumn("磷(%)", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "代谢能(kcal/kg)": st.column_config.NumberColumn("代谢能(kcal/kg)", min_value=0, step=10, format="%.0f"),
        },
        key="ingredients_editor"
    )

    st.session_state.feed_ingredients = edited_df
    st.caption(f"当前原料总数: {len(edited_df)} 种")

    with st.expander("📈 价格历史", expanded=False):
        st.markdown("上传每种原料的近12个月历史价格(CSV格式: 月份,单价)，用于价格趋势预测。")
        ingredient_names = edited_df['原料名称'].dropna().astype(str).str.strip().tolist()
        ingredient_names = [n for n in ingredient_names if n]

        price_history = st.session_state.ingredient_price_history
        predicted_prices = {}

        if ingredient_names:
            for ing_name in ingredient_names:
                col_up1, col_up2, col_up3 = st.columns([2, 1, 1])
                with col_up1:
                    st.markdown(f"**{ing_name}**")
                with col_up2:
                    history = price_history.get(ing_name, None)
                    if history is not None:
                        st.info(f"已导入 {len(history)} 条数据")
                    else:
                        st.info("暂无历史数据")
                with col_up3:
                    if history is not None:
                        if st.button(f"清除", key=f"clear_hist_{ing_name}"):
                            if ing_name in price_history:
                                del price_history[ing_name]
                                st.session_state.ingredient_price_history = price_history.copy()
                                st.rerun()

                col_f1, col_f2 = st.columns([2, 1])
                with col_f1:
                    uploaded = st.file_uploader(
                        f"上传 {ing_name} 价格历史CSV",
                        type=["csv"],
                        key=f"price_upload_{ing_name}",
                        label_visibility="collapsed"
                    )
                    if uploaded is not None:
                        parsed_df, parse_error = _parse_price_history_csv(uploaded)
                        if parse_error:
                            st.error(parse_error)
                        else:
                            price_history[ing_name] = parsed_df
                            st.session_state.ingredient_price_history = price_history.copy()
                            st.success(f"✅ {ing_name} 价格历史导入成功，共 {len(parsed_df)} 条记录")
                with col_f2:
                    history = price_history.get(ing_name, None)
                    if history is not None:
                        pred_price, pred_ok = _predict_next_month_price(history)
                        predicted_prices[ing_name] = pred_price if pred_ok else None
                        if pred_ok:
                            st.success(f"预测下月价格: {pred_price:.2f} 元/kg")
                        else:
                            st.warning("数据不足(需≥3个月)，无法预测")
                    else:
                        st.warning("未上传，跳过预测")

            st.session_state.predicted_prices = predicted_prices

            valid_histories = {name: hist for name, hist in price_history.items() if hist is not None and len(hist) > 0}
            if valid_histories:
                st.markdown("---")
                st.markdown("**📊 价格趋势与预测**")
                fig_prices = go.Figure()
                colors = px.colors.qualitative.Plotly
                for idx, (name, hist) in enumerate(valid_histories.items()):
                    color = colors[idx % len(colors)]
                    months_list = hist['月份'].tolist()
                    prices_list = hist['单价'].tolist()
                    has_prediction = name in predicted_prices and predicted_prices[name] is not None
                    x_vals = list(range(len(months_list)))
                    y_vals = prices_list

                    fig_prices.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers',
                        name=name,
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                        hovertemplate=f'{name}<br>%{{text}}: %{{y:.2f}}元/kg<extra></extra>',
                        text=months_list
                    ))

                    if has_prediction:
                        pred_x = [len(months_list) - 1, len(months_list)]
                        pred_y = [prices_list[-1], predicted_prices[name]]
                        fig_prices.add_trace(go.Scatter(
                            x=pred_x,
                            y=pred_y,
                            mode='lines+markers',
                            name=f'{name}(预测)',
                            line=dict(color=color, width=2, dash='dash'),
                            marker=dict(size=10, symbol='diamond'),
                            hovertemplate=f'{name}(预测)<br>%{{y:.2f}}元/kg<extra></extra>',
                            showlegend=False
                        ))
                        fig_prices.add_annotation(
                            x=len(months_list),
                            y=predicted_prices[name],
                            text=f"{predicted_prices[name]:.2f}",
                            showarrow=True,
                            arrowhead=2,
                            ax=40,
                            ay=0,
                            bgcolor=color,
                            font=dict(color='white', size=10)
                        )
                    else:
                        if len(hist) < 3:
                            fig_prices.add_annotation(
                                x=len(months_list) - 1,
                                y=prices_list[-1],
                                text="数据不足",
                                showarrow=False,
                                bgcolor='#808080',
                                font=dict(color='white', size=10),
                                xshift=40
                            )

                fig_prices.update_layout(
                    title='各原料价格走势与预测',
                    xaxis_title='月份',
                    yaxis_title='单价(元/kg)',
                    height=450,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                if months_list:
                    tick_positions = list(range(len(months_list)))
                    fig_prices.update_xaxes(
                        tickmode='array',
                        tickvals=tick_positions,
                        ticktext=months_list + ['预测下月'] if has_prediction else months_list
                    )
                st.plotly_chart(fig_prices, use_container_width=True)
            else:
                st.info("暂无已导入的价格历史数据")
        else:
            st.info("请先在原料库中添加原料")

    st.markdown("---")
    st.subheader("⚙️ 营养约束设置")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        col_name = '粗蛋白(%)'
        cp_min = st.number_input(f"{col_name} 最小值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['min']), step=0.1, key=f"{col_name}_min")
        cp_max = st.number_input(f"{col_name} 最大值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['max']), step=0.1, key=f"{col_name}_max")
        constraints[col_name] = {'min': cp_min, 'max': cp_max}

        col_name = '粗脂肪(%)'
        ee_min = st.number_input(f"{col_name} 最小值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['min']), step=0.1, key=f"{col_name}_min")
        ee_max = st.number_input(f"{col_name} 最大值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['max']), step=0.1, key=f"{col_name}_max")
        constraints[col_name] = {'min': ee_min, 'max': ee_max}

        col_name = '粗纤维(%)'
        cf_min = st.number_input(f"{col_name} 最小值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['min']), step=0.1, key=f"{col_name}_min")
        cf_max = st.number_input(f"{col_name} 最大值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['max']), step=0.1, key=f"{col_name}_max")
        constraints[col_name] = {'min': cf_min, 'max': cf_max}

    with col_c2:
        col_name = '钙(%)'
        ca_min = st.number_input(f"{col_name} 最小值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['min']), step=0.01, key=f"{col_name}_min")
        ca_max = st.number_input(f"{col_name} 最大值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['max']), step=0.01, key=f"{col_name}_max")
        constraints[col_name] = {'min': ca_min, 'max': ca_max}

        col_name = '磷(%)'
        p_min = st.number_input(f"{col_name} 最小值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['min']), step=0.01, key=f"{col_name}_min")
        p_max = st.number_input(f"{col_name} 最大值(%)", min_value=0.0, max_value=100.0, value=float(constraints[col_name]['max']), step=0.01, key=f"{col_name}_max")
        constraints[col_name] = {'min': p_min, 'max': p_max}

        col_name = '代谢能(kcal/kg)'
        me_min = st.number_input(f"{col_name} 最小值", min_value=0, max_value=5000, value=int(constraints[col_name]['min']), step=50, key=f"{col_name}_min")
        me_max = st.number_input(f"{col_name} 最大值", min_value=0, max_value=5000, value=int(constraints[col_name]['max']), step=50, key=f"{col_name}_max")
        constraints[col_name] = {'min': float(me_min), 'max': float(me_max)}

    st.session_state.feed_constraints = constraints

    st.info("💡 配方总量固定为 100kg，每种原料用量范围 0-80kg")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        if st.button("🧮 计算最优配方", type="primary", **_st_btn_kwargs()):
            with st.spinner("正在求解最优配方..."):
                valid_ings = edited_df.dropna(subset=['原料名称', '单价(元/kg)'])
                valid_ings = valid_ings[valid_ings['原料名称'].astype(str).str.strip() != '']
                if len(valid_ings) == 0:
                    st.error("❌ 请至少输入一种有效原料")
                else:
                    amounts, error = _solve_feed_formula(valid_ings, constraints)
                    if amounts is not None:
                        nutrient_values = _calculate_nutrient_values(valid_ings, amounts)
                        constraint_checks = _check_constraints(nutrient_values, constraints)
                        total_cost = np.sum(valid_ings['单价(元/kg)'].values * amounts)
                        unit_price = total_cost / 100.0

                        result_data = {
                            'ingredients': valid_ings.to_dict('records'),
                            'amounts': amounts.tolist(),
                            'nutrient_values': nutrient_values,
                            'constraint_checks': constraint_checks,
                            'total_cost': total_cost,
                            'unit_price': unit_price,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        st.session_state.feed_formula_result = result_data
                        st.success("✅ 最优配方计算成功!")
                    else:
                        st.error(f"❌ {error}")
                        st.info("💡 提示：请检查约束条件是否互相冲突，例如最小值大于最大值，或约束范围过窄导致无可行解")

    with col_btn2:
        if st.button("🔮 基于预测价格计算", **_st_btn_kwargs()):
            with st.spinner("正在基于预测价格求解最优配方..."):
                valid_ings = edited_df.dropna(subset=['原料名称', '单价(元/kg)'])
                valid_ings = valid_ings[valid_ings['原料名称'].astype(str).str.strip() != '']
                if len(valid_ings) == 0:
                    st.error("❌ 请至少输入一种有效原料")
                else:
                    predicted_prices = st.session_state.predicted_prices
                    predicted_ings = valid_ings.copy()
                    current_prices = valid_ings['单价(元/kg)'].values.tolist()
                    for i, (_, row) in enumerate(predicted_ings.iterrows()):
                        name = row['原料名称']
                        if name in predicted_prices and predicted_prices[name] is not None:
                            predicted_ings.iloc[i, predicted_ings.columns.get_loc('单价(元/kg)')] = predicted_prices[name]

                    amounts_pred, error_pred = _solve_feed_formula(predicted_ings.reset_index(drop=True), constraints)
                    if amounts_pred is not None:
                        nutrient_values_pred = _calculate_nutrient_values(predicted_ings.reset_index(drop=True), amounts_pred)
                        constraint_checks_pred = _check_constraints(nutrient_values_pred, constraints)
                        total_cost_pred = np.sum(predicted_ings['单价(元/kg)'].values * amounts_pred)
                        unit_price_pred = total_cost_pred / 100.0

                        amounts_curr, error_curr = _solve_feed_formula(valid_ings, constraints)
                        total_cost_current = None
                        if amounts_curr is not None:
                            total_cost_current = np.sum(valid_ings['单价(元/kg)'].values * amounts_curr)

                        result_data_pred = {
                            'ingredients': predicted_ings.to_dict('records'),
                            'original_ingredients': valid_ings.to_dict('records'),
                            'amounts': amounts_pred.tolist(),
                            'nutrient_values': nutrient_values_pred,
                            'constraint_checks': constraint_checks_pred,
                            'total_cost': total_cost_pred,
                            'unit_price': unit_price_pred,
                            'total_cost_current': total_cost_current,
                            'used_predicted_prices': True,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        st.session_state.predicted_feed_formula_result = result_data_pred
                        st.session_state.feed_formula_result = result_data_pred
                        st.success("✅ 基于预测价格的最优配方计算成功!")
                    else:
                        conflict_analysis = _analyze_prediction_conflict(
                            valid_ings, current_prices, predicted_prices, constraints
                        )
                        st.error(f"❌ {error_pred}")
                        st.warning(f"🔍 冲突分析:\n{conflict_analysis}")

    st.markdown("---")

    if st.session_state.feed_formula_result is not None:
        result = st.session_state.feed_formula_result
        valid_ings = pd.DataFrame(result['ingredients'])
        amounts = np.array(result['amounts'])
        nutrient_values = result['nutrient_values']
        constraint_checks = result['constraint_checks']
        total_cost = result['total_cost']
        unit_price = result['unit_price']

        st.subheader("📊 配方计算结果")

        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            pie_amounts = amounts[amounts > 0.1]
            pie_names = valid_ings['原料名称'].values[amounts > 0.1].tolist()
            if len(pie_amounts) > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=pie_names,
                    values=pie_amounts,
                    textinfo='label+percent',
                    hovertemplate='%{label}<br>用量: %{value:.2f}kg<br>占比: %{percent}',
                    marker=dict(colors=px.colors.qualitative.Plotly)
                )])
                fig_pie.update_layout(
                    title='原料用量占比',
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_r2:
            result_rows = []
            for i, (_, row) in enumerate(valid_ings.iterrows()):
                amt = amounts[i]
                if amt > 0.01:
                    pct = amt / 100.0 * 100
                    cost = row['单价(元/kg)'] * amt
                    result_rows.append({
                        '原料名称': row['原料名称'],
                        '用量(kg)': round(amt, 2),
                        '占比(%)': round(pct, 2),
                        '单价(元/kg)': row['单价(元/kg)'],
                        '分项成本(元)': round(cost, 2)
                    })

            result_df = pd.DataFrame(result_rows)
            if not result_df.empty:
                result_df = result_df.sort_values('用量(kg)', ascending=False)
                total_row = pd.DataFrame([{
                    '原料名称': '合计',
                    '用量(kg)': round(result_df['用量(kg)'].sum(), 2),
                    '占比(%)': round(result_df['占比(%)'].sum(), 2),
                    '单价(元/kg)': '',
                    '分项成本(元)': round(result_df['分项成本(元)'].sum(), 2)
                }])
                display_df = pd.concat([result_df, total_row], ignore_index=True)

                highlight_total = pd.DataFrame('', index=display_df.index, columns=display_df.columns)
                highlight_total.iloc[-1, :] = 'font-weight: bold; background-color: #E7E7E7'

                st.dataframe(
                    display_df.style.apply(lambda _: highlight_total, axis=None),
                    use_container_width=True,
                    hide_index=True
                )

        st.markdown("---")
        st.subheader("📋 营养成分验证")

        check_rows = []
        all_met = True
        for col in NUTRIENT_COLS:
            check = constraint_checks[col]
            status_icon = "✅" if check['met'] else "❌"
            deviation_text = "" if check['met'] else f" (偏差: {check['deviation']:+.4f})"
            check_rows.append({
                '营养成分': col,
                '实际值': round(check['actual'], 2),
                '约束范围': f"{check['min']} ~ {check['max']}",
                '状态': f"{status_icon}{deviation_text}",
                '是否满足': check['met']
            })
            if not check['met']:
                all_met = False

        check_df = pd.DataFrame(check_rows)

        def highlight_status(s):
            return ['background-color: #C6EFCE; color: #006100' if v else 'background-color: #FFC7CE; color: #9C0006' for v in s]

        st.dataframe(
            check_df.style.apply(highlight_status, subset=['是否满足']),
            use_container_width=True,
            hide_index=True
        )

        if all_met:
            st.success("✅ 所有营养约束均已满足!")
        else:
            st.warning("⚠️ 部分营养约束未满足，请检查约束条件")

        st.markdown("---")

        used_predicted = result.get('used_predicted_prices', False)
        total_cost_current = result.get('total_cost_current', None)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            if used_predicted:
                label_prefix = "预测价格配方成本"
            else:
                label_prefix = "配方总成本"
            st.metric(label_prefix, f"{total_cost:.2f} 元/100kg", delta_color="inverse")
        with col_m2:
            if used_predicted:
                label_prefix = "预测饲料单价"
            else:
                label_prefix = "饲料单价"
            st.metric(label_prefix, f"{unit_price:.2f} 元/kg", delta_color="inverse")

        with col_m3:
            apply_to_batch = st.toggle(
                "应用到批次计算",
                value=(st.session_state.global_feed_unit_price == unit_price) or
                      (used_predicted and st.session_state.global_feed_unit_price_predicted == unit_price),
            )
            if apply_to_batch:
                if used_predicted:
                    st.session_state.global_feed_unit_price = unit_price
                    st.session_state.global_feed_unit_price_predicted = unit_price
                    st.success("✅ 当前单价和预测单价均已应用到批次计算")
                else:
                    st.session_state.global_feed_unit_price = unit_price
                    st.success("✅ 已应用到批次计算")
            else:
                if st.session_state.global_feed_unit_price != DEFAULT_FEED_UNIT_PRICE:
                    st.session_state.global_feed_unit_price = DEFAULT_FEED_UNIT_PRICE
                if st.session_state.global_feed_unit_price_predicted is not None:
                    st.session_state.global_feed_unit_price_predicted = None
        with col_m4:
            info_text = f"当前批次模块单价: {st.session_state.global_feed_unit_price:.2f} 元/kg"
            if st.session_state.global_feed_unit_price_predicted is not None:
                info_text += f"\n预测批次单价: {st.session_state.global_feed_unit_price_predicted:.2f} 元/kg"
            st.info(info_text)

        if used_predicted and total_cost_current is not None:
            diff_pct = ((total_cost - total_cost_current) / total_cost_current) * 100 if total_cost_current > 0 else 0
            diff_color = "inverse" if diff_pct > 0 else "normal"
            delta_text = f"{diff_pct:+.2f}%"
            st.markdown("---")
            col_cmp1, col_cmp2, col_cmp3 = st.columns(3)
            with col_cmp1:
                st.metric("当前价格配方成本", f"{total_cost_current:.2f} 元/100kg", delta_color="inverse")
            with col_cmp2:
                st.metric("预测价格配方成本", f"{total_cost:.2f} 元/100kg", delta=delta_text, delta_color=diff_color)
            with col_cmp3:
                direction = "上涨" if diff_pct > 0 else "下降"
                arrow = "⬆️" if diff_pct > 0 else "⬇️"
                st.metric("成本变化", f"{arrow} {abs(diff_pct):.2f}%", delta_color=diff_color)
                st.caption(f"预测下月配方成本较当前{direction} {abs(diff_pct):.2f}%")

        st.markdown("---")
        st.subheader("📈 敏感性分析")

        if len(valid_ings) > 0:
            sen_col1, sen_col2, sen_col3 = st.columns([1, 1, 1])

            default_sen_idx = 0
            if 'sen_ingredient_default' in st.session_state:
                default_name = st.session_state.pop('sen_ingredient_default')
                if default_name in valid_ings['原料名称'].tolist():
                    default_sen_idx = valid_ings['原料名称'].tolist().index(default_name)

            default_sen_range = (-30, 50)
            if 'sen_range_default' in st.session_state:
                default_sen_range = st.session_state.pop('sen_range_default')

            with sen_col1:
                sen_ingredient = st.selectbox(
                    "选择原料",
                    valid_ings['原料名称'].tolist(),
                    index=default_sen_idx,
                    key="sen_ingredient")
            with sen_col2:
                price_range_min = st.slider(
                    "价格波动范围(%)",
                    min_value=-50,
                    max_value=100,
                    value=default_sen_range,
                    step=5,
                    key="sen_range")
            with sen_col3:
                price_step = st.selectbox(
                    "步长(%)",
                    [5, 10, 15],
                    index=0,
                    key="sen_step")

            if st.button("🔍 运行分析", **_st_btn_kwargs()):
                with st.spinner("正在进行敏感性分析..."):
                    match_mask = valid_ings['原料名称'].values == sen_ingredient
                    match_positions = np.where(match_mask)[0]
                    if len(match_positions) == 0:
                        st.error("❌ 未找到所选原料")
                    else:
                        ing_pos = match_positions[0]
                        base_price = valid_ings.iloc[ing_pos]['单价(元/kg)']
                        current_constraints = st.session_state.feed_constraints

                        price_changes = list(range(price_range_min[0], price_range_min[1] + 1, price_step))
                        sen_results = []

                        for change_pct in price_changes:
                            temp_ings = valid_ings.copy()
                            temp_ings.iloc[ing_pos, temp_ings.columns.get_loc('单价(元/kg)')] = base_price * (1 + change_pct / 100.0)
                            amounts_sen, error_sen = _solve_feed_formula(temp_ings.reset_index(drop=True), current_constraints)
                            if amounts_sen is not None:
                                cost_sen = np.sum(temp_ings['单价(元/kg)'].values * amounts_sen)
                                sen_results.append({
                                    'change_pct': change_pct,
                                    'price': base_price * (1 + change_pct / 100.0),
                                    'total_cost': cost_sen,
                                    'feasible': True
                                })
                            else:
                                sen_results.append({
                                    'change_pct': change_pct,
                                    'price': base_price * (1 + change_pct / 100.0),
                                    'total_cost': None,
                                    'feasible': False
                                })

                        st.session_state.sensitivity_result = {
                            'ingredient': sen_ingredient,
                            'base_price': base_price,
                            'results': sen_results
                        }

        if st.session_state.sensitivity_result is not None:
            sen_result = st.session_state.sensitivity_result
            sen_results = sen_result['results']
            base_price = sen_result['base_price']

            fig_sen = go.Figure()

            feasible_x = [r['change_pct'] for r in sen_results if r['feasible']]
            feasible_y = [r['total_cost'] for r in sen_results if r['feasible']]
            infeasible_x = [r['change_pct'] for r in sen_results if not r['feasible']]
            infeasible_y = [0 for _ in sen_results if not r['feasible']]

            fig_sen.add_trace(go.Scatter(
                x=feasible_x,
                y=feasible_y,
                mode='lines+markers',
                name='可行解',
                line=dict(color='#4472C4', width=2),
                marker=dict(size=8)
            ))

            if infeasible_x:
                fig_sen.add_trace(go.Scatter(
                    x=infeasible_x,
                    y=infeasible_y,
                    mode='markers',
                    name='无可行解',
                    marker=dict(color='red', size=10, symbol='x')
                ))

            fig_sen.add_vline(
                x=0,
                line_dash="dash",
                line_color="green",
                annotation_text="当前价格",
                annotation_position="top right"
            )

            fig_sen.update_layout(
                title=f"{sen_result['ingredient']} 价格敏感性分析",
                xaxis_title="价格变化率(%)",
                yaxis_title="配方总成本(元/100kg)",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig_sen, use_container_width=True)

            sen_df = pd.DataFrame([{
                '价格变化(%)': r['change_pct'],
                '原料价格(元/kg)': round(r['price'], 2),
                '配方成本(元/100kg)': round(r['total_cost'], 2) if r['feasible'] else '无可行解',
                '可行性': '✅ 可行' if r['feasible'] else '❌ 不可行'
            } for r in sen_results])

            st.dataframe(sen_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    save_col1, save_col2 = st.columns([3, 1])
    with save_col2:
        formula_name = st.text_input("方案名称", value=f"方案{len(st.session_state.saved_formulas) + 1}", key="save_formula_name")
    with save_col1:
        if st.button("💾 保存当前配方", **_st_btn_kwargs()):
            if st.session_state.feed_formula_result is None:
                st.warning("⚠️ 请先计算配方后再保存")
            else:
                saved = {
                    'name': formula_name.strip() if formula_name.strip() else f"方案{len(st.session_state.saved_formulas) + 1}",
                    'data': st.session_state.feed_formula_result.copy(),
                    'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.session_state.saved_formulas.append(saved)
                st.success(f"✅ 配方方案「{saved['name']}」保存成功!")
                st.rerun()

    if len(st.session_state.saved_formulas) > 0:
        st.markdown("---")
        st.subheader("💾 已保存配方方案")

        saved_names = [f"{s['name']} ({s['saved_at']})" for s in st.session_state.saved_formulas]
        to_delete = st.multiselect("选择要删除的方案", saved_names, key="delete_formulas")
        if st.button("🗑️ 删除选中方案", **_st_btn_kwargs()):
            if to_delete:
                delete_indices = [saved_names.index(n) for n in to_delete]
                st.session_state.saved_formulas = [s for i, s in enumerate(st.session_state.saved_formulas) if i not in delete_indices]
                st.success(f"✅ 已删除 {len(to_delete)} 个方案")
                st.rerun()

        if len(st.session_state.saved_formulas) >= 2:
            st.markdown("---")
            st.subheader("📊 配方方案对比")

            compare_fig = go.Figure()

            all_names = [s['name'] for s in st.session_state.saved_formulas]
            all_costs = [s['data']['total_cost'] for s in st.session_state.saved_formulas]
            max_cost_idx = all_costs.index(max(all_costs))

            compare_fig.add_trace(go.Bar(
                name='总成本(元/100kg)',
                x=all_names,
                y=all_costs,
                marker_color='#ED7D31',
                text=[f'{c:.2f}' for c in all_costs],
                textposition='auto',
            ))

            for col in NUTRIENT_COLS:
                nutrient_vals = []
                for s in st.session_state.saved_formulas:
                    nv = s['data']['nutrient_values'][col]
                    nutrient_vals.append(nv)
                normalize_factor = 1.0
                if '代谢能' in col:
                    normalize_factor = 100.0
                compare_fig.add_trace(go.Bar(
                        name=col,
                        x=all_names,
                        y=[v / normalize_factor for v in nutrient_vals],
                        text=[f'{v:.2f}' for v in nutrient_vals],
                        textposition='auto',
                    ))

            compare_fig.update_layout(
                title='各方案总成本与营养成分对比',
                xaxis_title='方案名称',
                yaxis_title='数值',
                barmode='group',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(compare_fig, use_container_width=True)

            st.markdown("**原料用量对比**")

            all_ingredients = set()
            for s in st.session_state.saved_formulas:
                for ing in s['data']['ingredients']:
                    all_ingredients.add(ing['原料名称'])

            compare_rows = []
            for ing_name in sorted(all_ingredients):
                row = {'原料名称': ing_name}
                for s in st.session_state.saved_formulas:
                    ings_df = pd.DataFrame(s['data']['ingredients'])
                    amounts = s['data']['amounts']
                    ing_idx = ings_df[ings_df['原料名称'] == ing_name].index
                    if len(ing_idx) > 0:
                        amt = amounts[ing_idx[0]]
                        row[s['name']] = f"{amt:.2f} kg"
                    else:
                        row[s['name']] = "0.00 kg"
                compare_rows.append(row)

            compare_df = pd.DataFrame(compare_rows)

            def highlight_max_cost(df):
                result = pd.DataFrame('', index=df.index, columns=df.columns)
                cost_col_name = all_names[max_cost_idx]
                if cost_col_name in df.columns:
                    result[cost_col_name] = 'background-color: #FFC7CE; color: #9C0006'
                return result

            st.dataframe(
                compare_df.style.apply(highlight_max_cost, axis=None),
                use_container_width=True,
                hide_index=True
            )

            st.caption("注: 红色标注为成本最高的方案")


def report_page(livestock_type, total_livestock):
    """报告导出页面"""
    st.header("📄 报告导出")
    st.markdown("---")
    
    if st.session_state.env_data is None or st.session_state.prod_data is None:
        st.warning("请先导入环境监测数据和生产数据")
        return
    
    env_df = st.session_state.env_data.copy()
    prod_df = st.session_state.prod_data.copy()
    env_df['时间戳'] = pd.to_datetime(env_df['时间戳'])
    prod_df['时间戳'] = pd.to_datetime(prod_df['时间戳'])
    
    status_df = evaluate_barn_status(env_df, livestock_type)
    anomaly_summary = summarize_anomalies(env_df)
    risk_df = get_all_barns_risk(env_df, prod_df, livestock_type, total_livestock)
    
    st.subheader("📋 报告预览")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("监测栋舍数", len(status_df))
    
    with col2:
        high_risk_count = len(risk_df[risk_df['风险等级'] == '高风险']) if not risk_df.empty else 0
        st.metric("高风险栋舍数", high_risk_count, delta_color="inverse")
    
    st.subheader("环境状态摘要")
    st.dataframe(
        style_dataframe(status_df[['栋舍编号', '温度(℃)', '湿度(%)', 'THI指数', 'THI状态', '综合状态']]),
        use_container_width=True,
        hide_index=True
    )
    
    st.subheader("疾病风险摘要")
    display_risk = risk_df[['栋舍编号', '风险评分', '风险等级', '触发信号']].copy()
    display_risk['触发信号'] = display_risk['触发信号'].apply(lambda x: ', '.join(x) if x else '无')
    st.dataframe(
        style_dataframe(display_risk, status_cols=['风险等级']),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("📥 生成并下载报告")
    
    report_date = st.date_input("报告日期", value=datetime.now())
    
    if st.button("📄 生成PDF报告", type="primary"):
        with st.spinner("正在生成PDF报告..."):
            try:
                pdf_buffer = generate_daily_report(
                    env_df, prod_df, status_df, anomaly_summary, risk_df,
                    livestock_type, total_livestock,
                    report_date=str(report_date),
                    barn_ids=status_df['栋舍编号'].unique().tolist()
                )
                
                st.success("✅ 报告生成成功!")
                
                st.download_button(
                    label="📥 下载PDF报告",
                    data=pdf_buffer.getvalue(),
                    file_name=f"养殖环境日报_{report_date}.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"报告生成失败: {str(e)}")
                st.info("提示: 请确保系统中文字体可用(simsun.ttc或simhei.ttf)")


if __name__ == "__main__":
    main()
