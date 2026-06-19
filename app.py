"""规模化养殖场环境监测与动物疾病预警系统"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

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
                use_container_width=True
            )
        else:
            st.button("📥 导出工单CSV", disabled=True, use_container_width=True)
    
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
