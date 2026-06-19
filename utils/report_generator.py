"""PDF报告导出模块"""
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from datetime import datetime
import pandas as pd
import numpy as np


def setup_matplotlib_chinese():
    """设置matplotlib中文字体"""
    try:
        rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False


def register_chinese_font():
    """注册中文字体"""
    setup_matplotlib_chinese()
    try:
        pdfmetrics.registerFont(TTFont('SimSun', 'C:\\Windows\\Fonts\\simsun.ttc'))
        return 'SimSun'
    except:
        try:
            pdfmetrics.registerFont(TTFont('SimHei', 'C:\\Windows\\Fonts\\simhei.ttf'))
            return 'SimHei'
        except:
            return 'Helvetica'


FONT_NAME = 'SimSun'


def create_summary_table(status_df, font_name=FONT_NAME):
    """创建环境参数统计摘要表"""
    if status_df.empty:
        return None
    
    data = [['栋舍编号', '温度(℃)', '湿度(%)', 'THI指数', '氨气(ppm)', 'CO2(ppm)', '综合状态']]
    
    for _, row in status_df.iterrows():
        data.append([
            str(row['栋舍编号']),
            f"{row['温度(℃)']}",
            f"{row['湿度(%)']}",
            f"{row['THI指数']}",
            f"{row['氨气(ppm)']}",
            f"{row['CO2(ppm)']}",
            row['综合状态']
        ])
    
    table = Table(data, colWidths=[2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2.5*cm])
    
    style = TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
    ])
    
    for i in range(1, len(data)):
        status = data[i][-1]
        if '良好' in status:
            bg_color = colors.HexColor('#C6EFCE')
            text_color = colors.HexColor('#006100')
        elif '轻度' in status:
            bg_color = colors.HexColor('#FFEB9C')
            text_color = colors.HexColor('#9C5700')
        elif '中度' in status:
            bg_color = colors.HexColor('#FFC000')
            text_color = colors.HexColor('#806000')
        else:
            bg_color = colors.HexColor('#FFC7CE')
            text_color = colors.HexColor('#9C0006')
        
        style.add('BACKGROUND', (-1, i), (-1, i), bg_color)
        style.add('TEXTCOLOR', (-1, i), (-1, i), text_color)
    
    table.setStyle(style)
    return table


def create_alert_list(anomaly_summary, font_name=FONT_NAME):
    """创建当日告警汇总列表"""
    if anomaly_summary.empty:
        return None
    
    data = [['栋舍编号', '采样点数', '异常点数', '异常占比', '温度', '湿度', '氨气', 'CO2']]
    
    for _, row in anomaly_summary.iterrows():
        data.append([
            str(row['栋舍编号']),
            str(int(row['采样点数'])),
            str(int(row['异常点数'])),
            f"{row['异常占比(%)']}%",
            str(int(row.get('温度', 0))),
            str(int(row.get('湿度', 0))),
            str(int(row.get('氨气浓度(ppm)', 0))),
            str(int(row.get('CO2浓度(ppm)', 0)))
        ])
    
    table = Table(data, colWidths=[1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm])
    
    style = TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ED7D31')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
    ])
    
    table.setStyle(style)
    return table


def create_risk_table(risk_df, font_name=FONT_NAME):
    """创建疾病风险评分表"""
    if risk_df.empty:
        return None
    
    data = [['栋舍编号', '风险评分', '风险等级', '触发信号', '建议']]
    
    for _, row in risk_df.iterrows():
        signals = ', '.join(row['触发信号']) if row['触发信号'] else '无'
        suggestion = generate_risk_suggestion(row['风险等级'], row['触发信号'])
        data.append([
            str(row['栋舍编号']),
            f"{row['风险评分']:.2f}",
            row['风险等级'],
            signals,
            suggestion
        ])
    
    table = Table(data, colWidths=[1.8*cm, 1.8*cm, 1.8*cm, 4*cm, 4*cm])
    
    style = TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#70AD47')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (2, -1), 'CENTER'),
        ('ALIGN', (3, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('LEFTPADDING', (3, 0), (-1, -1), 5),
        ('RIGHTPADDING', (3, 0), (-1, -1), 5),
    ])
    
    for i in range(1, len(data)):
        level = data[i][2]
        if '高' in level:
            bg_color = colors.HexColor('#FFC7CE')
            text_color = colors.HexColor('#9C0006')
        elif '中' in level:
            bg_color = colors.HexColor('#FFEB9C')
            text_color = colors.HexColor('#9C5700')
        elif '低' in level:
            bg_color = colors.HexColor('#FFFF00')
            text_color = colors.HexColor('#806000')
        else:
            bg_color = colors.HexColor('#C6EFCE')
            text_color = colors.HexColor('#006100')
        
        style.add('BACKGROUND', (2, i), (2, i), bg_color)
        style.add('TEXTCOLOR', (2, i), (2, i), text_color)
    
    table.setStyle(style)
    return table


def generate_risk_suggestion(risk_level, signals):
    """根据风险等级和触发信号生成建议"""
    suggestions = []
    
    if '采食量异常' in signals:
        suggestions.append('检查饲料质量和投喂系统')
    if '饮水量异常' in signals:
        suggestions.append('检查饮水系统和水质')
    if '死淘异常' in signals:
        suggestions.append('加强巡视，排查死亡原因')
    if '环境恶化' in signals:
        suggestions.append('改善通风降温条件')
    if '空气质量异常' in signals:
        suggestions.append('加强通风，降低氨气浓度')
    
    if not suggestions:
        if risk_level == '正常':
            suggestions.append('环境良好，继续保持')
        else:
            suggestions.append('密切关注，持续监测')
    
    return '; '.join(suggestions)


def create_thi_chart(env_df, barn_id, figsize=(6, 3), dpi=100):
    """生成THI趋势图"""
    from utils.comfort_eval import calculate_thi, THI_THRESHOLDS
    
    barn_data = env_df[env_df['栋舍编号'] == barn_id].sort_values('时间戳').copy()
    if barn_data.empty:
        return None
    
    barn_data['THI'] = barn_data.apply(
        lambda row: calculate_thi(row['温度'], row['湿度']), axis=1
    )
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(barn_data['时间戳'], barn_data['THI'], label='THI指数', color='#4472C4', linewidth=1.5)
    
    thresholds = THI_THRESHOLDS.get('肉鸡', THI_THRESHOLDS['肉鸡'])
    ax.axhline(y=thresholds['正常'], color='green', linestyle='--', alpha=0.7, label='正常阈值')
    ax.axhline(y=thresholds['轻度热应激'], color='orange', linestyle='--', alpha=0.7, label='轻度应激')
    ax.axhline(y=thresholds['中度热应激'], color='red', linestyle='--', alpha=0.7, label='中度应激')
    
    ax.set_title(f'{barn_id} THI趋势图', fontsize=12)
    ax.set_xlabel('时间', fontsize=10)
    ax.set_ylabel('THI指数', fontsize=10)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    img_buffer.seek(0)
    return img_buffer


def create_mortality_chart(prod_df, barn_id, total_livestock=10000, figsize=(6, 3), dpi=100):
    """生成死淘率曲线"""
    from utils.mortality_analysis import calculate_cumulative_mortality
    
    barn_data = calculate_cumulative_mortality(prod_df, barn_id, total_livestock)
    if barn_data.empty:
        return None
    
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax1.bar(barn_data['日龄'], barn_data['日死淘率(%)'], color='#ED7D31', alpha=0.6, label='日死淘率', width=0.8)
    ax1.set_xlabel('日龄', fontsize=10)
    ax1.set_ylabel('日死淘率(%)', color='#ED7D31', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#ED7D31')
    
    ax2 = ax1.twinx()
    ax2.plot(barn_data['日龄'], barn_data['累计死淘率(%)'], color='#4472C4', linewidth=2, label='累计死淘率')
    ax2.set_ylabel('累计死淘率(%)', color='#4472C4', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#4472C4')
    
    ax1.set_title(f'{barn_id} 死淘率曲线', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    img_buffer.seek(0)
    return img_buffer


def generate_daily_report(env_df, prod_df, status_df, anomaly_summary, risk_df, 
                          livestock_type='肉鸡', total_livestock=10000, 
                          report_date=None, barn_ids=None):
    """生成养殖环境日报 PDF"""
    font_name = register_chinese_font()
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                            leftMargin=2*cm, rightMargin=2*cm, 
                            topMargin=2*cm, bottomMargin=2*cm)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'], fontName=font_name, fontSize=18, spaceAfter=20
    )
    h2_style = ParagraphStyle(
        'CustomH2', parent=styles['Heading2'], fontName=font_name, fontSize=14, 
        spaceBefore=15, spaceAfter=10, textColor=colors.HexColor('#4472C4')
    )
    normal_style = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=16
    )
    
    story = []
    
    if report_date is None:
        report_date = datetime.now().strftime('%Y年%m月%d日')
    
    story.append(Paragraph(f'养殖环境与健康日报 - {report_date}', title_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f'畜种类型: {livestock_type}', normal_style))
    story.append(Paragraph(f'栋舍数量: {len(status_df)}栋', normal_style))
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph('一、各栋舍环境参数统计摘要', h2_style))
    summary_table = create_summary_table(status_df, font_name)
    if summary_table:
        story.append(summary_table)
    else:
        story.append(Paragraph('暂无数据', normal_style))
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph('二、当日告警汇总', h2_style))
    alert_table = create_alert_list(anomaly_summary, font_name)
    if alert_table:
        story.append(alert_table)
    else:
        story.append(Paragraph('暂无告警数据', normal_style))
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph('三、疾病风险评分与建议', h2_style))
    risk_table = create_risk_table(risk_df, font_name)
    if risk_table:
        story.append(risk_table)
    else:
        story.append(Paragraph('暂无风险数据', normal_style))
    
    story.append(PageBreak())
    
    story.append(Paragraph('四、关键图表', h2_style))
    
    if barn_ids is None:
        barn_ids = status_df['栋舍编号'].unique().tolist() if not status_df.empty else []
    
    for i, barn_id in enumerate(barn_ids[:3]):
        story.append(Paragraph(f'{barn_id} THI趋势图', ParagraphStyle(
            'ChartTitle', parent=styles['Heading3'], fontName=font_name, fontSize=12, spaceBefore=10
        )))
        
        thi_img = create_thi_chart(env_df, barn_id)
        if thi_img:
            story.append(Image(thi_img, width=15*cm, height=7*cm))
        
        story.append(Paragraph(f'{barn_id} 死淘率曲线', ParagraphStyle(
            'ChartTitle', parent=styles['Heading3'], fontName=font_name, fontSize=12, spaceBefore=10
        )))
        
        mort_img = create_mortality_chart(prod_df, barn_id, total_livestock)
        if mort_img:
            story.append(Image(mort_img, width=15*cm, height=7*cm))
        
        if i < len(barn_ids[:3]) - 1:
            story.append(Spacer(1, 0.3*cm))
    
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph('报告生成时间: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                           ParagraphStyle('Footer', parent=styles['Normal'], fontName=font_name, 
                                         fontSize=8, textColor=colors.grey)))
    
    doc.build(story)
    buffer.seek(0)
    
    return buffer
