import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 加载数据文件
file_33AH_001 = pd.read_csv('33AH_001.csv')
file_33AH_006 = pd.read_csv('33AH_006.csv')
file_exchange_001 = pd.read_csv('33AH_exchange_001.csv')
file_exchange_006 = pd.read_csv('33AH_exchange_006.csv')

# 将“总运行时间”转换为秒，便于绘图
def runtime_to_seconds(runtime_str):
    days, time = runtime_str.split('d ')
    hours, minutes, seconds = map(int, time.split(':'))
    total_seconds = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds

# 对所有数据文件应用转换
for df in [file_33AH_001, file_33AH_006, file_exchange_001, file_exchange_006]:
    df['总运行时间(秒)'] = df['Total Runtime'].apply(runtime_to_seconds)

# 初始化Dash应用
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("电池测试数据可视化"),

    # 创建选项卡来展示不同数据文件的分析
    dcc.Tabs([
        dcc.Tab(label='33AH_001设备', children=[
            dcc.Graph(id='runtime-voltage-current-001'),
            dcc.Graph(id='temperature-runtime-001'),
            dcc.Graph(id='capacity-cycle-001'),
            dcc.Graph(id='capacity-retention-001'),
            dcc.Graph(id='voltage-capacity-001'),
            dcc.Graph(id='current-capacity-001'),
            dcc.Graph(id='temperature-capacity-001'),
            dcc.Graph(id='cumulative-capacity-001'),
            dcc.Graph(id='dqdv-v-001'),
            dcc.Graph(id='dqdv-capacity-001'),
            dcc.Graph(id='temperature-heatmap-001'),
            dcc.Graph(id='correlation-heatmap-001'),
            dcc.Graph(id='voltage-capacity-scatter-001'),
            dcc.Graph(id='current-capacity-scatter-001'),
            dcc.Graph(id='temperature-capacity-scatter-001'),
            dcc.Graph(id='cycle-capacity-regression-001'),
            dcc.Graph(id='runtime-capacity-001'),
            dcc.Graph(id='cycle-voltage-current-001'),
            dcc.Graph(id='cycle-capacity-001'),
            dcc.Graph(id='temperature-voltage-current-001')

        ]),
        dcc.Tab(label='33AH_006设备', children=[
            dcc.Graph(id='runtime-voltage-current-006'),
            dcc.Graph(id='temperature-runtime-006'),
            dcc.Graph(id='capacity-cycle-006'),
            dcc.Graph(id='capacity-retention-006'),
            dcc.Graph(id='voltage-capacity-006'),
            dcc.Graph(id='current-capacity-006'),
            dcc.Graph(id='temperature-capacity-006'),
            dcc.Graph(id='cumulative-capacity-006'),
            dcc.Graph(id='dqdv-v-006'),
            dcc.Graph(id='dqdv-capacity-006'),
            dcc.Graph(id='temperature-heatmap-006'),
            dcc.Graph(id='correlation-heatmap-006'),
            dcc.Graph(id='voltage-capacity-scatter-006'),
            dcc.Graph(id='current-capacity-scatter-006'),
            dcc.Graph(id='temperature-capacity-scatter-006'),
            dcc.Graph(id='cycle-capacity-regression-006'),
            dcc.Graph(id='runtime-capacity-006'),
            dcc.Graph(id='cycle-voltage-current-006'),
            dcc.Graph(id='cycle-capacity-006'),
            dcc.Graph(id='temperature-voltage-current-006')
        ]),
        dcc.Tab(label='33AH_001对调设备', children=[
            dcc.Graph(id='runtime-voltage-current-ex-001'),
            dcc.Graph(id='temperature-runtime-ex-001'),
            dcc.Graph(id='capacity-cycle-ex-001'),
            dcc.Graph(id='capacity-retention-ex-001'),
            dcc.Graph(id='voltage-capacity-ex-001'),
            dcc.Graph(id='current-capacity-ex-001'),
            dcc.Graph(id='temperature-capacity-ex-001'),
            dcc.Graph(id='cumulative-capacity-ex-001'),
            dcc.Graph(id='dqdv-v-ex-001'),
            dcc.Graph(id='dqdv-capacity-ex-001'),
            dcc.Graph(id='temperature-heatmap-ex-001'),
            dcc.Graph(id='correlation-heatmap-ex-001'),
            dcc.Graph(id='voltage-capacity-scatter-ex-001'),
            dcc.Graph(id='current-capacity-scatter-ex-001'),
            dcc.Graph(id='temperature-capacity-scatter-ex-001'),
            dcc.Graph(id='cycle-capacity-regression-ex-001'),
            dcc.Graph(id='runtime-capacity-ex-001'),
            dcc.Graph(id='cycle-voltage-current-ex-001'),
            dcc.Graph(id='cycle-capacity-ex-001'),
            dcc.Graph(id='temperature-voltage-current-ex-001')
        ]),
        dcc.Tab(label='33AH_006对调设备', children=[
            dcc.Graph(id='runtime-voltage-current-ex-006'),
            dcc.Graph(id='temperature-runtime-ex-006'),
            dcc.Graph(id='capacity-cycle-ex-006'),
            dcc.Graph(id='capacity-retention-ex-006'),
            dcc.Graph(id='voltage-capacity-ex-006'),
            dcc.Graph(id='current-capacity-ex-006'),
            dcc.Graph(id='temperature-capacity-ex-006'),
            dcc.Graph(id='cumulative-capacity-ex-006'),
            dcc.Graph(id='dqdv-v-ex-006'),
            dcc.Graph(id='dqdv-capacity-ex-006'),
            dcc.Graph(id='temperature-heatmap-ex-006'),
            dcc.Graph(id='correlation-heatmap-ex-006'),
            dcc.Graph(id='voltage-capacity-scatter-ex-006'),
            dcc.Graph(id='current-capacity-scatter-ex-006'),
            dcc.Graph(id='temperature-capacity-scatter-ex-006'),
            dcc.Graph(id='cycle-capacity-regression-ex-006'),
            dcc.Graph(id='runtime-capacity-ex-006'),
            dcc.Graph(id='cycle-voltage-current-ex-006'),
            dcc.Graph(id='cycle-capacity-ex-006'),
            dcc.Graph(id='temperature-voltage-current-ex-006')
        ])
    ])
])


@app.callback(
    Output('runtime-voltage-current-001', 'figure'),
    Output('temperature-runtime-001', 'figure'),
    Output('capacity-cycle-001', 'figure'),
    Output('capacity-retention-001', 'figure'),
    Output('voltage-capacity-001', 'figure'),
    Output('current-capacity-001', 'figure'),
    Output('temperature-capacity-001', 'figure'),
    Output('cumulative-capacity-001', 'figure'),
    Output('dqdv-v-001', 'figure'),
    Output('dqdv-capacity-001', 'figure'),
    Output('temperature-heatmap-001', 'figure'),
    Output('correlation-heatmap-001', 'figure'),
    Output('voltage-capacity-scatter-001', 'figure'),
    Output('current-capacity-scatter-001', 'figure'),
    Output('temperature-capacity-scatter-001', 'figure'),
    Output('cycle-capacity-regression-001', 'figure'),
    Output('runtime-capacity-001', 'figure'),
    Output('cycle-voltage-current-001', 'figure'),
    Output('cycle-capacity-001', 'figure'),
    Output('temperature-voltage-current-001', 'figure'),
    Output('runtime-voltage-current-006', 'figure'),
    Output('temperature-runtime-006', 'figure'),
    Output('capacity-cycle-006', 'figure'),
    Output('capacity-retention-006', 'figure'),
    Output('voltage-capacity-006', 'figure'),
    Output('current-capacity-006', 'figure'),
    Output('temperature-capacity-006', 'figure'),
    Output('cumulative-capacity-006', 'figure'),
    Output('dqdv-v-006', 'figure'),
    Output('dqdv-capacity-006', 'figure'),
    Output('temperature-heatmap-006', 'figure'),
    Output('correlation-heatmap-006', 'figure'),
    Output('voltage-capacity-scatter-006', 'figure'),
    Output('current-capacity-scatter-006', 'figure'),
    Output('temperature-capacity-scatter-006', 'figure'),
    Output('cycle-capacity-regression-006', 'figure'),
    Output('runtime-capacity-006', 'figure'),
    Output('cycle-voltage-current-006', 'figure'),
    Output('cycle-capacity-006', 'figure'),
    Output('temperature-voltage-current-006', 'figure'),
    Output('runtime-voltage-current-ex-001', 'figure'),
    Output('temperature-runtime-ex-001', 'figure'),
    Output('capacity-cycle-ex-001', 'figure'),
    Output('capacity-retention-ex-001', 'figure'),
    Output('voltage-capacity-ex-001', 'figure'),
    Output('current-capacity-ex-001', 'figure'),
    Output('temperature-capacity-ex-001', 'figure'),
    Output('cumulative-capacity-ex-001', 'figure'),
    Output('dqdv-v-ex-001', 'figure'),
    Output('dqdv-capacity-ex-001', 'figure'),
    Output('temperature-heatmap-ex-001', 'figure'),
    Output('correlation-heatmap-ex-001', 'figure'),
    Output('voltage-capacity-scatter-ex-001', 'figure'),
    Output('current-capacity-scatter-ex-001', 'figure'),
    Output('temperature-capacity-scatter-ex-001', 'figure'),
    Output('cycle-capacity-regression-ex-001', 'figure'),
    Output('runtime-capacity-ex-001', 'figure'),
    Output('cycle-voltage-current-ex-001', 'figure'),
    Output('cycle-capacity-ex-001', 'figure'),
    Output('temperature-voltage-current-ex-001', 'figure'),
    Output('runtime-voltage-current-ex-006', 'figure'),
    Output('temperature-runtime-ex-006', 'figure'),
    Output('capacity-cycle-ex-006', 'figure'),
    Output('capacity-retention-ex-006', 'figure'),
    Output('voltage-capacity-ex-006', 'figure'),
    Output('current-capacity-ex-006', 'figure'),
    Output('temperature-capacity-ex-006', 'figure'),
    Output('cumulative-capacity-ex-006', 'figure'),
    Output('dqdv-v-ex-006', 'figure'),
    Output('dqdv-capacity-ex-006', 'figure'),
    Output('temperature-heatmap-ex-006', 'figure'),
    Output('correlation-heatmap-ex-006', 'figure'),
    Output('voltage-capacity-scatter-ex-006', 'figure'),
    Output('current-capacity-scatter-ex-006', 'figure'),
    Output('temperature-capacity-scatter-ex-006', 'figure'),
    Output('cycle-capacity-regression-ex-006', 'figure'),
    Output('runtime-capacity-ex-006', 'figure'),
    Output('cycle-voltage-current-ex-006', 'figure'),
    Output('cycle-capacity-ex-006', 'figure'),
    Output('temperature-voltage-current-ex-006', 'figure'),
    Input('runtime-voltage-current-001', 'id')  # 虚拟输入以触发回调
)
def update_graphs(_):
    def create_figures(df, use_second_cycle_as_initial=False):
        # 将列名转换为中文
        df_renamed = df.rename(columns={
            'Total voltage (V)': '总电压 (V)',
            'Current (A)': '电流 (A)',
            'Charging capacity (AH)': '充电容量 (AH)',
            'Discharge capacity (AH)': '放电容量 (AH)',
            'Temperature': '温度 (°C)',
            'Cycle': '循环次数',
            'Charge or discharge': '充电或放电'
        })

        # 将 "CH" 和 "DIS" 替换为 "充电" 和 "放电"
        df_renamed['充电或放电'] = df_renamed['充电或放电'].replace({'CH': '充电', 'DIS': '放电'})

        # 获取每次充电和放电结束时的容量
        charge_end_capacity = df_renamed[df_renamed['充电容量 (AH)'].notna()].groupby('循环次数')['充电容量 (AH)'].max().reset_index()
        discharge_end_capacity = df_renamed[df_renamed['放电容量 (AH)'].notna()].groupby('循环次数')['放电容量 (AH)'].max().reset_index()

        # 合并充电和放电数据到同一数据框
        combined_capacity = pd.merge(charge_end_capacity, discharge_end_capacity, on='循环次数', how='outer', suffixes=('_充电', '_放电'))

        # 计算容量保持率
        if use_second_cycle_as_initial:
            initial_charge_capacity = combined_capacity['充电容量 (AH)'].iloc[1]
            initial_discharge_capacity = combined_capacity['放电容量 (AH)'].iloc[1]
        else:
            initial_charge_capacity = combined_capacity['充电容量 (AH)'].iloc[0]
            initial_discharge_capacity = combined_capacity['放电容量 (AH)'].iloc[0]

        combined_capacity['充电容量保持率 (%)'] = (combined_capacity['充电容量 (AH)'] / initial_charge_capacity) * 100
        combined_capacity['放电容量保持率 (%)'] = (combined_capacity['放电容量 (AH)'] / initial_discharge_capacity) * 100

        # 总运行时间 vs 总电压和电流
        fig_runtime_voltage_current = px.line(df_renamed, x='总运行时间(秒)', y=['总电压 (V)', '电流 (A)'],
                                              labels={'value': '值', 'variable': ' '},
                                              title='总运行时间 vs 总电压和电流')

        # 总运行时间 vs 温度
        fig_temperature_time = px.line(df_renamed, x='总运行时间(秒)', y='温度 (°C)',
                                       labels={'总运行时间(秒)': '总运行时间 (秒)', '温度 (°C)': '温度 (°C)'},
                                       title='总运行时间 vs 温度')

        # 充电和放电结束时的容量随循环次数变化的曲线
        fig_capacity_cycle = px.line(combined_capacity, x='循环次数', y=['充电容量 (AH)', '放电容量 (AH)'],
                                     labels={'value': '容量 (AH)', 'variable': ' ', '循环次数': '循环次数'},
                                     title='容量衰减曲线')
        fig_capacity_cycle.update_traces(mode='lines+markers')

        # 循环次数 vs 容量保持率
        fig_capacity_retention = px.line(combined_capacity, x='循环次数',
                                         y=['充电容量保持率 (%)', '放电容量保持率 (%)'],
                                         labels={'value': '容量保持率 (%)', 'variable': ' ', '循环次数': '循环次数'},
                                         title='循环次数 vs 容量保持率')
        fig_capacity_retention.update_traces(mode='lines+markers')

        # 筛选出充电和放电状态的数据
        df_charge = df_renamed[df_renamed['充电或放电'] == '充电']
        df_discharge = df_renamed[df_renamed['充电或放电'] == '放电']

        # 获取共有的循环次数
        common_cycles = np.intersect1d(df_charge['循环次数'].dropna().unique(),
                                       df_discharge['循环次数'].dropna().unique())

        # 随机选择5个共有的循环次数
        random_cycles = np.random.choice(common_cycles, 5, replace=False)

        # 筛选出这5个共有的循环次数的数据
        df_charge_filtered = df_charge[df_charge['循环次数'].isin(random_cycles)]
        df_discharge_filtered = df_discharge[df_discharge['循环次数'].isin(random_cycles)]

        # 电压 vs 充电容量和放电容量
        fig_voltage_capacity = px.line(df_charge_filtered, x='充电容量 (AH)', y='总电压 (V)', color='循环次数',
                                       labels={'充电容量 (AH)': '容量 (AH)', '总电压 (V)': '总电压 (V)',
                                               '循环次数': '循环次数'},
                                       title='电压 vs 容量')
        fig_voltage_capacity.add_traces(
            px.line(df_discharge_filtered, x='放电容量 (AH)', y='总电压 (V)', color='循环次数').data)

        # 电流 vs 充电容量和放电容量
        fig_current_capacity = px.line(df_charge_filtered, x='充电容量 (AH)', y='电流 (A)', color='循环次数',
                                       labels={'充电容量 (AH)': '容量 (AH)', '电流 (A)': '电流 (A)',
                                               '循环次数': '循环次数'},
                                       title='电流 vs 容量')
        fig_current_capacity.add_traces(
            px.line(df_discharge_filtered, x='放电容量 (AH)', y='电流 (A)', color='循环次数').data)

        # 温度 vs 充电容量和放电容量
        fig_temperature_capacity = px.line(df_charge_filtered, x='充电容量 (AH)', y='温度 (°C)', color='循环次数',
                                           labels={'充电容量 (AH)': '容量 (AH)', '温度 (°C)': '温度 (°C)',
                                                   '循环次数': '循环次数'},
                                           title='温度 vs 容量')
        fig_temperature_capacity.add_traces(
            px.line(df_discharge_filtered, x='放电容量 (AH)', y='温度 (°C)', color='循环次数').data)

        # 累积容量曲线，使用相同的循环次数
        df_filtered = df_renamed[df_renamed['循环次数'].isin(random_cycles)].copy()
        df_filtered['累积容量 (AH)'] = df_filtered.apply(
            lambda row: row['充电容量 (AH)'] if row['充电或放电'] == '充电' else -row['放电容量 (AH)'], axis=1)
        df_filtered['累积容量 (AH)'] = df_filtered.groupby('循环次数')['累积容量 (AH)'].cumsum()

        fig_cumulative_capacity = px.line(df_filtered, x='累积容量 (AH)', y='总电压 (V)', color='循环次数',
                                          labels={'累积容量 (AH)': '累积容量 (AH)', '总电压 (V)': '总电压 (V)',
                                                  '循环次数': '循环次数'},
                                          title='累积容量 vs 电压')

        # 计算dQ/dV
        df_charge_filtered['dQ'] = df_charge_filtered['充电容量 (AH)'].diff()
        df_charge_filtered['dV'] = df_charge_filtered['总电压 (V)'].diff()
        df_charge_filtered['dQ/dV'] = df_charge_filtered['dQ'] / df_charge_filtered['dV']

        df_discharge_filtered['dQ'] = df_discharge_filtered['放电容量 (AH)'].diff()
        df_discharge_filtered['dV'] = df_discharge_filtered['总电压 (V)'].diff()
        df_discharge_filtered['dQ/dV'] = df_discharge_filtered['dQ'] / df_discharge_filtered['dV']

        # 绘制dQ/dV曲线(以电压为横坐标)
        fig_dqdv_v = go.Figure()

        for cycle in df_charge_filtered['循环次数'].unique():
            df_cycle = df_charge_filtered[df_charge_filtered['循环次数'] == cycle]
            fig_dqdv_v.add_trace(
                go.Scatter(x=df_cycle['总电压 (V)'], y=df_cycle['dQ/dV'], mode='lines', name=f'充电循环 {cycle}'))

        for cycle in df_discharge_filtered['循环次数'].unique():
            df_cycle = df_discharge_filtered[df_discharge_filtered['循环次数'] == cycle]
            fig_dqdv_v.add_trace(
                go.Scatter(x=df_cycle['总电压 (V)'], y=df_cycle['dQ/dV'], mode='lines', name=f'放电循环 {cycle}',
                           line=dict(dash='dash')))

        fig_dqdv_v.update_layout(title='dQ/dV曲线 (电压)', xaxis_title='电压 (V)', yaxis_title='dQ/dV')

        # 绘制dQ/dV曲线(以容量为横坐标)
        fig_dqdv_capacity = go.Figure()

        for cycle in df_charge_filtered['循环次数'].unique():
            df_cycle = df_charge_filtered[df_charge_filtered['循环次数'] == cycle]
            fig_dqdv_capacity.add_trace(
                go.Scatter(x=df_cycle['充电容量 (AH)'], y=df_cycle['dQ/dV'], mode='lines', name=f'充电循环 {cycle}'))

        for cycle in df_discharge_filtered['循环次数'].unique():
            df_cycle = df_discharge_filtered[df_discharge_filtered['循环次数'] == cycle]
            fig_dqdv_capacity.add_trace(
                go.Scatter(x=df_cycle['放电容量 (AH)'], y=df_cycle['dQ/dV'], mode='lines', name=f'放电循环 {cycle}',
                           line=dict(dash='dash')))

        fig_dqdv_capacity.update_layout(title='dQ/dV曲线 (容量)', xaxis_title='容量 (AH)', yaxis_title='dQ/dV')

        # 绘制温度热力图
        fig_temperature_heatmap = px.density_heatmap(df_renamed, x='充电或放电', y='循环次数', z='温度 (°C)',
                                                     title='温度热力图',
                                                     labels={'充电或放电': '充电或放电', '循环次数': '循环次数', '温度 (°C)': '温度 (°C)'})

        # 相关性热力图
        correlation = df_renamed[['总电压 (V)', '电流 (A)', '充电容量 (AH)', '放电容量 (AH)', '温度 (°C)', '总运行时间(秒)']].corr()
        fig_correlation_heatmap = px.imshow(correlation, text_auto=True, aspect="auto",
                                            labels={'color': '相关性'},
                                            title='参数间相关性热力图')

        # 电压与充放电容量的散点图
        cycle_25_data = df_renamed[df_renamed['循环次数'] == 25]
        fig_voltage_capacity_scatter = px.scatter(cycle_25_data, x=['充电容量 (AH)', '放电容量 (AH)'], y='总电压 (V)',
                                                  labels={'总电压 (V)': '总电压 (V)', 'value': '容量 (AH)', 'variable': ' '},
                                                  title='电压与充放电容量的关系')

        # 电流与充放电容量的散点图
        cycle_25_data = df_renamed[df_renamed['循环次数'] == 25]
        fig_current_capacity_scatter = px.scatter(cycle_25_data, x=['充电容量 (AH)', '放电容量 (AH)'], y='电流 (A)',
                                                  labels={'电流 (A)': '电流 (A)', 'value': '容量 (AH)',
                                                          'variable': ' '},
                                                  title='电流与充放电容量的关系')

        # 温度与充放电容量的散点图
        cycle_25_data = df_renamed[df_renamed['循环次数'] == 27]
        fig_temperature_capacity_regression = px.scatter(cycle_25_data, x=['充电容量 (AH)', '放电容量 (AH)'],y='温度 (°C)',
                                                         labels={'温度 (°C)': '温度 (°C)', 'value': '容量 (AH)',
                                                                 'variable': ' '},
                                                         title='温度与充放电容量的散点图')
        # 绘制循环次数与充电容量的线性回归图
        fig_capacity_regression = px.scatter(combined_capacity, x='循环次数', y='充电容量 (AH)',
                                             trendline='ols', color_discrete_sequence=['blue'],
                                             labels={'循环次数': '循环次数', '充电容量 (AH)': '容量 (AH)'})
        # 为充电容量图例命名
        fig_capacity_regression.update_traces(name='充电容量 (AH)')

        # 绘制循环次数与放电容量的线性回归图
        fig_discharge_regression = px.scatter(combined_capacity, x='循环次数', y='放电容量 (AH)',
                                              trendline='ols', color_discrete_sequence=['red'],
                                              labels={'循环次数': '循环次数', '放电容量 (AH)': '容量 (AH)'})
        # 为放电容量图例命名
        fig_discharge_regression.update_traces(name='放电容量 (AH)')

        # 将放电容量的回归线和散点添加到同一图中
        for trace in fig_discharge_regression.data:
            fig_capacity_regression.add_trace(trace)

        # 更新图例以区分充电和放电
        fig_capacity_regression.for_each_trace(lambda t: t.update(showlegend=True))

        fig_capacity_regression.update_layout(title='循环次数 vs 容量的线性回归图',
                                              xaxis_title='循环次数', yaxis_title='容量 (AH)',
                                              showlegend=True)

        # 总运行时间 vs 充电容量和放电容量
        fig_runtime_capacity = px.line(df_renamed, x='总运行时间(秒)', y=['充电容量 (AH)', '放电容量 (AH)'],
                                       labels={'value': '容量 (AH)', 'variable': ' '},
                                       title='总运行时间 vs 充电容量和放电容量')

        # 循环次数 vs 总电压和电流
        fig_cycle_voltage_current = px.line(df_renamed, x='循环次数', y=['总电压 (V)', '电流 (A)'],
                                            labels={'value': '值', 'variable': ' '},
                                            title='循环次数 vs 总电压和电流')

        # 循环次数 vs 充电容量和放电容量
        fig_cycle_capacity = px.line(df_renamed, x='循环次数', y=['充电容量 (AH)', '放电容量 (AH)'],
                                     labels={'value': '容量 (AH)', 'variable': ' '},
                                     title='循环次数 vs 充电容量和放电容量')

        # 温度 vs 总电压和电流
        fig_temperature_voltage_current = px.area(df_renamed, x='温度 (°C)', y=['总电压 (V)', '电流 (A)'],
                                                     labels={'value': '值', 'variable': ' '},
                                                     title='温度 vs 总电压和电流')

        return (fig_runtime_voltage_current, fig_temperature_time, fig_capacity_cycle, fig_capacity_retention,
                fig_voltage_capacity, fig_current_capacity, fig_temperature_capacity, fig_cumulative_capacity,
                fig_dqdv_v, fig_dqdv_capacity, fig_temperature_heatmap, fig_correlation_heatmap, fig_voltage_capacity_scatter,
                fig_current_capacity_scatter,fig_temperature_capacity_regression,fig_capacity_regression,fig_runtime_capacity,
                fig_cycle_voltage_current, fig_cycle_capacity,fig_temperature_voltage_current)

    # 创建所有图表
    return (create_figures(file_33AH_001) + create_figures(file_33AH_006) +
            create_figures(file_exchange_001) + create_figures(file_exchange_006, use_second_cycle_as_initial=True))

if __name__ == '__main__':
    app.run_server(debug=True)
