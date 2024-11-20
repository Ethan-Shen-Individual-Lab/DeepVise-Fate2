import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # 导入 PillowWriter
import tempfile
import fate
from flask import Flask, jsonify, send_from_directory, send_file, request, render_template, Response
from flask_cors import CORS
import mysql.connector
import redis
import json
import random
from datetime import datetime, timedelta
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import io
import base64
import subprocess
import numpy as np
from flask import Flask, render_template, send_from_directory, redirect
from flask_cors import CORS
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from fate_client.pipeline import FateFlowPipeline
import requests
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import HeteroSecureBoost, Reader, PSI, Evaluation
from matplotlib import rcParams
import sys
from io import StringIO
import time
import threading


# 设置字体为 SimHei
rcParams['font.family'] = 'SimHei'  # 这是常用的中文字体，确保字体已装
rcParams['axes.unicode_minus'] = False  # 处理负号问题

# 创建 Flask 应用，设置模板文件夹和静态文件夹路径

flask_app = Flask(
    __name__,
    template_folder='../frontend',  # 设置模板文件夹路径，调整为相对于 backend\py3820 的路径
    static_folder='../frontend',    # 设置静态文件夹路径
    static_url_path='/frontend'        # 静态文件的 URL 路径前缀保持不变
)


# 配置用于提供 figures 文件夹中图片的静态文件路径
flask_app.config['FIGURES_FOLDER'] = '../figures'  # 设置 figures 文件夹路径

CORS(flask_app)  # 启用跨域支持

# 默认路由，访问根路径时跳转到 intro 页
@flask_app.route('/')
def default_redirect():
    return redirect('/frontend/intro')  # 默认跳转到 intro 页

# 路由定义
@flask_app.route('/frontend/index', endpoint='index_page')
def index_page():
    return render_template('index.html')  # 首页

@flask_app.route('/frontend/intro', endpoint='intro_page')
def intro_page():
    return render_template('intro.html')  # 介绍页面

@flask_app.route('/frontend/health_monitor', endpoint='health_monitor_page')
def health_monitor_page():
    return render_template('health_monitor.html')  # 健康监测页面

@flask_app.route('/frontend/maintenance_optimization', endpoint='maintenance_optimization_page')
def maintenance_optimization_page():
    return render_template('maintenance_optimization.html')  # 维护优化页面

@flask_app.route('/frontend/privacy_security_management', endpoint='privacy_security_management_page')
def privacy_security_management_page():
    return render_template('privacy_security_management.html')  # 数据安全页面

@flask_app.route('/frontend/production_scheduling', endpoint='production_scheduling_page')
def production_scheduling_page():
    return render_template('production_scheduling.html')  # 产能预测与调度页面

@flask_app.route('/frontend/quality_control_defect_prediction', endpoint='quality_control_defect_prediction_page')
def quality_control_defect_prediction_page():
    return render_template('quality_control_defect_prediction.html')  # 质量控制页面

# 路由定义用于访问 figures 文件夹中的图片
@flask_app.route('/figures/<filename>')
def serve_figure(filename):
    return send_from_directory(flask_app.config['FIGURES_FOLDER'], filename)




# 数据库配置
REMOTE_DB_CONFIG = {
    'host': 'dbconn.sealosbja.site',
    'user': 'root',
    'password': '5559vgr',
    'database': 'press_machine_db',
    'port': 32723
}

LOCAL_DB_CONFIG = {
    'host': 'fintecthon-db-mysql.ns-shzmhcci.svc',
    'user': 'root',
    'password': '5559vgrs',
    'database': 'press_machine_db', 
    'port':3306
}

def connect_to_database():
    try:
        # 尝试连接到外网数据库
        connection = mysql.connector.connect(**REMOTE_DB_CONFIG)
        print("成功连接到外网数据库")
        return connection
    except mysql.connector.Error as err:
        print(f"连接外网数据库失败: {err}")
        print("尝试连接到本地数据库...")
        try:
            # 如果连接失，尝试连接到本地数据库
            connection = mysql.connector.connect(**LOCAL_DB_CONFIG)
            print("成功连接到本地数据库")
            return connection
        except mysql.connector.Error as err:
            print(f"连接本地数据库失败: {err}")
            return None


# 读取数据
try:
    # 使用绝对路径读取原始数据集
    df = pd.read_csv('../111/dataset/2023.7.24.csv', encoding='utf-8')
    print(f"Successfully read the original dataset. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows of original dataset:\n{df.head()}")

    # 读取拆分后的数据集
    guest_df = pd.read_csv('../111/dataset/guest_data.csv', encoding='utf-8')
    host_df = pd.read_csv('../111/dataset/host_data.csv', encoding='utf-8')
    print(f"\nSuccessfully read guest dataset. Shape: {guest_df.shape}")
    print(f"Successfully read host dataset. Shape: {host_df.shape}")
    print(f"\nFirst few rows of guest dataset:\n{guest_df.head()}")
    print(f"\nFirst few rows of host dataset:\n{host_df.head()}")

except FileNotFoundError as e:
    print(f"Error reading the dataset: {str(e)}")
    df = pd.DataFrame()  # 创建空的 DataFrame
    guest_df = pd.DataFrame()
    host_df = pd.DataFrame()
except Exception as e:
    print(f"Error reading the dataset: {str(e)}")
    df = pd.DataFrame()  # 创建空的 DataFrame
    guest_df = pd.DataFrame()
    host_df = pd.DataFrame()



# 设置图表样式
def create_motor_temperature_chart(df):
    try:
        # 只使用最后7行数据
        df = df.tail(7)

        # 检查数据是否足够
        if df.shape[0] < 7:  # 确保至少有7行数据
            print("数据不足，无法生成图表")
            return None

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor('none')  # 背景透明
        ax.spines['top'].set_color('white')  # 白色边框
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(axis='x', colors='white')  # x轴字体为白色
        ax.tick_params(axis='y', colors='white')  # y轴字体为白色
        ax.set_xlabel('时间', color='white')  # 设置x轴标签
        ax.set_ylabel('电机温度 (℃)', color='white')  # 设置y轴标签

        # 用来存储时间和温度数据
        time_data = df['时间'].tolist()
        temperature_data = df['电机模块电机温度'].tolist()

        # 绘制电机温度数据
        ax.plot(time_data, temperature_data, color='purple', lw=2)  # 亮紫色
        ax.fill_between(time_data, temperature_data, color='purple', alpha=0.1)  # 光影效果（半透明填充）

        # 格式化x轴时间显示
        plt.xticks(rotation=45)

        # 设置图表的布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.9)  # 调整边距

        # 临时保存图表文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            tmpfile_path = tmpfile.name
            plt.savefig(tmpfile_path, format='png', transparent=True)  # 保存PNG格式

        # 读取临时文件返回图像
        with open(tmpfile_path, 'rb') as f:
            img = io.BytesIO(f.read())
        
        return img

    except Exception as e:
        print(f'Error in creating chart: {e}')
        return None


@flask_app.route('/api/data_description')
def data_description():
    try:
        # 计算描述统计量
        description = host_df[['电机模块电机温度','电机速度实际值(RPM)','高速轴前轴承轴温','高速轴后轴承轴温']].describe(include='all').to_dict()  # 计算所有列的描述统计量并转换为字典

        # 格式化描述结果
        formatted_description = {}
        for key, value in description.items():
            formatted_description[key] = {
                "均值": f"{value['mean']:.2f}" if 'mean' in value else "N/A",
                "方差": f"{value['std']:.2f}" if 'std' in value else "N/A",
                "最小值": f"{value['min']:.2f}" if 'min' in value else "N/A",
                "最大值": f"{value['max']:.2f}" if 'max' in value else "N/A",
                "计数": value['count'],
                "25%": f"{value['25%']:.2f}" if '25%' in value else "N/A",
                "50%": f"{value['50%']:.2f}" if '50%' in value else "N/A",
                "75%": f"{value['75%']:.2f}" if '75%' in value else "N/A"
            }

        return jsonify(formatted_description)  # 返回 JSON 应
    except Exception as e:
        print(f"Error in generating data description: {e}")
        return jsonify({"error": str(e)}), 500

# 创建动态图表的路由
@flask_app.route('/api/motor_temperature_dynamic_chart')
def motor_temperature_dynamic_chart():
    img = create_motor_temperature_chart(host_df.copy())
    if img:
        return send_file(img, mimetype='image/gif')  # 返回GIF图像
    else:
        return "Error in creating dynamic chart", 500

# FATE Flow的IP和端口
FATE_FLOW_IP = '127.0.0.1'
FATE_FLOW_PORT = 9380

# 构建FATE Flow的URL
FATE_FLOW_URL = f'http://{FATE_FLOW_IP}:{FATE_FLOW_PORT}'

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode(errors='ignore'))  # 忽略无法解码的字节
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error: {e.stderr.decode(errors='ignore')}")  # 忽略无法解码的字节
        # 提示用户检查FATE安装
        print("请确保FATE已正确安装，并且命令可以在命令行中运行。")

def initialize_fate_services():
    # 初始化FATE Flow和Pipeline
    run_command(f"fate_flow init --ip {FATE_FLOW_IP} --port {FATE_FLOW_PORT}")
    run_command(f"pipeline init --ip {FATE_FLOW_IP} --port {FATE_FLOW_PORT}")
    run_command("fate_flow start")
    # 检查FATE Flow服务状态
    check_fate_flow_status()

def check_fate_flow_status():
    try:
        response = requests.get(f'{FATE_FLOW_URL}/v1/version/get')
        if response.status_code == 200:
            print("成功连接到FATE Flow服务")
        else:
            print("无法连接到FATE Flow服务")
    except Exception as e:
        print(f"连接FATE Flow服务时出错: {e}")


def run_federated_learning():
    try:
        # 数据准备
        guest_data_path = '../111/dataset/guest_data.csv'
        host_data_path = '../111/dataset/host_data.csv'

        data_pipeline = FateFlowPipeline().set_parties(local="0")
        guest_meta = {
            "delimiter": ",", "dtype": "float64", "label_type": "int64","label_name": "状态", "match_id_name": "id"
        }
        host_meta = {
            "delimiter": ",", "input_format": "dense", "match_id_name": "id"
        }
        guest_result = data_pipeline.transform_local_file_to_dataframe(file=guest_data_path, namespace="experiment", name="breast_hetero_guest",
                                                        meta=guest_meta, head=True, extend_sid=True)
        host_result = data_pipeline.transform_local_file_to_dataframe(file=host_data_path, namespace="experiment", name="breast_hetero_host",
                                                        meta=host_meta, head=True, extend_sid=True)

        # create pipeline for training
        pipeline = FateFlowPipeline().set_parties(guest="9999", host="10000")
        federated_learning_status(pipeline)
        # create reader task_desc
        reader_0 = Reader("reader_0")
        reader_0.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
        reader_0.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")

        # create psi component_desc
        psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])

        # create hetero secure_boost component_desc
        hetero_secureboost_0 = HeteroSecureBoost(
            "hetero_secureboost_0", num_trees=1, max_depth=5,
            train_data=psi_0.outputs["output_data"],
            validate_data=psi_0.outputs["output_data"]
        )

        # create evaluation component_desc
        evaluation_0 = Evaluation(
            'evaluation_0', runtime_parties=dict(guest="9999"), metrics=["auc"], input_datas=[hetero_secureboost_0.outputs["train_output_data"]]
        )

        # add training task
        pipeline.add_tasks([reader_0, psi_0, hetero_secureboost_0, evaluation_0])

        # compile and train
        pipeline.compile()
        pipeline.fit()

        # print metric and model info
        print(pipeline.get_task_info("hetero_secureboost_0").get_output_model())
        print(pipeline.get_task_info("evaluation_0").get_output_metric())

        # save pipeline for later usage
        pipeline.dump_model("./pipeline.pkl")

        # create pipeline for predicting
        predict_pipeline = FateFlowPipeline()

        # reload trained pipeline
        pipeline = FateFlowPipeline.load_model("./pipeline.pkl")

        # deploy task for inference
        pipeline.deploy([pipeline.psi_0, pipeline.hetero_secureboost_0])

        # add input to deployed_pipeline
        deployed_pipeline = pipeline.get_deployed_pipeline()
        reader_1 = Reader("reader_1")
        reader_1.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
        reader_1.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")
        deployed_pipeline.psi_0.input_data = reader_1.outputs["output_data"]

        # add task to predict pipeline
        predict_pipeline.add_tasks([reader_1, deployed_pipeline])

        # compile and predict
        predict_pipeline.compile()
        predict_pipeline.predict()

        return True, "联邦学习执行成功"
    except Exception as e:
        return False, f"联邦学习执行失败: {str(e)}"


# 在文件顶部添加全局变量
current_task_id = 0
task_start_times = {}  # 用于存储每个任务的开始时间

# 修改 federated_learning 路由
@flask_app.route('/api/federated_learning', methods=['GET'])
def federated_learning():
    try:
        global current_task_id
        current_task_id += 1
        task_start_times[current_task_id] = datetime.now()
        
        print(f"创建新任务 - ID: {current_task_id}")  # 更详细的日志
        print(f"任务开始时间: {task_start_times[current_task_id]}")
        
        success, message = run_federated_learning()
        
        response_data = {
            'success': True,
            'message': message,
            'details': {
                'status': '完成',
                'taskId': current_task_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        print(f"返回响应数据: {response_data}")  # 输出响应数据
        return jsonify(response_data)
        
    except Exception as e:
        error_message = str(e)
        print(f"任务创建失败 - ID: {current_task_id}, 错误: {error_message}")
        return jsonify({
            'success': False,
            'message': error_message,
            'details': {
                'error_type': type(e).__name__,
                'taskId': current_task_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }), 500

# 修改 federated_learning_status 路由
@flask_app.route('/api/federated_learning/status', methods=['GET'])
def federated_learning_status():  # 移除 pipeline 参数
    def generate():
        try:
            task_id = current_task_id
            start_time = task_start_times.get(task_id, datetime.now())
            
            # 发送初始任务ID
            yield f"data: {json.dumps({'taskId': task_id, 'elapsedTime': 0, 'status': '已启动'})}\n\n"
            
            # 模拟任务进行，每秒更新状态
            while True:
                current_time = datetime.now()
                elapsed_seconds = int((current_time - start_time).total_seconds())
                
                status_data = {
                    'taskId': task_id,
                    'elapsedTime': elapsed_seconds,
                    'status': '进行中'
                }
                
                yield f"data: {json.dumps(status_data)}\n\n"
                
                # 模拟任务完成
                if elapsed_seconds >= 30:  # 30秒后完成
                    final_status = {
                        'taskId': task_id,
                        'elapsedTime': elapsed_seconds,
                        'status': '完成'
                    }
                    yield f"data: {json.dumps(final_status)}\n\n"
                    break
                
                time.sleep(1)  # 每秒更新一次
            
        except Exception as e:
            error_status = {
                'taskId': task_id,
                'error': str(e),
                'status': '执行出错'
            }
            yield f"data: {json.dumps(error_status)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@flask_app.route('/api/host-status', methods=['GET'])
def host_status():
    node_counts = host_df['id'].nunique()
    data_counts = host_df.shape[0]
    return jsonify({'node_counts': node_counts, 'data_counts': data_counts})

@flask_app.route('/api/guest-nodes-count', methods=['GET'])
def guest_nodes_count():
    num_guest_nodes = guest_df['id'].nunique()
    num_guest_data_counts = guest_df.shape[0]
    return jsonify({'count': num_guest_nodes,'data_counts': num_guest_data_counts})

@flask_app.route('/api/federal-learning', methods=['POST'])
def federal_learning():
    try:
        config = request.json
        
        # 获取配置参数
        data_source = config['dataSource']
        node_count = config['nodeCount']
        min_data = config['minData']
        epochs = config['epochs']
        learning_rate = config['learningRate']
        model_type = config['modelType']
        hidden_layers = config['hiddenLayers']
        
        # 根据数据源类型选择数据
        if data_source == 'local':
            # 仅使用本地数据
            training_data = host_df
        else:
            # 使用联邦数据
            # 根据node_count选择指定数量的外部节点数据
            selected_guest_nodes = guest_df['id'].unique()[:node_count]
            training_data = pd.concat([
                host_df,
                guest_df[guest_df['id'].isin(selected_guest_nodes)]
            ])
        
        # 这里添加实际的联邦学习训练逻辑
        # ...
        
        return jsonify({
            'success': True,
            'message': '联邦学习训练已启动'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/features', methods=['GET'])
def features():
    try:
        # 获取数据集的特征列，排除'id'列
        features = [col for col in host_df.columns if col != 'id']
        print(f"Available features: {features}")  # 添加调试输出
        return jsonify({
            'success': True,
            'features': features
        })
    except Exception as e:
        print(f"Error getting features: {str(e)}")  # 添加错误日志
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@flask_app.route('/api/run_federal_learning', methods=['POST'])
def handle_federal_learning():
    try:
        # 获取前端传来的参数
        data = request.get_json()
        task_id = data.get('taskId')
        
        # 异步执行联邦学习任务
        # 这里可以使用线程或异步任务来执行，避免阻塞
        thread = threading.Thread(target=run_federated_learning)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '联邦学习任务已启动',
            'taskId': task_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'启动联邦学习任务失败: {str(e)}'
        }), 500

# 在主程序中也使用新的函数
if __name__ == '__main__':
    '''
    print("\n=== 开始测试联邦学习系统 ===")
    try:
        # 直接调用封装的函数
        success, message = run_federated_learning()
        if success:
            print(f"联邦学习测试成功: {message}")
        else:
            print(f"联邦学习测试失败: {message}")
    except Exception as e:
        print(f"联邦学习系统测试出错: {str(e)}")
    finally:
        print("=== 联邦学习系统测试结束 ===\n")
    '''
    # 初始化FATE服务并启动Flask应用
    initialize_fate_services()
    flask_app.run(host='0.0.0.0', port=5000, debug=True)

