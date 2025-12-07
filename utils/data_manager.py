# utils/data_manager.py

import os
import pickle
import json
import pandas as pd
from typing import Optional, Any, Dict, List
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential

# --- 配置 ---
USER_DATA_DIR = "user_data" # 与 user_auth.py 中定义的保持一致

# --- 用户数据目录操作 ---
def get_user_data_path(username: str, subfolder: Optional[str] = None) -> str:
    """
    获取用户数据的根目录或子目录路径。
    例如: get_user_data_path('user1', 'datasets') -> 'user_data/user1/datasets'
    """
    base_path = os.path.join(USER_DATA_DIR, username)
    if subfolder:
        return os.path.join(base_path, subfolder)
    return base_path

def ensure_user_subdir_exists(username: str, subfolder: str):
    """
    确保用户的某个子目录存在，如果不存在则创建。
    """
    path = get_user_data_path(username, subfolder)
    os.makedirs(path, exist_ok=True)
    print(f"[DEBUG] Ensured directory exists: {path}") # 调试信息

# --- 数据集 (Dataset) 操作 ---
def save_dataset(username: str, filename: str, df: pd.DataFrame):
    """
    将 pandas DataFrame 保存为 CSV 文件到指定用户的 datasets 目录。
    """
    datasets_dir = get_user_data_path(username, "datasets")
    os.makedirs(datasets_dir, exist_ok=True) # 确保目录存在
    file_path = os.path.join(datasets_dir, filename)
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"[INFO] Dataset saved to {file_path}")

def load_dataset(username: str, filename: str) -> Optional[pd.DataFrame]:
    """
    从指定用户的 datasets 目录加载 CSV 文件为 pandas DataFrame。
    """
    file_path = os.path.join(get_user_data_path(username, "datasets"), filename)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"[INFO] Dataset loaded from {file_path}")
            return df
        except Exception as e:
            print(f"[ERROR] Failed to load dataset {file_path}: {e}")
            return None
    else:
        print(f"[WARNING] Dataset file {file_path} does not exist.")
        return None

def list_user_datasets(username: str) -> List[str]:
    """
    列出指定用户的所有数据集文件名 (.csv, .xlsx)。
    """
    datasets_dir = get_user_data_path(username, "datasets")
    if os.path.exists(datasets_dir):
        files = [f for f in os.listdir(datasets_dir) if f.lower().endswith(('.csv', '.xlsx'))]
        print(f"[INFO] Found datasets for {username}: {files}")
        return files
    return []

# --- 模型 (Model) 操作 ---
def save_model(username: str, model, model_name: str, model_type: str = 'keras'):
    """
    保存模型到指定用户的 models 目录。
    model_type: 'keras', 'word2vec', 'sklearn' 等
    """
    models_dir = get_user_data_path(username, "models")
    os.makedirs(models_dir, exist_ok=True) # 确保目录存在
    file_path = os.path.join(models_dir, f"{model_name}.{model_type}")
    
    try:
        if model_type == 'keras':
            # 假设 model 是 Keras 模型
            model.save(file_path)
        elif model_type == 'word2vec':
            # 假设 model 是 Gensim Word2Vec 模型
            model.save(file_path)
        elif model_type in ['pickle', 'sklearn']:
            # 假设 model 可以用 pickle 序列化 (适用于 sklearn 模型等)
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"[INFO] Model saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"[ERROR] Failed to save model {file_path}: {e}")
        return None

def load_model(username: str, model_name: str, model_type: str = 'keras'):
    """
    从指定用户的 models 目录加载模型。
    model_type: 'keras', 'word2vec', 'pickle', 'sklearn' 等
    """
    file_path = os.path.join(get_user_data_path(username, "models"), f"{model_name}.{model_type}")
    if os.path.exists(file_path):
        try:
            if model_type == 'keras':
                from tensorflow.keras.models import load_model as load_keras_model
                model = load_keras_model(file_path)
            elif model_type == 'word2vec':
                from gensim.models import Word2Vec
                model = Word2Vec.load(file_path)
            elif model_type in ['pickle', 'sklearn']:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            print(f"[INFO] Model loaded from {file_path}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load model {file_path}: {e}")
            return None
    else:
        print(f"[WARNING] Model file {file_path} does not exist.")
        return None

def list_user_models(username: str, model_type: str = 'keras') -> List[str]:
    """
    列出指定用户的所有指定类型的模型文件名 (不含扩展名)。
    """
    models_dir = get_user_data_path(username, "models")
    if os.path.exists(models_dir):
        ext = f".{model_type}"
        files = [f[:-len(ext)] for f in os.listdir(models_dir) if f.endswith(ext)]
        print(f"[INFO] Found models for {username}: {files}")
        return files
    return []

# --- 结果 (Results) 操作 ---
def save_result(username: str, filename: str, data: Any, format: str = 'json'):
    """
    保存结果数据 (如生成的文本、分析报告、配置等) 到指定用户的 results 目录。
    data: 通常是字典、列表或字符串。
    format: 'json', 'txt', 'pickle' 等
    """
    results_dir = get_user_data_path(username, "results")
    os.makedirs(results_dir, exist_ok=True) # 确保目录存在
    file_path = os.path.join(results_dir, filename)
    
    try:
        if format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        elif format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"[INFO] Result saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"[ERROR] Failed to save result {file_path}: {e}")
        return None

def load_result(username: str, filename: str, format: str = 'json') -> Optional[Any]:
    """
    从指定用户的 results 目录加载结果数据。
    format: 'json', 'txt', 'pickle' 等
    """
    file_path = os.path.join(get_user_data_path(username, "results"), filename)
    if os.path.exists(file_path):
        try:
            if format == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif format == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
            elif format == 'pickle':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"[INFO] Result loaded from {file_path}")
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load result {file_path}: {e}")
            return None
    else:
        print(f"[WARNING] Result file {file_path} does not exist.")
        return None

def list_user_results(username: str, format: str = 'json') -> List[str]:
    """
    列出指定用户的所有指定格式的结果文件名。
    """
    results_dir = get_user_data_path(username, "results")
    if os.path.exists(results_dir):
        ext = f".{format}"
        files = [f for f in os.listdir(results_dir) if f.endswith(ext)]
        print(f"[INFO] Found results for {username}: {files}")
        return files
    return []

# --- 其他辅助函数 ---
def get_user_config(username: str) -> Optional[Dict]:
    """
    读取用户的配置文件 (如果存在)。
    """
    config_path = os.path.join(get_user_data_path(username), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load config {config_path}: {e}")
    return None

def save_user_config(username: str, config: Dict):
    """
    保存用户的配置文件。
    """
    config_path = os.path.join(get_user_data_path(username), "config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Config saved to {config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save config {config_path}: {e}")

# --- 可选：保存可视化图 (如 Plotly) ---
def save_plotly_fig(username: str, fig: go.Figure, filename: str):
    """
    保存 Plotly 图为 HTML 文件到用户 results 目录。
    """
    results_dir = get_user_data_path(username, "results")
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, filename)
    fig.write_html(file_path)
    print(f"[INFO] Plotly figure saved to {file_path}")
    return file_path

def save_numpy_array(username: str, array: np.ndarray, filename: str):
    """
    保存 numpy 数组到用户 results 目录。
    """
    results_dir = get_user_data_path(username, "results")
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, filename)
    np.save(file_path, array)
    print(f"[INFO] Numpy array saved to {file_path}")
    return file_path

def load_numpy_array(username: str, filename: str) -> Optional[np.ndarray]:
    """
    从用户 results 目录加载 numpy 数组。
    """
    file_path = os.path.join(get_user_data_path(username, "results"), filename)
    if os.path.exists(file_path):
        try:
            array = np.load(file_path)
            print(f"[INFO] Numpy array loaded from {file_path}")
            return array
        except Exception as e:
            print(f"[ERROR] Failed to load numpy array {file_path}: {e}")
            return None
    else:
        print(f"[WARNING] Numpy array file {file_path} does not exist.")
        return None
