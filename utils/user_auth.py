# utils/user_auth.py

import streamlit as st
import hashlib
import os
import json
import time

# --- 配置 ---
USERS_FILE = "users.json" # 存储用户信息的文件
USER_DATA_DIR = "user_data" # 用户数据根目录

# --- 辅助函数 ---

def hash_password(password: str) -> str:
    """对密码进行哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """从文件加载用户信息"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"警告: {USERS_FILE} 文件损坏或不存在，将创建新文件。")
            return {}
    return {}

def save_users(users: dict):
    """将用户信息保存到文件"""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

def initialize_user_directory(username: str):
    """为新用户创建数据目录"""
    user_dir = os.path.join(USER_DATA_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    # 为每个用户创建子目录
    os.makedirs(os.path.join(user_dir, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(user_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(user_dir, "results"), exist_ok=True)
    # 创建一个用户配置文件 (可选)
    config_file = os.path.join(user_dir, "config.json")
    if not os.path.exists(config_file):
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"created_at": time.time(), "username": username}, f, ensure_ascii=False)

# --- 核心认证函数 ---

def login(username: str, password: str) -> tuple[bool, str]:
    """用户登录"""
    users = load_users()
    hashed_input_password = hash_password(password)
    if username in users and users[username]["password"] == hashed_input_password:
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        return True, f"登录成功！欢迎回来，{username}！"
    else:
        return False, "用户名或密码错误。"

def register(username: str, password: str) -> tuple[bool, str]:
    """用户注册"""
    users = load_users()
    if username in users:
        return False, "用户名已存在，请选择其他用户名。"

    if len(password) < 6:
        return False, "密码长度至少为6位。"

    # 添加新用户
    users[username] = {"password": hash_password(password)}
    save_users(users)
    # 为新用户创建数据目录
    initialize_user_directory(username)
    return True, "注册成功！请登录。"

def logout():
    """用户登出"""
    # 清除 session_state 中的用户信息
    for key in ["authenticated", "username"]:
        if key in st.session_state:
            del st.session_state[key]

def is_authenticated():
    """检查用户是否已认证"""
    return st.session_state.get('authenticated', False)

def get_current_username():
    """获取当前用户名"""
    return st.session_state.get('username', None)