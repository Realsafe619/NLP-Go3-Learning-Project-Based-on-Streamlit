# NLP Go3 Project

这是一个基于 Streamlit 的自然语言处理(NLP)项目，包含了多个实验和应用模块，旨在帮助团队学习和实践NLP技术。

## 项目结构
streamlit_app.py/
│
├── .streamlit/
│   └── secrets.toml          # 存放API密钥、数据库密码等敏感信息 (重要!)
│
├── pages/
│   ├── 1_HW01_Gradient_WordCloud.py    # HW01 页面
│   ├── 2_HW02_Representation_Learning.py # HW02 页面
│   ├── 3_HW03_LSTM_Text_Generation.py  # HW03 页面
│   ├── 4_HW04_Some_Task.py           # HW04 页面 (根据实际内容命名)
│   ├── 5_HW05_Some_Task.py           # HW05 页面 (根据实际内容命名)
│   ├── 6_LLM_QA.py                   # 大模型问答功能页面
│   ├── 7_User_Center.py              # 用户中心页面
│   └── 8_NLP_Applications.py         # NLP 应用任务页面 (情感分析、NER)
│
├── utils/
│   ├── user_auth.py                  # 用户认证逻辑 (核心文件)
│   ├── data_manager.py               # 用户数据管理逻辑 (如加载/保存数据集、模型)
│   └── llm_api.py                    # 大模型 API 调用封装
│
├── user_data/                        # 用户数据存储目录 (每个用户一个子目录)
│   ├── user1/
│   │   ├── datasets/                 # 用户1上传的数据集
│   │   ├── models/                   # 用户1训练的模型 (.h5, .pkl等)
│   │   └── results/                  # 用户1生成的结果
│   ├── user2/
│   │   ├── datasets/
│   │   ├── models/
│   │   └── results/
│   └── ...                           # 其他用户
│
├── assets/                           # 静态资源 (如字体、默认图片)
│   └── simhei.ttf
│   └── ...
│
├── requirements.txt                  # 项目依赖
└── streamlit_app.py                  # 主应用入口 (负责用户认证检查和页面导航)
└── test_qwen.py                      # 测试通义千问大模型接口
└── dataset.csv                       # Hw02 默认数据集
└── stopwords.txt                     # Hw02 停用词
└── user.json                         # 用户配置文件



## 核心模块介绍

### 1. 用户认证系统 (`utils/user_auth.py`)

负责用户注册、登录、登出等功能，确保只有授权用户才能访问特定功能。

主要功能：
- 用户注册与密码加密存储
- 用户登录验证
- 用户登出
- 用户状态检查
- 为每位用户创建独立的数据目录

### 2. 数据管理工具 (`utils/data_manager.py`)

提供统一的数据管理接口，支持用户数据的保存、加载和管理。

主要功能：
- 管理用户数据集(CSV、Excel等)
- 保存和加载机器学习模型(Keras、Word2Vec等)
- 管理实验结果(JSON、TXT等)
- 用户配置文件管理
- 可视化图表保存功能

### 3. 大语言模型接口 (`utils/llm_api.py`)

封装了与通义千问大模型的交互接口。

主要功能：
- 调用通义千问API
- 支持流式和非流式响应
- 错误处理机制

## 页面功能详解

### 1. HW01: 梯度下降与词云图 (`pages/1_HW01_Gradient_WordCloud.py`)

包含两个主要部分：

1. **梯度下降可视化**
   - 支持多种目标函数（二次函数、Rosenbrock函数、Himmelblau函数）
   - 可视化二维和三维梯度下降过程
   - 动画演示梯度下降路径
   - 可调节学习率、初始点等参数

2. **交互式词云图生成**
   - 支持文本文件上传或直接输入文本
   - 自定义背景颜色和遮罩图片
   - 支持停用词过滤
   - 词云图下载功能

### 2. HW02: 表征学习 (`pages/2_HW02_Representation_Learning.py`)

实现了两种主流文本表征方法：

1. **TF-IDF 实验**
   - 文本预处理（分词、去停用词等）
   - TF-IDF关键词提取
   - 关键词权重可视化
   - TF-IDF词云生成

2. **Word2Vec 实验**
   - Word2Vec模型训练（支持CBOW和Skip-Gram）
   - 相似词查询功能
   - 词向量PCA可视化
   - 基于TF-IDF加权的文档向量表示
   - 文档向量分类可视化

### 3. HW03: LSTM文本生成 (`pages/3_HW03_LSTM_Text_Generation.py`)

基于深度学习的文本生成实验：

- 文本清洗和预处理
- 基于jieba的中文分词
- LSTM神经网络模型构建
- 文本序列化和向量化
- 基于种子文本的自动文本生成

### 4. HW04: ... (`pages/4_HW04_Some_Task.py`)
待完善

### 5. HW05: ... (`pages/5_HW05_Some_Task.py`)
待完善

### 6. 大模型问答系统 (`pages/6_LLM_QA.py`)

集成通义千问的大模型问答界面：

- 上下文感知的对话系统
- 实验相关上下文设置
- 对话历史记录维护
- 流式响应显示

### 7. 用户中心 (`pages/7_User_Center.py`)

用户个人数据管理面板：

- 用户信息展示
- 数据集、模型、结果文件浏览
- 个人空间使用情况查看

### 8. NLP应用任务 (`pages/8_NLP_Applications.py`)

集成常见NLP任务的应用界面：

1. **情感分析**
   - 基于TF-IDF+LR的传统方法
   - 基于Flair的深度学习方法
   - 基于大模型的智能分析

2. **命名实体识别(NER)**
   - 基于Flair的NER模型
   - 基于大模型的智能识别

## 使用指南
1.切换至项目目录下，激活安装streamlit的对应环境
2.运行项目：streamlit run streamlit_app.py
3.默认数据集及停用词路径需要更改为项目目录下的dataset.csv和stopwords.txt
4.用户数据存储目录为项目目录下的user_data文件夹
5.用户配置文件路径为项目目录下的user.json
6.首次使用需要注册用户，注册后会自动创建个人数据目录。登录后才能访问所有功能模块。

### 环境配置

1. 安装依赖包：
```bash
pip install streamlit numpy matplotlib wordcloud jieba gensim scikit-learn tensorflow plotly pandas dashscope flair
```

2.配置API密钥：
在项目根目录创建secre.toml文件，并添加以下内容：
[secrets]
DASHSCOPE_API_KEY = "your_dashscope_api_key_here"

3.运行项目
```bash
streamlit run streamlit_app.py
```

## 团队协作建议
1. 数据隔离：每位团队成员应使用独立账户，确保实验数据互不干扰

2. 模型共享：训练好的模型保存在个人目录中，可通过文件共享方式交换

3. 实验记录：利用结果保存功能记录重要实验参数和结果

4. 版本控制：建议使用Git管理代码变更

5. 功能开发：新增功能建议创建新的页面文件，遵循现有代码风格

## 注意事项

1.确保中文字体正确配置以支持中文显示

2.大模型功能需要有效API密钥才能正常使用

3.Word2Vec和LSTM训练可能消耗较多计算资源和时间

4.Flair模型首次使用会自动下载，需要稳定的网络连接
