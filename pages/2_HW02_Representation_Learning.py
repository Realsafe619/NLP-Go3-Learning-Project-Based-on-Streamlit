# pages/2_HW02_Representation_Learning.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import jieba
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle # 用于序列化模型或数据
import json # 用于序列化配置

# --- 权限检查 ---
from utils.user_auth import is_authenticated, get_current_username
if not is_authenticated():
    st.error("⚠️ 请先登录以访问此功能。")
    st.stop()

# --- 页面配置 ---
st.set_page_config(
    page_title="HW02: 表征学习",
    page_icon="📚",
    layout="wide"
)

# --- 页面标题 ---
st.title("📚 HW02: 表征学习 (TF-IDF & Word2Vec)")

# --- 初始化 Session State ---
# 用于存储预处理后的数据
if 'df_preprocessed_hw02' not in st.session_state:
    st.session_state.df_preprocessed_hw02 = None
# 用于存储 TF-IDF 模型和结果
if 'tfidf_vectorizer_hw02' not in st.session_state:
    st.session_state.tfidf_vectorizer_hw02 = None
if 'df_tfidf_hw02' not in st.session_state:
    st.session_state.df_tfidf_hw02 = None
# 用于存储 Word2Vec 模型和结果
if 'word2vec_model_hw02' not in st.session_state:
    st.session_state.word2vec_model_hw02 = None
if 'df_weighted_avg_hw02' not in st.session_state:
    st.session_state.df_weighted_avg_hw02 = None

# --- 用户数据路径 ---
current_user = get_current_username()
USER_DATA_DIR = "user_data"
user_models_dir = os.path.join(USER_DATA_DIR, current_user, "models")
user_results_dir = os.path.join(USER_DATA_DIR, current_user, "results")
os.makedirs(user_models_dir, exist_ok=True)
os.makedirs(user_results_dir, exist_ok=True)

# --- 功能函数定义 ---

def load_stopwords(filepath):
    """加载停用词"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    except Exception as e:
        st.error(f"加载停用词失败：{e}")
        return set()

def preprocess_text(text, stopwords, remove_punctuation=True, remove_numbers=True, remove_english=True, min_word_len=1):
    """预处理单个文本"""
    if not isinstance(text, str):
        return []
    text = text.lower()
    if remove_punctuation:
        punctuation = string.punctuation + "，。！？；：“”‘’（）【】《》、·…—"
        text = re.sub(f"[{re.escape(punctuation)}]", "", text)
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    if remove_english:
        text = re.sub(r'[a-zA-Z]+', '', text) # 去除英文单词
    words = jieba.lcut(text)
    filtered_words = [word.strip() for word in words if word.strip() not in stopwords and len(word.strip()) >= min_word_len]
    return filtered_words

# --- Tab 布局 ---
tab1, tab2, tab3 = st.tabs(["1️⃣ 数据加载与预处理", "2️⃣ TF-IDF 实验", "3️⃣ Word2Vec 实验"])

# ==================== Tab 1: 数据加载与预处理 ====================
with tab1:
    st.header("数据加载与预处理")
    st.write("上传CSV文件或使用默认数据集，并进行文本预处理。")

    # 上传文件 or 使用默认
    uploaded_file = st.file_uploader("上传 CSV 文件（要求包含 'review' 列）", type=["csv"], key="hw02_upload")
    use_default = st.checkbox("使用默认数据集", value=True, key="hw02_default")

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        st.success("上传成功！")
    elif use_default:
        default_dataset_path = r"C:\Users\Railg\Desktop\nlp_go3_project\dataset.csv" # 你的默认数据集路径
        try:
            df = pd.read_csv(default_dataset_path, encoding='utf-8')
            st.info(f"使用默认数据集: {default_dataset_path}")
        except FileNotFoundError:
            st.error(f"无法找到默认数据集文件: {default_dataset_path}")
            df = None # 确保 df 为 None，以便后续不执行预处理
        except Exception as e:
            st.error(f"加载默认数据集时出错: {e}")
            df = None
    else:
        st.warning("请上传文件或勾选使用默认数据集。")

    if df is not None:
        st.subheader("原始数据预览")
        st.dataframe(df.head())

        # 预处理参数设置 (更符合设计思路)
        st.subheader("预处理参数设置")
        stopwords_path = st.text_input("停用词文件路径 (如为空则不使用)", value=r"C:\Users\Railg\Desktop\nlp_go3_project\stopwords.txt", key="hw02_stopwords_path") # 设置默认路径        
        remove_punctuation = st.toggle("去掉标点", value=True, key="hw02_punct")
        remove_numbers = st.toggle("去除数字", value=True, key="hw02_nums")
        remove_english = st.toggle("去除英文", value=True, key="hw02_eng")
        min_word_len = st.slider("最小词长度", min_value=1, max_value=5, value=1, key="hw02_min_len")

        # 加载停用词
        stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            stopwords = load_stopwords(stopwords_path)
        elif stopwords_path: # 如果路径不为空但文件不存在
            st.error(f"停用词文件不存在: {stopwords_path}")

        # 执行预处理
        if st.button("执行预处理", key="hw02_preprocess_btn"):
            # 获取当前组件的值，确保它们是最新的
            processed_stopwords_path = st.session_state["hw02_stopwords_path"]
            processed_remove_punctuation = st.session_state["hw02_punct"]
            processed_remove_numbers = st.session_state["hw02_nums"]
            processed_remove_english = st.session_state["hw02_eng"]
            processed_min_word_len = st.session_state["hw02_min_len"]
            
            # 重新加载停用词（以防路径改变）
            processed_stopwords = set()
            if processed_stopwords_path and os.path.exists(processed_stopwords_path):
                processed_stopwords = load_stopwords(processed_stopwords_path)
            
            with st.spinner("正在预处理数据..."):
                df['review_wd'] = df['review'].apply(
                    lambda x: preprocess_text(
                        x, 
                        processed_stopwords, 
                        processed_remove_punctuation, 
                        processed_remove_numbers, 
                        processed_remove_english, 
                        processed_min_word_len
                    )
                )
            st.success("预处理完成！")
            st.subheader("预处理后数据预览")
            # 显示原句 vs 预处理结果 (两列对照)
            preview_df = df[['review', 'review_wd']].head(10).copy()
            preview_df['review_wd_str'] = preview_df['review_wd'].apply(lambda x: ', '.join(x))
            st.dataframe(preview_df[['review', 'review_wd_str']].rename(columns={'review_wd_str': '预处理结果'}))

            # 统计信息
            st.subheader("预处理统计信息")
            df['word_count'] = df['review_wd'].apply(len)
            avg_len = df['word_count'].mean()
            max_len = df['word_count'].max()
            min_len = df['word_count'].min()
            st.write(f"- 平均词数: {avg_len:.2f}")
            st.write(f"- 最长评论词数: {max_len}")
            st.write(f"- 最短评论词数: {min_len}")

            # 保存到 session_state
            st.session_state.df_preprocessed_hw02 = df.copy()
            st.session_state.hw02_preprocess_params = {
                "stopwords_path": processed_stopwords_path,
                "remove_punctuation": processed_remove_punctuation,
                "remove_numbers": processed_remove_numbers,
                "remove_english": processed_remove_english,
                "min_word_len": processed_min_word_len
            }
            st.success("预处理数据已保存至会话状态。")


# ==================== Tab 2: TF-IDF 实验 ====================
with tab2:
    st.header("TF-IDF 实验")
    st.write("计算TF-IDF权重，提取关键词，并生成词云。")

    # 检查是否有预处理数据
    if st.session_state.df_preprocessed_hw02 is None:
        st.warning("请先在 '数据加载与预处理' 页面完成数据预处理。")
    else:
        df = st.session_state.df_preprocessed_hw02

        # 参数设置 (更符合设计思路)
        st.subheader("TF-IDF 关键词提取模块")
        top_k = st.slider("topK 关键词数量", min_value=5, max_value=50, value=20, key="hw02_topk")
        ngram_range = st.selectbox("ngram 范围", options=[(1,1), (1,2), (1,3)], index=1, key="hw02_ngram")
        min_df = st.slider("最小文档频率 (min_df)", min_value=1, max_value=10, value=1, key="hw02_min_df")
        max_features = st.slider("最大特征数 (max_features)", min_value=100, max_value=5000, value=1000, key="hw02_max_feat")

        # 准备语料
        df['review_for_tfidf'] = df['review_wd'].apply(lambda x: ' '.join(x))

        # 执行 TF-IDF
        if st.button("执行 TF-IDF 关键词提取", key="hw02_tfidf_btn"):
            with st.spinner("正在计算 TF-IDF..."):
                vectorizer = TfidfVectorizer(
                    stop_words=None, # 停用词已在预处理中处理
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features
                )
                tfidf_matrix = vectorizer.fit_transform(df['review_for_tfidf'].tolist())
                feature_names = vectorizer.get_feature_names_out()

                # 提取关键词
                top_keywords_list = []
                for i in range(tfidf_matrix.shape[0]):
                    tfidf_scores = tfidf_matrix[i].toarray().flatten()
                    sorted_indices = tfidf_scores.argsort()[::-1][:top_k]
                    top_keywords = [feature_names[idx] for idx in sorted_indices if tfidf_scores[idx] > 0]
                    top_keywords_list.append(top_keywords)

                df['tfidf_keywords'] = top_keywords_list

            st.success("TF-IDF 关键词提取完成！")
            st.session_state.df_tfidf_hw02 = df.copy()
            st.session_state.tfidf_vectorizer_hw02 = vectorizer

            # 显示结果 (DataFrame)
            st.subheader("每条评论的 Top-K 关键词")
            display_df = df[['review', 'tfidf_keywords']].copy()
            display_df['tfidf_keywords_str'] = display_df['tfidf_keywords'].apply(lambda x: ', '.join(x))
            st.dataframe(display_df[['review', 'tfidf_keywords_str']].head(10).rename(columns={'tfidf_keywords_str': 'TF-IDF 关键词'}))

            # 单条评论关键词查询
            st.subheader("单条评论关键词查询")
            def safe_string_slice(s, length=50):
                if isinstance(s, str):
                    return s[:length]
                else:
                    return ""
                    
            selected_index = st.selectbox("选择评论", options=range(len(df)), format_func=lambda x: f"评论 {x+1}: {safe_string_slice(df.iloc[x]['review'])}...")
            show_keywords = st.button("查询关键词", key="hw02_query_single")
            
            # 保存查询结果状态
            if 'show_keyword_result' not in st.session_state:
                st.session_state.show_keyword_result = False
            if 'last_selected_index' not in st.session_state:
                st.session_state.last_selected_index = None
                
            if show_keywords:
                st.session_state.show_keyword_result = True
                st.session_state.last_selected_index = selected_index
                
            if st.session_state.show_keyword_result and st.session_state.last_selected_index is not None:
                current_index = st.session_state.last_selected_index
                selected_keywords = df.iloc[current_index]['tfidf_keywords']
                st.write(f"评论 {current_index+1} 的关键词: {selected_keywords}")
                # 关键词柱状图
                if selected_keywords:
                    keyword_scores = [vectorizer.transform([df.iloc[current_index]['review_for_tfidf']]).toarray()[0][vectorizer.vocabulary_[kw]] for kw in selected_keywords if kw in vectorizer.vocabulary_]
                    fig_bar = px.bar(x=selected_keywords, y=keyword_scores, labels={'x': '关键词', 'y': 'TF-IDF 权重'}, title=f"评论 {current_index+1} 的关键词权重")
                    st.plotly_chart(fig_bar, use_container_width=True)


        # TF-IDF 词云模块
        if st.session_state.df_tfidf_hw02 is not None and st.session_state.tfidf_vectorizer_hw02 is not None:
            st.subheader("TF-IDF 词云模块")
            # 参数设置
            bg_color = st.color_picker("背景颜色", value="#ffffff", key="hw02_wc_bg")
            max_words = st.slider("最大词数", min_value=50, max_value=500, value=200, key="hw02_wc_max")
            # mask_image = st.file_uploader("上传 Mask 图片 (可选)", type=["png", "jpg", "jpeg"], key="hw02_wc_mask")

            if st.button("生成 TF-IDF 词云", key="hw02_wc_btn"):
                df_tfidf = st.session_state.df_tfidf_hw02
                vectorizer = st.session_state.tfidf_vectorizer_hw02
                # 使用所有文档的 TF-IDF 矩阵，获取平均权重或总权重来生成词云
                tfidf_matrix_full = vectorizer.transform(df_tfidf['review_for_tfidf'].tolist())
                # 计算每个词的平均TF-IDF分数作为权重
                mean_scores = np.array(tfidf_matrix_full.mean(axis=0)).flatten()
                feature_names = vectorizer.get_feature_names_out()
                word_freq_dict = dict(zip(feature_names, mean_scores))

                if word_freq_dict:
                    # 确保字体路径正确，这里使用 matplotlib 默认字体或 simhei.ttf
                    # 注意：在 Streamlit Cloud 等环境中，字体路径可能需要特殊处理
                    try:
                        # 尝试使用 matplotlib 字体
                        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS'] # 调整字体优先级
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        # 检查系统中可用的字体
                        import matplotlib.font_manager as fm
                        available_fonts = [f.name for f in fm.fontManager.ttflist]
                        
                        # 查找可用的中文字体
                        chinese_font = None
                        preferred_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'SimSun']
                        for font_name in preferred_fonts:
                            if font_name in available_fonts:
                                chinese_font = font_name
                                break
                        
                        # 构建WordCloud参数
                        wc_params = {
                            "background_color": bg_color,
                            "width": 800,
                            "height": 400,
                            "max_words": max_words,
                            "relative_scaling": 0.5,
                            "colormap": 'viridis',
                            "font_path": None  # 默认不指定字体路径
                        }
                        
                        # 如果找到中文字体，则添加font_path参数
                        if chinese_font:
                            # 获取字体路径
                            font_paths = [f.fname for f in fm.fontManager.ttflist if f.name == chinese_font]
                            if font_paths:
                                wc_params["font_path"] = font_paths[0]
                        
                        wordcloud = WordCloud(**wc_params).generate_from_frequencies(word_freq_dict)

                        fig_wc, ax = plt.subplots(figsize=(15, 7))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title('TF-IDF 权重词云', fontsize=16)
                        st.pyplot(fig_wc)
                    except Exception as e:
                        st.error(f"生成词云时出错 (可能与字体有关): {e}")
                        # 尝试不指定字体生成
                        try:
                            # 使用最基本的配置尝试生成词云
                            wordcloud = WordCloud(
                                background_color=bg_color,
                                width=800,
                                height=400,
                                max_words=max_words,
                                relative_scaling=0.5,
                                colormap='viridis',
                                # 强制不使用特定字体路径
                            ).generate_from_frequencies(word_freq_dict)

                            fig_wc2, ax2 = plt.subplots(figsize=(15, 7))
                            ax2.imshow(wordcloud, interpolation='bilinear')
                            ax2.axis('off')
                            ax2.set_title('TF-IDF 权重词云 (基础版本)', fontsize=16)
                            st.pyplot(fig_wc2)
                        except Exception as e2:
                            st.error(f"尝试备用方法生成词云也失败: {e2}")

                else:
                    st.warning("没有有效的TF-IDF权重用于生成词云。")


# ==================== Tab 3: Word2Vec 实验 ====================
with tab3:
    st.header("Word2Vec 实验")
    st.write("训练Word2Vec模型，查询相似词，并进行词向量可视化。")

    # 检查是否有预处理数据
    if st.session_state.df_preprocessed_hw02 is None:
        st.warning("请先在 '数据加载与预处理' 页面完成数据预处理。")
    else:
        df = st.session_state.df_preprocessed_hw02

        # 参数设置 (更符合设计思路)
        st.subheader("Word2Vec 模型训练")
        vector_size = st.slider("向量维度", min_value=50, max_value=300, value=100, key="hw02_w2v_vs")
        window = st.slider("窗口大小", min_value=2, max_value=10, value=5, key="hw02_w2v_win")
        min_count = st.slider("最小词频", min_value=1, max_value=10, value=1, key="hw02_w2v_mc")
        sg = st.radio("模型类型", ["CBOW (sg=0)", "Skip-Gram (sg=1)"], index=1, key="hw02_w2v_sg")
        sg_val = 1 if sg == "Skip-Gram (sg=1)" else 0

        # 训练模型 or 加载模型
        model = st.session_state.word2vec_model_hw02
        train_model = st.checkbox("重新训练模型（耗时较长）", value=True, key="hw02_w2v_train")

        if train_model:
            if st.button("开始训练 Word2Vec 模型", key="hw02_w2v_train_btn"):
                with st.spinner("正在训练 Word2Vec 模型..."):
                    sentences = df['review_wd'].tolist()
                    model = Word2Vec(
                        sentences=sentences,
                        vector_size=vector_size,
                        window=window,
                        min_count=min_count,
                        workers=4,
                        sg=sg_val
                    )
                st.success("模型训练完成！")
                st.session_state.word2vec_model_hw02 = model
                # 可选：保存模型到用户目录
                model_save_path = os.path.join(user_models_dir, f"hw02_word2vec_user_{current_user}.model")
                model.save(model_save_path)
                st.info(f"模型已保存至: {model_save_path}")
        else:
            # 加载模型
            model_path_input = st.text_input("从文件加载模型路径", value=os.path.join(r"C:\Users\Railg\Desktop\nlp_go3_project", f"hw02_word2vec_user_{current_user}.model"), key="hw02_w2v_load_path") # 修改默认路径
            if st.button("加载模型", key="hw02_w2v_load_btn"):
                try:
                    model = Word2Vec.load(model_path_input)
                    st.success("模型加载成功！")
                    st.session_state.word2vec_model_hw02 = model
                except Exception as e:
                    st.error(f"加载模型失败：{e}")


        # 模型功能展示
        if st.session_state.word2vec_model_hw02 is not None:
            model = st.session_state.word2vec_model_hw02
            st.subheader("模型信息与功能")
            st.write(f"- 词汇表大小：{len(model.wv.key_to_index)}")
            st.write(f"- 向量维度：{model.vector_size}")

            # 相似词查询
            st.subheader("相似词查询")
            query_word = st.text_input("输入查询词", value="酒店", key="hw02_w2v_query")
            top_n_similar = st.slider("显示最相似词数量", min_value=1, max_value=20, value=10, key="hw02_w2v_topn")

            if st.button("查询相似词", key="hw02_w2v_query_btn"):
                try:
                    similar_words = model.wv.most_similar(query_word, topn=top_n_similar)
                    st.write(f"'{query_word}' 的最相似词:")
                    for word, score in similar_words:
                        st.write(f"- {word}: {score:.4f}")
                except KeyError:
                    st.warning(f"词 '{query_word}' 不在词汇表中。")


            # PCA 降维 + 向量可视化
            st.subheader("词向量可视化 (PCA)")
            st.write("使用PCA将词向量降维至2D进行可视化。")

            custom_words = st.text_area("输入要可视化的词（用逗号或空格分隔）", "干净,整洁,舒适,温馨,安静,简陋,破旧,年代,恶劣,笑容,细心", key="hw02_w2v_pca_words")
            word_list = [w.strip() for w in re.split(r'[,\s]+', custom_words) if w.strip()]

            if st.button("生成 PCA 图", key="hw02_w2v_pca_btn"):
                vectors = []
                valid_words = []
                for word in word_list:
                    try:
                        vectors.append(model.wv[word])
                        valid_words.append(word)
                    except KeyError:
                        st.warning(f"词 '{word}' 不在词汇表中，已跳过。")
                        continue

                if len(vectors) == 0:
                    st.warning("没有有效的词向量可供绘制。")
                else:
                    vectors = np.array(vectors)
                    pca = PCA(n_components=2)
                    reduced_vectors = pca.fit_transform(vectors)

                    # 使用 Plotly 创建交互式图表
                    fig_pca = go.Figure()
                    fig_pca.add_trace(go.Scatter(
                        x=reduced_vectors[:, 0],
                        y=reduced_vectors[:, 1],
                        mode='markers+text',
                        text=valid_words,
                        textposition="top center",
                        marker=dict(size=8, opacity=0.7),
                        name="Words"
                    ))
                    fig_pca.update_layout(
                        title='词向量分布图（PCA 2D）',
                        xaxis_title='PCA Component 1',
                        yaxis_title='PCA Component 2',
                        width=800,
                        height=600,
                        hovermode='closest'
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)


            # 计算加权平均向量 (如果TF-IDF结果存在)
            if st.session_state.df_tfidf_hw02 is not None:
                df_tfidf = st.session_state.df_tfidf_hw02
                st.subheader("评论向量可视化 (基于TF-IDF加权平均)")

                # 为每行计算加权平均向量
                if st.button("计算TF-IDF加权平均向量", key="hw02_w2v_weighted_btn"):
                    vectorizer_full = st.session_state.tfidf_vectorizer_hw02 # 使用之前训练好的TF-IDF模型
                    if vectorizer_full is None:
                        st.error("无法计算加权向量，TF-IDF模型未找到。请先运行TF-IDF实验。")
                    else:
                        with st.spinner("计算加权平均向量..."):
                            # 重新计算TF-IDF矩阵（用于获取精确权重）
                            tfidf_matrix_full = vectorizer_full.fit_transform(df_tfidf['review_for_tfidf'].tolist())
                            feature_names = vectorizer_full.get_feature_names_out()

                            df_tfidf['tfidf_word_weights'] = [
                                dict(zip(feature_names, tfidf_matrix_full[i].toarray().flatten()))
                                for i in range(tfidf_matrix_full.shape[0])
                            ]

                            def get_weighted_average_vector(tfidf_keywords, tfidf_scores_dict, model):
                                vectors = []
                                weights = []
                                for word in tfidf_keywords:
                                    if word in model.wv.key_to_index and word in tfidf_scores_dict:
                                        vectors.append(model.wv[word])
                                        weights.append(tfidf_scores_dict[word])
                                if len(vectors) == 0:
                                    return np.zeros(model.vector_size)
                                vectors = np.array(vectors)
                                weights = np.array(weights)
                                if weights.sum() > 0:
                                    weights = weights / weights.sum()
                                return np.average(vectors, axis=0, weights=weights)

                            df_tfidf['weighted_avg_vec'] = df_tfidf.apply(
                                lambda row: get_weighted_average_vector(row['tfidf_keywords'], row['tfidf_word_weights'], model),
                                axis=1
                            )

                        st.success("加权平均向量计算完成！")
                        st.session_state.df_weighted_avg_hw02 = df_tfidf.copy() # 保存到 session_state


                # PCA 可视化加权平均向量 (按类别染色)
                if st.session_state.df_weighted_avg_hw02 is not None:
                    df_weighted = st.session_state.df_weighted_avg_hw02
                    if 'label' in df_weighted.columns:
                        st.subheader("评论向量 PCA 可视化（按类别染色）")
                        n_each = st.slider("每类样本数量", min_value=5, max_value=100, value=20, key="hw02_w2v_samples")

                        if st.button("生成 PCA 图（按类别）", key="hw02_w2v_pca_label_btn"):
                            df_0 = df_weighted[df_weighted['label'] == 0].sample(n=min(n_each, len(df_weighted[df_weighted['label'] == 0])), random_state=42)
                            df_1 = df_weighted[df_weighted['label'] == 1].sample(n=min(n_each, len(df_weighted[df_weighted['label'] == 1])), random_state=42)
                            df_samples = pd.concat([df_0, df_1]).reset_index(drop=True)

                            vector_matrix_weighted = np.vstack(df_samples['weighted_avg_vec'].values)
                            pca_weighted = PCA(n_components=2)
                            reduced_vectors_weighted = pca_weighted.fit_transform(vector_matrix_weighted)

                            fig_pca_label = go.Figure()
                            fig_pca_label.add_trace(go.Scatter(
                                x=reduced_vectors_weighted[:len(df_0), 0],
                                y=reduced_vectors_weighted[:len(df_0), 1],
                                mode='markers',
                                marker=dict(color='lightblue', size=8, line=dict(color='darkblue', width=1)),
                                name='Label 0 (差评)',
                                opacity=0.7
                            ))
                            fig_pca_label.add_trace(go.Scatter(
                                x=reduced_vectors_weighted[len(df_0):, 0],
                                y=reduced_vectors_weighted[len(df_0):, 1],
                                mode='markers',
                                marker=dict(color='lightcoral', size=8, line=dict(color='darkred', width=1)),
                                name='Label 1 (好评)',
                                opacity=0.7
                            ))
                            fig_pca_label.update_layout(
                                title='PCA 可视化（评论TF-IDF加权平均向量，按标签染色）',
                                xaxis_title='PCA Component 1',
                                yaxis_title='PCA Component 2',
                                width=800,
                                height=600,
                                hovermode='closest'
                            )
                            st.plotly_chart(fig_pca_label, use_container_width=True)

                    else:
                        st.warning("数据中没有 'label' 列，无法按标签可视化评论向量。")


# --- 信息提示 ---
st.divider()
st.info("💡 提示：此页面整合了HW02的全部实验内容。数据和模型已与用户中心关联，训练结果会自动保存。")