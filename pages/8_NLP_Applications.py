# pages/8_NLP_Application.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from utils.user_auth import is_authenticated, get_current_username
from utils.llm_api import call_qwen_api # 假设你已封装好 Qwen API

# --- 为了简化示例，我们不使用 Word2Vec 和 LSTM 进行完整训练 ---
# --- 这里使用一个预训练的 TF-IDF + LR 模型作为示例 ---
# --- 你可以根据 HW02 和 HW03 的代码来构建 Word2Vec+LSTM 模型 ---
# --- 为了演示，我们创建一个非常简单的示例模型 ---
def create_simple_sentiment_model():
    """创建一个简单的示例情感分析模型 (仅用于演示)"""
    # 示例训练数据
    texts = [
        "这家酒店真不错，服务态度很好，环境优美，下次还会来。",
        "房间干净，位置方便，早餐很棒。",
        "非常满意的一次住宿体验。",
        "房间很大，设施齐全，性价比很高。",
        "前台小姐姐很热情，解决了我的问题。",
        "非常糟糕的体验，房间又小又脏。",
        "服务态度很差，让人很失望。",
        "房间设施老旧，隔音效果不好。",
        "价格太贵，性价比不高。",
        "卫生条件堪忧，不会再来了。"
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] # 1: 正面, 0: 负面

    # 简单的预处理 (实际项目中需要更复杂的预处理)
    processed_texts = [t.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ') for t in texts]
    
    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed_texts)

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X, labels)
    
    # 保存模型和向量化器 (模拟保存过程)
    # 你可以将它们保存到 utils 或 user_data 目录下
    # 为了演示，我们直接返回它们
    return vectorizer, model

# --- 模拟加载预训练模型 ---
# 在实际应用中，这里会从文件加载用户训练好的模型或预训练模型
# vectorizer, sentiment_model = load_pretrained_sentiment_model()
vectorizer, sentiment_model = create_simple_sentiment_model()

# --- Flair NER 模型加载 ---
# 注意：Flair 模型较大，首次运行会下载
# 在实际部署时，确保环境已安装 flair
try:
    from flair.models import SequenceTagger
    from flair.data import Sentence
    ner_tagger = SequenceTagger.load('ner') # 加载英文 NER 模型
    flair_available = True
except ImportError:
    st.warning("Flair 库未安装，NER 功能将不可用。请运行 'pip install flair'")
    flair_available = False
except Exception as e:
    st.error(f"加载 Flair NER 模型时出错: {e}")
    flair_available = False

# --- 页面配置 ---
st.set_page_config(
    page_title="NLP 应用任务",
    page_icon="🤖",
    layout="wide"
)

# --- 权限检查 ---
if not is_authenticated():
    st.error("⚠️ 请先登录以访问此功能。")
    st.stop()

# --- 主页面标题 ---
st.title("🤖 NLP 应用任务")

# --- 创建选项卡 ---
tab1, tab2 = st.tabs(["😊 情感分析", "🏷️ 命名实体识别"])

# ==================== Tab 1: 情感分析 ====================
with tab1:
    st.header("情感分析 (Sentiment Analysis)")
    st.write("判断输入文本的情感倾向（如正面/负面）。")

    # 用户输入
    user_text_sentiment = st.text_area("请输入待分析的文本（如评论）:", height=100, key="sentiment_input")

    # 选择模型
    model_options_sentiment = ["TF-IDF + 逻辑回归 (示例)"]
    if flair_available:
        model_options_sentiment.append("Flair (NER 模型，示例，用于演示调用)")
    model_options_sentiment.append("大模型 (Qwen)")
    selected_model_sentiment = st.selectbox("选择情感分析模型:", model_options_sentiment)

    if st.button("分析情感", key="analyze_sentiment"):
        if not user_text_sentiment.strip():
            st.error("请输入要分析的文本。")
        else:
            with st.spinner(f"使用 {selected_model_sentiment} 分析中..."):
                if selected_model_sentiment == "TF-IDF + 逻辑回归 (示例)":
                    # 预处理输入文本
                    processed_input = user_text_sentiment.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ')
                    # 向量化
                    input_vec = vectorizer.transform([processed_input])
                    # 预测
                    prediction = sentiment_model.predict(input_vec)[0]
                    probability = sentiment_model.predict_proba(input_vec)[0]
                    
                    # 显示结果
                    sentiment_label = "正面" if prediction == 1 else "负面"
                    confidence = max(probability)
                    st.success(f"预测情感: **{sentiment_label}**")
                    st.write(f"置信度: {confidence:.2f}")
                    # 简单的概率分布
                    st.write("**概率分布:**")
                    prob_df = pd.DataFrame({
                        "情感": ["负面", "正面"],
                        "概率": probability
                    })
                    st.bar_chart(prob_df.set_index("情感"))

                elif selected_model_sentiment == "Flair (NER 模型，示例，用于演示调用)":
                    # Flair 主要用于 NER，这里只是演示如何调用其他模型
                    # 对于情感分析，Flair 也有相应模型，但这里我们用它来演示
                    if flair_available:
                        sentence = Sentence(user_text_sentiment)
                        # Flair NER 通常不直接输出情感，这里仅演示调用
                        # st.info("Flair NER 模型已加载，但此示例不用于情感分析。")
                        # 你可以加载 Flair 的情感分析模型，例如 'sentiment-fast'
                        try:
                            from flair.models import TextClassifier
                            flair_sentiment_model = TextClassifier.load('sentiment')
                            flair_sentiment_model.predict(sentence)
                            # 解析结果
                            flair_result = sentence.labels[0].value
                            flair_confidence = sentence.labels[0].score
                            st.success(f"Flair 预测情感: **{flair_result}**")
                            st.write(f"置信度: {flair_confidence:.2f}")
                        except Exception as e:
                             st.error(f"使用 Flair 情感分析模型时出错: {e}")
                             st.info("Flair 情感分析模型可能需要额外安装或加载，请参考 Flair 文档。")
                    else:
                        st.error("Flair 未安装或模型加载失败。")

                elif selected_model_sentiment == "大模型 (Qwen)":
                    # 构建提示词，让大模型进行情感分析
                    prompt = f"请分析以下文本的情感倾向（正面/负面/中性），并给出简短的理由：\n\n文本: {user_text_sentiment}"
                    success, response = call_qwen_api([{"role": "user", "content": prompt}])
                    if success:
                        st.write("**大模型分析结果:**")
                        st.write(response)
                    else:
                        st.error(f"调用大模型 API 失败: {response}")


# ==================== Tab 2: 命名实体识别 (NER) ====================
with tab2:
    st.header("命名实体识别 (Named Entity Recognition, NER)")
    st.write("识别并标注文本中的人名、地名、组织名等实体。")

    # 用户输入
    user_text_ner = st.text_area("请输入待识别的文本:", height=100, key="ner_input")

    # 选择模型
    model_options_ner = []
    if flair_available:
        model_options_ner.append("Flair (BiLSTM-CRF)")
    model_options_ner.append("大模型 (Qwen)")
    selected_model_ner = st.selectbox("选择 NER 模型:", model_options_ner)

    if st.button("识别实体", key="run_ner"):
        if not user_text_ner.strip():
            st.error("请输入要识别的文本。")
        else:
            with st.spinner(f"使用 {selected_model_ner} 识别中..."):
                if selected_model_ner == "Flair (BiLSTM-CRF)":
                    if flair_available:
                        sentence = Sentence(user_text_ner)
                        ner_tagger.predict(sentence)

                        # 提取实体和标签
                        entities = [(entity.text, entity.tag, entity.score) for entity in sentence.get_spans('ner')]
                        
                        if entities:
                            st.success("识别到以下实体:")
                            # 创建 DataFrame 便于展示
                            entities_df = pd.DataFrame(entities, columns=["实体", "类型", "置信度"])
                            st.dataframe(entities_df)
                            
                            # 简单的可视化：在文本中高亮实体
                            highlighted_text = user_text_ner
                            for entity_text, entity_tag, _ in sorted(entities, key=lambda x: x[0], reverse=True): # 从后往前替换，避免索引变化
                                # 这里使用简单的 HTML 标签进行高亮，需要 st.markdown(unsafe_allow_html=True)
                                # 为了安全，也可以用其他方式，如在文本旁边标注
                                # highlighted_text = highlighted_text.replace(entity_text, f"<mark>{entity_text} ({entity_tag})</mark>")
                                pass # 暂不实现 HTML 高亮，因为有安全风险

                            # 用 Pandas 表格展示带标签的词
                            tokens_with_tags = [(token.text, token.get_tag('ner').value) for token in sentence]
                            tokens_df = pd.DataFrame(tokens_with_tags, columns=["Token", "NER Tag"])
                            # 过滤掉非实体的标签 (O)
                            entities_only_df = tokens_df[tokens_df['NER Tag'] != 'O']
                            if not entities_only_df.empty:
                                st.subheader("实体详情:")
                                st.dataframe(entities_only_df)
                            else:
                                st.info("未识别到命名实体。")

                        else:
                            st.info("未识别到命名实体。")
                    else:
                        st.error("Flair 未安装或模型加载失败。")

                elif selected_model_ner == "大模型 (Qwen)":
                    # 构建提示词，让大模型进行 NER
                    prompt = f"请识别以下文本中的命名实体（如人名 PER、地名 LOC、组织名 ORG 等），并以 JSON 格式返回结果：\n\n文本: {user_text_ner}\n\n输出格式示例: {{'entities': [{'text': '实体文本', 'label': '实体类型', 'start': 开始位置, 'end': 结束位置}]}}"
                    success, response = call_qwen_api([{"role": "user", "content": prompt}])
                    if success:
                        st.write("**大模型识别结果:**")
                        st.json(response) # 假设大模型返回了 JSON 格式
                        # 你可能需要解析 response 字符串为 JSON 对象，然后处理
                        # try:
                        #     parsed_response = json.loads(response)
                        #     # ... 解析和展示逻辑 ...
                        # except json.JSONDecodeError:
                        #     st.write(response) # 如果不是 JSON，直接显示
                    else:
                        st.error(f"调用大模型 API 失败: {response}")

# --- 信息提示 ---
st.divider()
st.info("💡 提示：此页面集成了多种 NLP 应用任务。情感分析和 NER 的模型实现可以进一步优化和扩展。")
