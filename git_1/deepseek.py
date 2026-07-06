import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# ==================== 加载环境变量 ====================
# 优先从 .env 文件加载（本地开发），如果文件不存在则跳过
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# 再尝试从 Streamlit Secrets 读取（部署到 Streamlit Cloud 时）
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
    DEEPSEEK_BASE_URL = st.secrets["DEEPSEEK_BASE_URL"]
except (FileNotFoundError, KeyError):
    # 本地开发：从系统环境变量读取
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1")

if not DEEPSEEK_API_KEY:
    st.error("❌ 未设置 API Key。请创建 .env 文件或配置 Streamlit Secrets。")
    st.stop()

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# ==================== 嵌入模型（本地加载） ====================
@st.cache_resource
def load_embedding():
    model_path = "./local_embedding"
    if os.path.exists(model_path):
        # 本地有模型，直接加载
        model_name = model_path
    else:
        # 云端没有本地模型，从 HuggingFace 下载（使用国内镜像加速）
        model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
embedding = load_embedding()

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="📄 RAG 文档问答", layout="wide")
st.title("📄 智能文档问答（PDF + RAG + 硅基流动）")
st.markdown("上传一个 PDF 文件，AI 将基于文档内容回答你的问题。")

uploaded_file = st.file_uploader("📤 上传 PDF 文件", type="pdf")

if uploaded_file:
    # 1. 读取 PDF 文字
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    os.unlink(pdf_path)

    if not text.strip():
        st.error("未提取到文字，请确认 PDF 是文本格式（非扫描图片）。")
    else:
        st.success(f"提取了 {len(text)} 个字符")

        # 2. 文字分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        st.info(f"分成 {len(chunks)} 个文本块")

        # 3. 向量化存储
        persist_dir = tempfile.mkdtemp()
        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embedding,
            persist_directory=persist_dir
        )

        # 4. 问答
        st.subheader("❓ 向文档提问")
        question = st.text_input("输入你的问题：", placeholder="例如：本文的主要结论是什么？")

        if question:
            with st.spinner("检索相关段落并调用大模型生成答案..."):
                retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(question)
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""你是一个专业的文档分析助手。请根据以下文档内容回答问题，如果文档中没有答案，请说“文档中未提及”。

文档内容：
{context}

问题：{question}
答案："""

                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",  # 硅基流动免费模型
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                answer = response.choices[0].message.content

            st.success(f"**答案：** {answer}")

            with st.expander("📌 参考的文本片段"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**片段 {i + 1}:** {doc.page_content[:300]}...")