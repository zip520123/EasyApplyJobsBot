#!/usr/bin/env python3
"""
rag_agent_service.py

此模組提供一個 agent_service 函數，供其他程式呼叫：
傳入問題後，透過本地 LLM 產生答案，再對答案進行評分，
回傳一個 0～1 的分數，用以衡量文件內容與問題的契合度。

本範例同時保留了未來切換到 OpenAI API 的可能性，只需修改 USE_LOCAL 旗標。

需要的套件：
  - transformers (來自 Hugging Face 的 Transformers 庫，可用 pip install transformers)
  - langchain (pip install langchain)
  - faiss-cpu (pip install faiss-cpu)
  - torch (pip install torch)  ※ M1 MacBook Pro 上建議使用支援 MPS 的版本

資源清單檔案 "resources.txt" 中，每一行為檔案路徑（若存在則讀取文件內容）
或直接為純文字參考內容。
"""

import os

# -------------------------------
# LLM 設定區：決定使用本地模型或 OpenAI API
# -------------------------------
USE_LOCAL = True

if USE_LOCAL:
    # 使用 Hugging Face Transformers 建立本地 text generation pipeline
    # Transformers 庫提供的 pipeline 可以簡化模型載入與推論流程
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline
    import torch

    # 設定運算裝置，若 M1 MacBook Pro 可利用 MPS 加速（注意 MPS 支援仍屬實驗性）
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device 參數在 pipeline 中：若使用 CPU，設為 -1；否則設為裝置索引（這裡假設 index 0）
    device_idx = 0 if device != "cpu" else -1

    # local_pipeline = pipeline(
    #     "text-generation",
    #     model="distilgpt2",
    #     tokenizer="distilgpt2",
    #     max_new_tokens=128,
    #     device=device_idx,
    #     truncation=True
    # )
    # llm_instance = HuggingFacePipeline(pipeline=local_pipeline)
    model_id = os.getenv("BEDROCK_MODEL_ID", "mistral.mistral-7b-instruct-v0:2")  # 替換成你的 Bedrock 模型 ID
    region = os.getenv("AWS_REGION", "eu-west-2")  # 替換成你使用的 AWS 區域
    from langchain_aws.llms.bedrock import BedrockLLM
    # 初始化 BedrockLLM 實例，這個類別會封裝對 AWS Bedrock 的 API 調用
    llm_instance = BedrockLLM(
        model_id=model_id,
        region_name=region,
        credentials_profile_name= os.getenv("profile_name"),
        max_tokens=128,      # 最大生成 token 數量
        temperature=0.7,     # 控制生成隨機性的溫度參數
    )
    llm_instance = HuggingFacePipeline(pipeline=local_pipeline)

    from langchain.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
else:
    # 若未使用本地模型，則使用 OpenAI API 的 LLM（例如 gpt-3.5-turbo）
    from langchain.llms import OpenAI
    llm_instance = OpenAI(model_name="gpt-3.5-turbo")
    from langchain.embeddings import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

# -------------------------------
# 資源讀取與處理函式
# -------------------------------
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_document(file_path: str) -> str:
    """
    根據檔案副檔名讀取文件內容，目前支援 PDF 與文字檔。
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支援的檔案格式: {ext}")

    docs = loader.load()
    # 將所有頁面或段落內容合併成一個字串
    return "\n".join([doc.page_content for doc in docs])

def load_resource_list(resource_list_path: str) -> list:
    """
    讀取資源清單檔案，每行判斷是否為檔案路徑（存在則讀取文件內容），
    否則直接當作純文字參考，回傳所有資源的文字列表。
    """
    resources = []
    with open(resource_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if os.path.exists(line):
                try:
                    content = load_document(line)
                    resources.append(content)
                except Exception as e:
                    print(f"載入檔案 {line} 失敗: {e}")
            else:
                resources.append(line)
    return resources

# -------------------------------
# RAG 服務類別定義
# -------------------------------
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # 也可以替換成 Chroma、Pinecone 等
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGService:
    def __init__(self, resource_texts: list, chain_type: str = "stuff", llm=None):
        """
        初始化 RAG 服務：
          - 將所有參考資料切分成適合嵌入的片段。
          - 建立向量資料庫（此處使用 FAISS）。
          - 利用自訂 prompt template 建立 RetrievalQA chain。

        參數:
          resource_texts (list): 參考資料的文字列表。
          chain_type (str): Chain 類型，預設 "stuff"（也可選 "map_reduce"、"refine"）。
          llm: 用於生成回答與評分的 LLM 物件，若未提供則使用全域設定的 llm_instance。
        """
        # self.llm = llm or llm_instance
        self.llm = llm_instance

        # 切分文件成適合嵌入的片段
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []
        for text in resource_texts:
            docs.extend(text_splitter.split_text(text))

        # 建立 embeddings 與向量資料庫
        # embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_texts(docs, embeddings)

        # 定義自訂的 prompt template，統一格式化 context 與問題
        prompt_template = PromptTemplate(
            template=(
                "請根據 job description 與關於我的資料回答問題職缺和我目前能力的契合度，並給我一個綜合評分（數值越高表示表現越好）\n"
                "我的資料：\n{context}\n\n"
                "job description：\n{question}\n\n"
                "不要重複我的問題，請根據以上內容給出詳細且精確的回答："
            ),
            input_variables=["context", "question"]
        )

        # 建立 RetrievalQA chain，利用自訂模板與指定 chain_type
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template}
        )

    def answer_query(self, query: str) -> str:
        """
        透過 RetrievalQA chain 針對問題生成回答。
        """
        answer = self.qa_chain.run(query)
        return answer

    def dummy_score_response(self, query: str, answer: str) -> str:
        return self.llm(query)[len(query):]

    def score_response(self, query: str, answer: str) -> float:
        """
        使用 LLM 生成一個 0～1 之間的評分，代表回答的正確性與相關性。
        """
        prompt = (
            f"請根據下列 job description 與關於我的資料，僅回傳一個介於0到1之間的數字，"
            f"代表職缺和我目前能力的契合度（數值越高表示表現越好）。\n\n"
            f"job description: {query}\n"
            f"我的資料：{answer}\n\n"
            "請只回傳數值"
        )
        score_str = self.llm(prompt)
        try:
            score_value = float(score_str.strip())
        except Exception as e:
            print(f"解析評分失敗: {e}")
            score_value = 0.0
        return score_value

# -------------------------------
# 模組級初始化與對外接口
# -------------------------------
# 硬編碼資源清單檔案與 chain_type 設定
_RESOURCE_LIST_FILE = "resources.txt"  # 資源清單檔案，每行為檔案路徑或純文字參考
_CHAIN_TYPE = "stuff"

# 讀取資源並初始化全域 RAGService 實例
_resource_texts = load_resource_list(_RESOURCE_LIST_FILE)
_agent = RAGService(_resource_texts, chain_type=_CHAIN_TYPE, llm=llm_instance)

def agent_service(query: str):
    """
    對外接口：傳入問題文字，透過本地 LLM 產生回答並進行評分，
    回傳一個 0～1 的數值，代表文件與問題的契合度。

    參數:
      query (str): 使用者提問的問題。

    回傳:
      float: 評分數值。
    """
    answer = _agent.answer_query(query)
    score = _agent.dummy_score_response(answer, answer)
    return score

# -------------------------------
# 測試區：直接執行本模組以便檢查效果（非必需）
# -------------------------------
if __name__ == "__main__":
    sample_query = """
    Python, FastAPI, Azure
    Hybrid working in Central London
    Pays £65k-£70k + equity
    Python Developer - Python, FastAPI, Azure, LLMs
    My client are a high-growth start-up in the EdTech sector, who are leveraging AI on their platform, to deliver highly personalised experiences to their consumers.
    You'll be joining as the 3rd hire within their engineering team, working closely with the CTO and other engineers to help scale their platform, and is an ideal role for someone who has demonstrated experience scaling systems in high-growth environments.
    
    Key skills and experience:
    Strong skills with Python
    Demonstrated experience with FastAPI
    Hands-on experience with Cloud technologies (Azure preferred, but AWS/GCP will still be considered)
    Exposure to AI/ML technologies (preferred but not essential)
    This is a hybrid role in Central London, paying £65k-£70k depending on skills and experience along with an equity package.
    
    Apply now to register your interest!
    Python Developer - Python, FastAPI, Azure, LLMs
    """
    sample_score = agent_service(sample_query)
    # print(f"問題：{sample_query}")
    print(f"評分：{sample_score}")