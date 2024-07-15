import os
import bs4
from dotenv import load_dotenv

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
#os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
#os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

#OPENAI_API_KEY가 기본값이라, openai_api_key는 필수가 아님. OpenAIEmbeddings()에서 openai_api_key를 명시할때 필요함.
#openai_api_key = os.getenv("OPENAI_API_KEY")



# HuggingFaceBgeEmbeddings를 사용하기 전에 토크나이저 실행이 끝나 있어야 함.
# 함수나 클래스로 정돈하면 정상 동작할거 같은데... 현재는 테스트 버젼에서는 여전히 경고가 발생함.
import multiprocessing
from transformers import AutoTokenizer

def initialize_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # 여기서 토크나이저 사용

if __name__ == "__main__":
    process = multiprocessing.Process(target=initialize_tokenizer)
    process.start()
    process.join()



#### INDEXING ####

### LOADER ###


# from langchain.document_loaders import PyPDFLoader

# # PDF 파일 로드. 파일의 경로 입력
# loader = PyPDFLoader("data/SPRI_AI_Brief_2023_12.pdf")


# from langchain_community.document_loaders import DirectoryLoader

# loader = DirectoryLoader(".", glob="data/*.pdf")

# # 페이지 별 문서 로드
# docs = loader.load()
# print(f"문서의 수: {len(docs)}")

# # 10번째 페이지의 내용 출력
# print(f"\n[페이지내용]\n{docs[10].page_content[:500]}")
# print(f"\n[metadata]\n{docs[10].metadata}\n")



# from langchain_community.document_loaders.csv_loader import CSVLoader

# # CSV 파일 로드
# loader = CSVLoader(file_path="data/titanic.csv")
# docs = loader.load()
# print(f"문서의 수: {len(docs)}")

# # 10번째 페이지의 내용 출력
# print(f"\n[페이지내용]\n{docs[10].page_content[:500]}")
# print(f"\n[metadata]\n{docs[10].metadata}\n")


# from langchain_community.document_loaders import TextLoader
# loader = TextLoader("data/appendix-keywords.txt")

# from langchain_community.document_loaders import DirectoryLoader
# loader = DirectoryLoader(".", glob="data/*.txt", show_progress=True)

# docs = loader.load()
# print(f"문서의 수: {len(docs)}")

# # 10번째 페이지의 내용 출력
# print(f"\n[페이지내용]\n{docs[10].page_content[:500]}")
# print(f"\n[metadata]\n{docs[10].metadata}\n")




# from langchain_community.document_loaders import PythonLoader

# loader = DirectoryLoader(".", glob="**/*.py", loader_cls=PythonLoader)
# docs = loader.load()

# print(f"문서의 수: {len(docs)}\n")
# print("[메타데이터]\n")
# print(docs[0].metadata)
# print("\n========= [앞부분] 미리보기 =========\n")
# print(docs[0].page_content[:500])



# Load Documents
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()


url = "https://n.news.naver.com/article/437/0000378416"
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)
docs = loader.load()


# Documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."



### Split ###


# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings

# # SemanticChunker 를 생성합니다.
# semantic_text_splitter = SemanticChunker(
#     OpenAIEmbeddings(), add_start_index=True)

# # chain of density 논문의 일부 내용을 불러옵니다
# with open("data/chain-of-density.txt", "r") as f:
#     text = f.read()

# for sent in semantic_text_splitter.split_text(text):
#     print(sent)
#     print("===" * 20)


# RecursivecharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)



### Vector Store ###



# # fastembed 적용
# # pip3 install fastembed -q
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# vectorstore = FAISS.from_documents(
#     documents=splits, embedding=FastEmbedEmbeddings())




# # FAISS DB 적용
# from langchain_community.vectorstores import FAISS

# vectorstore = FAISS.from_documents(
#     documents=splits, embedding=OpenAIEmbeddings())


# Chroma DB 적용
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=splits, 
                                    # embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
                                    embedding=OpenAIEmbeddings())



# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# # 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
# # 벡터스토어를 생성합니다.
# vectorstore = FAISS.from_documents(
#     documents=splits, embedding=HuggingFaceBgeEmbeddings()
# )



retriever = vectorstore.as_retriever()








query = "회사의 저출생 정책이 뭐야?"

retriever = vectorstore.as_retriever(search_type="similarity")
search_result = retriever.get_relevant_documents(query)
print(search_result)








#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")


