import os
import requests

from dotenv import load_dotenv
from bs4 import BeautifulSoup


from langchain import hub
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA,create_retrieval_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import create_retrieval_chain



load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
#os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
#os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

#OPENAI_API_KEY가 기본값이라, openai_api_key는 필수가 아님. OpenAIEmbeddings()에서 openai_api_key를 명시할때 필요함.
#openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ['USER_AGENT'] = "Mozilla/5.0 (compatible; MyAppName/1.0; +https://n.news.naver.com)"
USER_AGENT = os.getenv("USER_AGENT")
# Retrieve access token from environment variables
ACCESS_TOKEN = os.getenv('GITHUB_PERSONAL_TOKEN')

## 토크나이저 병렬화 경고 제거용
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# Function to load content from URL
def load_url_content(url):
    headers = {
        'User-Agent': os.environ['USER_AGENT']
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

content = load_url_content(url)

# Initialize SentenceTransformerEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



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
# Split the content into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#docs = text_splitter.split_text(content)
# docs = text_splitter.create_documents(content)
chunks = text_splitter.split_text(content)

# Create Document objects with source metadata
docs = [Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]





### Vector Store ###



# # fastembed 적용
# # pip3 install fastembed -q
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# vectorstore = FAISS.from_documents(
#     documents=splits, embedding=FastEmbedEmbeddings())



# Chroma DB 적용
# from langchain_community.vectorstores import Chroma

# vectorstore = Chroma.from_documents(documents=splits, 
#                                     # embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
#                                     embedding=OpenAIEmbeddings())


# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# # 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
# # 벡터스토어를 생성합니다.
# vectorstore = FAISS.from_documents(
#     documents=splits, embedding=HuggingFaceBgeEmbeddings()
# )


# # FAISS DB 적용
from langchain_community.vectorstores import FAISS

# vectorstore = FAISS.from_documents(
#     documents=splits, embedding=OpenAIEmbeddings())

# Create a vector store from the documents
# vector_store = FAISS.from_texts(docs, embeddings)
vector_store = FAISS.from_documents(docs, embeddings)

# Initialize the retriever with the vector store
#retriever = BM25Retriever(vector_store=vector_store)
retriever = BM25Retriever.from_documents(docs)




query = "회사의 저출생 정책이 뭐야?"

# retriever = vector_store.as_retriever(search_type="similarity")

# print(search_result)





#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = OllamaLLM(model="llama3.1" ) # ChatOpenAI(model_name="gpt-4o", temperature=0)

# # Post-processing
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Question
# print(rag_chain.invoke(query))


# qa_chain = create_retrieval_chain(retriever)
prompt_template = PromptTemplate(template="Answer the question based on the context: {context}\nQuestion: {question}\nAnswer:")

# Load the QA with sources chain
combine_documents_chain = load_qa_with_sources_chain(llm=llm, prompt=prompt_template, document_variable_name="context")

# Combine retriever and combine_documents_chain into a RetrievalQA chain
#retrieval_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_documents_chain)
retrieval_chain = create_retrieval_chain(llm=llm, retriever=retriever, prompt=prompt_template)


response = retrieval_chain.run(query=query)
print(response)