import os
import bs4
from dotenv import load_dotenv

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.docstores import InMemoryDocstore
# from langchain.index_to_docstore_id import SimpleIndexToDocstoreId


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# prompt_model = "google/pegasus-xsum"
# generator_model = "facebook/bart-large-cnn"

# tokenizer = AutoTokenizer.from_pretrained(prompt_model)
# model = AutoModelForCausalLM.from_pretrained(generator_model)



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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load models and tokenizers
prompt_model = "google/pegasus-xsum"
generator_model = "facebook/bart-large-cnn"

prompt_tokenizer = AutoTokenizer.from_pretrained(prompt_model)
prompt = AutoModelForSeq2SeqLM.from_pretrained(prompt_model)

generator_tokenizer = AutoTokenizer.from_pretrained(generator_model)
generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)

# # Initialize the tokenizer and model for Llama 3
# tokenizer = AutoTokenizer.from_pretrained("facebook/llama-3")
# model = AutoModelForCausalLM.from_pretrained("facebook/llama-3")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


# Create pipeline for question answering
qa_pipeline = pipeline("summarization", model=generator, tokenizer=generator_tokenizer)

# def initialize_tokenizer():
#     global tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define post-processing function
def format_docs(docs):
    return "\n\n".join(doc["page_content"] for doc in docs)

def generate_response(prompt, model, tokenizer, max_new_tokens=50, max_input_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    attention_mask = inputs.attention_mask
    outputs = model.generate(inputs.input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def rag_chain_simple(question, retriever):
    docs = retriever.similarity_search(question)
    return docs

# Chain process
def rag_chain(question, retriever):
    # Retrieve relevant documents
    docs = retriever.retrieve(question)
    
    # Format documents
    context = format_docs(docs)
    
    # Prepare the prompt with context
    inputs = prompt_tokenizer(context, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = prompt.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    summarized_context = prompt_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Generate final answer
    final_input = generator_tokenizer(question + " " + summarized_context, return_tensors="pt", max_length=1024, truncation=True)
    final_ids = generator.generate(final_input["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    final_output = generator_tokenizer.decode(final_ids[0], skip_special_tokens=True)
    
    return final_output

# Example retriever function (stubbed for demonstration purposes)
def example_retriever(query):
    # This function should retrieve relevant documents based on the query.
    # Here, it returns a stubbed list of documents.
    return [{"page_content": "Task decomposition involves breaking down a complex task into smaller, more manageable components."}]


def rag_func():
        

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


    # # Documents
    # question = "What kinds of pets do I like?"
    # document = "My favorite pet is a cat."



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


    # # Chroma DB 적용
    # from langchain_community.vectorstores import Chroma

    # vectorstore = Chroma.from_documents(documents=splits, 
    #                                     # embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
    #                                     embedding=OpenAIEmbeddings())


    # 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(
        documents=splits, embedding=HuggingFaceBgeEmbeddings()
    )

    query = "회사의 저출생 정책이 뭐야?"

    # retriever = vectorstore.as_retriever()

    # retriever = vectorstore.as_retriever(search_type="similarity")
    # search_result = retriever.get_relevant_documents(query)
    # print(search_result)


    # Example usage
    # Retrieve relevant documents
    retrieved_docs = rag_chain_simple(query, vectorstore)

    print("--------------",flush=True)
    print(retrieved_docs,flush=True)
    print("--------------",flush=True)
    
    # Generate a prompt combining the query and retrieved documents
    
    # prompt = query + "\n\n" + "\n".join([doc['text'] for doc in retrieved_docs])
    prompt = query + "\n\n" + "\n".join([doc.page_content for doc in retrieved_docs])

    # Truncate prompt if it exceeds max_input_length
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding="max_length")
    truncated_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    
    
    # Generate response using GPT-2
    response = generate_response(truncated_prompt, model, tokenizer)

    # response = generate_response(prompt, model, tokenizer)
    print(response)

    embeddings = HuggingFaceBgeEmbeddings()  # Initialize with appropriate parameters
    
    # Create an index using FAISS
    faiss_index = FAISS.build_index(embeddings)
    
    # Create an in-memory docstore
    docstore = InMemoryDocstore()  # or use a persistent one if needed
    
    # Create an index to docstore ID mapping
    index_to_docstore_id = SimpleIndexToDocstoreId()

    retriever = FAISS(index=faiss_index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

    # Question
    # result = rag_chain(query, retriever)
    result = rag_chain_simple(query, retriever)
    print(result)



    # #### RETRIEVAL and GENERATION ####

    # # Prompt
    # prompt = hub.pull("rlm/rag-prompt")

    # # LLM
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # # Post-processing
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    # # Chain
    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | generator
    #     | StrOutputParser()
    # )

    # # Question
    # rag_chain.invoke("What is Task Decomposition?")


    
if __name__ == "__main__":
    # initialize_tokenizer()
    process = multiprocessing.Process(target=rag_func)
    process.start()
    process.join()


