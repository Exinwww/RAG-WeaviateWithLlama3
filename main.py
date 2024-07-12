from langchain_community.document_loaders import TextLoader # 文本加载器
from langchain.text_splitter import CharacterTextSplitter # 文本分块器
from langchain_community.embeddings import OllamaEmbeddings # Ollama嵌入
import weaviate # Weaviate客户端
from weaviate.embedded import EmbeddedOptions # Weaviate嵌入选项
from langchain.prompts import ChatPromptTemplate # 聊天提示模板
from langchain_community.chat_models import ChatOllama # ChatOllma聊天模型
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser # 输出解析器
from langchain_community.vectorstores import Weaviate # 向量数据库

# 加载文本
loader = TextLoader('./sysu_en.txt')
documents = loader.load()
# 文本分块
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 嵌入以及存储到向量数据库
client = weaviate.Client(
    embedded_options = EmbeddedOptions()
)
print('store vector')
vector_store = Weaviate.from_documents(
    client = client,
    documents = chunks,
    embedding = OllamaEmbeddings(model='llama3'),
    by_text = False
)
# 检索&增强
# 检索器
retriever = vector_store.as_retriever()


# LLM提示模板
template = """You are an assistant for question-answering tasks. 
   Use the following pieces of retrieved context to answer the question. 
   Just tell me the answer.
   If you don't know the answer, just say that you don't know. 
   Use three sentences maximum and keep the answer concise.
   Question: {question} 
   Context: {context} 
   Answer:
   """
prompt = ChatPromptTemplate.from_template(template)
# 生成器
llm = ChatOllama(model="llama3", temperature=10)
rag_chain = (
    # 上下文信息
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()
)

# Q1: What was the original name of Sun Yat-sen University
# Q2: Who founded Sun Yat-sen University?
# Q3: Where is Sun Yat-sen University?
# Q4: What are the advantages of Sun Yat-sen University?
# Q5: What are the main development disciplines of Sun Yat-sen University?

if __name__ == '__main__':
    print('######################')
    while True:
        print('[EXIT to exit.]')
        query = input("## Question >> ")
        if query == 'EXIT': break
        answer = rag_chain.invoke(query)
        print('** Answer: ')
        print(answer)

