from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
loader = TextLoader("PawsitiveHealth/txt.file",encoding="utf-8")


def get_vector_store():
    docs=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        separators=["Disease: "],
        chunk_size=400,
        chunk_overlap=0
    )

    chunks=text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
    )

    vector_store.save_local("./faiss_database")

    return vector_store

def retrieve(query:str,k:int=4)->str:
    vector_store=get_vector_store()
    retriever = vector_store.similarity_search(query,k)
    context=""
    for doc in retriever:
        context+=doc.page_content+"\n\n"
    return context


if __name__ == "__main__":
    print(retrieve("rohan's dog ate chocolate and is now vomitting,panting and shaking"))