import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import pickle
import os

def add_emb(f_index,emb):
    pass

if __name__ == "__main__":
    if os.path.exists("contents.pkl") == False:

        loader = DirectoryLoader("new_data")
        documents = loader.load()

        # text_spliter = CharacterTextSplitter(chunk_size=200,chunk_overlap=0) # 400   0:0-200  1:200-400
        text_spliter = CharacterTextSplitter(chunk_size=300,chunk_overlap=50) # 400   0:0-200  1:150-350 2:300-400

        split_docs = text_spliter.split_documents(documents)
        contents = [i.page_content for i in split_docs]
        with open("contents.pkl","wb") as f:
            pickle.dump(contents,f)
    else:
        with open("contents.pkl","rb") as f:
            contents = pickle.load(f)


    sentence_model = SentenceTransformer(".\\moka-ai_m3e-base")
    # emb =

    faiss_index = faiss.IndexFlatL2(sentence_model.get_sentence_embedding_dimension())

    print("...向量构建中...")
    faiss_index.add(sentence_model.encode(contents))

    # faiss_index.add(sentence_model.encode(["你好","abc"]))  # index = 0
    # faiss_index.add(sentence_model.encode(["我爱你"])) #index = 1
    # faiss_index.add(sentence_model.encode(["我好"])) # index = 2
    #
    # # “我好爱你”
    # distance , index = faiss_index.search(sentence_model.encode(["我好爱你"]),2)

    while True:
        input_text = input("请输入：")
        d,idx = faiss_index.search(sentence_model.encode(contents),1) # mlivus
        print(contents[idx][0])





