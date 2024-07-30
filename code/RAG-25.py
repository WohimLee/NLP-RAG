# coding:utf-8
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import pickle
from zhipuai import ZhipuAI
import os
import pandas as pd
import time
import gradio as gr
import random

def get_ans(prompt):
    client = ZhipuAI(api_key="0f56bcd3ce36d22b5b6564de4faeebfe.nvc4AlZb8rw1WhFG")

    response = client.chat.completions.create(
        model="glm-4",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        top_p=0.3,
        temperature=0.45,
        max_tokens=1024,
        stream=True,
    )
    # ans = ""
    # for trunk in response:
    #     ans += trunk.choices[0].delta.content
    # return ans
    return response

class MyDocument():
    def __init__(self,dir,name):
        loader = DirectoryLoader(dir)
        documents = loader.load()

        if os.path.exists(os.path.join(".cache",f"{name}_contents.pkl")) == False:
            text_spliter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)  # 400   0:0-200  1:150-350 2:300-400

            split_docs = text_spliter.split_documents(documents)
            contents = [i.page_content for i in split_docs]
            with open(os.path.join(".cache",f"{name}_contents.pkl"), "wb") as f:
                pickle.dump(contents, f)
        else:
            with open(os.path.join(".cache",f"{name}_contents.pkl"), "rb") as f:
                contents = pickle.load(f)
        self.contents = contents

class MyEmbModel():
    def __init__(self,model_dir):
        self.model = SentenceTransformer(model_dir)


    def to_emb(self,sentence):
        if isinstance(sentence,str):
            sentence = [sentence]
        return self.model.encode(sentence)

class MyEmbDatabase():
    def __init__(self,emb_dir,contents,name):
        self.emb_model = MyEmbModel(emb_dir)

        if os.path.exists(os.path.join(".cache",f"{name}_faiss_index.pkl")) == False:
            index = faiss.IndexFlatL2(self.emb_model.model.get_sentence_embedding_dimension())
            embs = self.emb_model.to_emb(contents)
            index.add(embs)

            with open(os.path.join(".cache",f"{name}_faiss_index.pkl"),"wb") as f:
                pickle.dump(index,f)
        else:
            with open(os.path.join(".cache",f"{name}_faiss_index.pkl"),"rb") as f:
                index = pickle.load(f)
        self.index = index
        self.contents = contents

    def add(self,emb):
        self.index.add(emb)

    def search(self,content,topn=3):
        if isinstance(content,str):
            content = self.emb_model.to_emb(content)

        distances , idxs = self.index.search(content,topn) # mlivus

        results = [self.contents[i] for i in idxs[0]]
        return results

class MyDataBase:
    def __init__(self,path,name=None):
        if name is None:
            name = path

        if os.path.exists(os.path.join(path,"txt")) == False:
            print("对应的知识库txt文件夹不存在")
            exit(-999)
        if os.path.exists(os.path.join(path,"prompt.xlsx")) == False:
            print("prompt模板不存在")
            exit(-998)
        if os.path.exists(".cache") == False:
            os.mkdir(".cache")

        self.prompt_data = pd.read_excel(os.path.join(path,"prompt.xlsx"))
        self.document = MyDocument(os.path.join(path,"txt"),name)
        self.emb_database = MyEmbDatabase("..\\moka-ai_m3e-base", self.document.contents,name)

    def search(self,text,topn=3):
        return self.emb_database.search(text,topn)

def load_database(dir_path="database_dir"):
    dirs = [name for name in os.listdir(dir_path) if os.path.isdir(f"{dir_path}/{name}")]

    database_list = []
    database_namelist = []

    for dir in dirs:
        database = MyDataBase(f"{dir_path}/{dir}",dir)

        database_list.append(database)
        database_namelist.append(dir)

    return database_list,database_namelist

def function1():
    pass

def function2():
    pass

if __name__ == "__main__":
    database_list, database_namelist = load_database()

    # ----------------------------- 界面控件 -----------------------------
    input1 = gr.Dropdown(choices=database_namelist,label="知识库选择",value=database_namelist[0])
    input2 = gr.DataFrame(database_list[0].prompt_data,height=400)
    input3 = gr.UploadButton(label="上传知识库",file_count="directory")

    output1 = gr.Textbox(label="报告生成过程", lines=11, max_lines=14)
    output2 = gr.Textbox(label="报告生成内容", lines=11, max_lines=14)


    # ----------------------------- 界面启动 -----------------------------

    interface1 = gr.Interface(function1,[input1,input3,input2],[output1,output2],submit_btn="点击生成报告",clear_btn=gr.Button("clear"),visible=False)
    interface2 = gr.Interface(function2,"text","text")

    tab_interface = gr.TabbedInterface([interface1,interface2],["报告生成","知识库问答"],title="RAG报告生成和问答")

    with tab_interface as tab_interface:

        tab_interface.launch(server_name="0.0.0.0",server_port=9999,show_api=False, auth=("username", "password"))