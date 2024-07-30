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
import shutil
import  jieba.analyse as aly
from collections import Counter

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

def function1(name,input3,input2):
    global database_list, database_namelist
    database = database_list[database_namelist.index(name)]


    result1 = []
    result2 = []

    result1.append("内容解析中......")
    yield "\n".join(result1),"\n".join(result2)

    all_report = ""
    for p_n, group in database.prompt_data.groupby("段落"):
        result1.append(f"第{p_n}段内容生成中......")

        yield "\n".join(result1), "\n".join(result2)

        p_n_content = []
        for question in group["prompt"]:
            search_result = database.search(question, 3)

            search_result = "\n".join(search_result)

            prompt = f"请根据已知内容简洁明了的回复用户的问题，已知内容如下：```{search_result}```,用户的问题是：{question}，如何已知内容无法回答用户的问题，请直接回复：不知道，无需输出其他内容"

            response = get_ans(prompt)
            result1.append("检索及回答内容:\n")
            for trunk in response:
                result1[-1] += trunk.choices[0].delta.content
                yield "\n".join(result1), "\n".join(result2)


            result1[-1] = result1[-1].replace("\n", "")
            p_n_content.append(result1[-1])

            result1.append("*"*30)
            yield "\n".join(result1), "\n".join(result2)

        prompt_report = f"你是一个大学教授，你需要根据相关内容，来撰写一段内容，生成的结果必须严格来自相关内容，语言必须严谨、符合事实，不能使用第一人称，相关内容如下：\n```\n{''.join(p_n_content)}\n```\n生成的结果为："

        result1.append("第一段报告内容:\n")
        result2.append("\t\t\t")
        yield "\n".join(result1), "\n".join(result2)

        response = get_ans(prompt_report)

        for trunk in response:
            result1[-1] += trunk.choices[0].delta.content
            result2[-1] += trunk.choices[0].delta.content

            result1[-1] = result1[-1].replace("\n", "")
            result2[-1] = result2[-1].replace("\n", "")
            yield "\n".join(result1), "\n".join(result2)


        all_report += result2[-1]
        all_report += "\n"

        result1.append("*"*30)
        yield "\n".join(result1), "\n".join(result2)



def function2():
    pass

def get_type_name(files):

    content = []
    for file in files:
        try:
            with open(file.name,encoding="utf-8") as f:
                data = f.readlines(1)
                content.extend(aly.tfidf(data[0]))
        except:
            continue
    count = Counter(content)
    kw = count.most_common(2)

    return "".join([i[0] for i in kw])

def upload(files):
    global  database_list, database_namelist,input1

    check_txt = False
    check_prompt_xlsx = False

    for file in files:
        if check_txt and check_prompt_xlsx:
            break
        if file.name.endswith(".txt"):
            check_txt = True
        elif file.name.endswith("prompt.xlsx"):
            check_prompt_xlsx = True
    else:
        if check_txt == False:
            raise Exception("请上传包含txt文档的文件夹")
        if check_prompt_xlsx  == False:
            raise Exception("请上传包含prompt.xlsx的文件夹")

    type_name = get_type_name(files)
    save_path = os.path.join("database_dir", type_name)

    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path,"txt"))
    for file in files:
        if file.name.endswith(".txt"):
            shutil.copy(file.name,os.path.join(save_path,"txt"))
        elif file.name.endswith("prompt.xlsx"):
            shutil.copy(file.name, save_path)

    database = MyDataBase(save_path,type_name)
    database_list.append(database)
    database_namelist.append(type_name)
    input1.choices.append((type_name,type_name))

    return type_name,database.prompt_data

def database_change(name):
    global database_list, database_namelist

    return database_list[database_namelist.index(name)].prompt_data

if __name__ == "__main__":
    database_list, database_namelist = load_database()

    # ----------------------------- 界面控件 -----------------------------
    input1 = gr.Dropdown(choices=database_namelist,label="知识库选择",value=database_namelist[0])
    input2 = gr.DataFrame(database_list[0].prompt_data,height=400)
    input3 = gr.UploadButton(label="上传知识库",file_count="directory")

    input4 = gr.Dropdown(choices=database_namelist,label="知识库选择",value=database_namelist[0])

    output1 = gr.Textbox(label="报告生成过程", lines=11, max_lines=14)
    output2 = gr.Textbox(label="报告生成内容", lines=11, max_lines=14)


    # ----------------------------- 界面启动 -----------------------------

    interface1 = gr.Interface(function1,[input1,input3,input2],[output1,output2],submit_btn="点击生成报告",clear_btn=gr.Button("clear",visible=False),allow_flagging="never")
    interface2 = gr.Interface(function2,[input4],"text",allow_flagging="never")

    tab_interface = gr.TabbedInterface([interface1,interface2],["报告生成","知识库问答"],title="RAG报告生成和问答")

    with tab_interface as tab_interface:
        input1.change(database_change,input1,input2)
        input3.upload(upload,input3,[input1,input2])
        tab_interface.launch(server_name="0.0.0.0",server_port=9999,show_api=False, auth=("username", "password"))