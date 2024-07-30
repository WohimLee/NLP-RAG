import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import pickle
from zhipuai import ZhipuAI
import os
import pandas as pd

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
    ans = ""
    for trunk in response:
        ans += trunk.choices[0].delta.content
    return ans

class MyDocument():
    def __init__(self,dir):
        loader = DirectoryLoader(dir)
        documents = loader.load()

        if os.path.exists("contents.pkl") == False:
            text_spliter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)  # 400   0:0-200  1:150-350 2:300-400

            split_docs = text_spliter.split_documents(documents)
            contents = [i.page_content for i in split_docs]
            with open("contents.pkl", "wb") as f:
                pickle.dump(contents, f)
        else:
            with open("contents.pkl", "rb") as f:
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
    def __init__(self,emb_dir,contents):
        self.emb_model = MyEmbModel(emb_dir)

        if os.path.exists("faiss_index.pkl") == False:
            index = faiss.IndexFlatL2(emb_model.get_sentence_embedding_dimension())
            embs = self.emb_model.to_emb(contents)
            self.add(embs)

            with open("faiss_index.pkl","wb") as f:
                pickle.dump(index,f)
        else:
            with open("faiss_index.pkl","rb") as f:
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


if __name__ == "__main__":
    # 1. ：什么是大模型？
    # 1. : 什么时间开始火的？
    # 2 ： 大模型应用于什么领域?
    # 2 ： 有什么关键技术
    # 3 ： 大模型的未来如何？
    # question_list = ["什么是大模型？","什么时间开始火的？","大模型应用于什么领域？","有什么关键技术？","大模型的未来如何？"]

    question_content = pd.read_excel('prompt.xlsx')
    prompt_data = question_content.groupby("段落")

    document = MyDocument("new_data")

    emb_database = MyEmbDatabase(".\\moka-ai_m3e-base",document.contents)


    report = ""
    for p_n,group in prompt_data:
        print(f"第{p_n}段内容生成中......")
        p_n_content = []
        for question in group["prompt"]:

            search_result = emb_database.search(question,3)

            # print("向量库检索内容：",search_result)
            search_result = "\n".join(search_result)

            prompt = f"请根据已知内容简洁明了的回复用户的问题，已知内容如下：```{search_result}```,用户的问题是：{question}，如何已知内容无法回答用户的问题，请直接回复：不知道，无需输出其他内容"
            ans = get_ans(prompt)
            ans = ans.replace("\n","")
            p_n_content.append(ans)
            # ans = "    " + ans
            # report += ans
            # report += "\n"
            # print(report)
            # print("*"*50)

        report += ( "    "+ "".join(p_n_content))
        report += "\n"

        print(report)
        print("*"*30)

    with open("report.txt","w",encoding="utf-8") as f:
        f.write(report)







