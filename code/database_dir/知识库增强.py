import  os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from tqdm import tqdm
from zhipuai import ZhipuAI

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
    # return response

if __name__ == "__main__":
    for name in ["淄博烧烤"]:
        nnn = 0

        dir = f"{name}/txt"
        loader = DirectoryLoader(dir)
        documents = loader.load()

        text_spliter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)  # 400   0:0-200  1:150-350 2:300-400

        split_docs = text_spliter.split_documents(documents)
        contents = [i.page_content for i in split_docs]

        for content in tqdm(contents):
            if nnn >= 50:
                break

            if len(content) < 200:
                continue

            prompt = f"""假设你是一个新闻记者，你需要根据主题词和文章内容中帮我提取有价值和意义的问答对，有助于我进行采访。
主题词：{name}
文章内容：
{content}

请注意，你提取的问答内容必须和主题词高度符合，无需输出其他内容，提取的每个问答返回一个python字典的格式，样例如下：
{{"问":"xxx","答":"xxx"}}
{{"问":"xxx","答":"xxx"}}
提取的问答内容为："""

            answers = get_ans(prompt)
            answers = answers.split("\n")

            for answer in answers:
                if len(answer) < 10:
                    continue
                try :
                    answer = eval(answer)
                except:
                    continue
                if "问" not in answer or "答" not in answer:
                    continue

                file_num = len(os.listdir(dir))

                with open(f"{dir}/{file_num}_____.txt","w",encoding="utf-8") as f:
                    f.write(answer["问"])
                    f.write('\n')
                    f.write(answer["答"])
                    nnn += 1

