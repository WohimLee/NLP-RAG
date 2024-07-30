import os
import fitz
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

if __name__ == "__main__":

    files = os.listdir("./data2/")

    for file in files:

        doc = fitz.open(os.path.join("./data2",file))
        for page in doc :
            text = page.get_text()

            prompt = f"""以下是从pdf的文章中提取的内容，最后一句话存在被截断的可能，你需要去掉文章中的冗余无用信息，比如页眉、页脚，还需要去掉被截断的文本，并保留其他原文中的中文内容输出，请注意文章的流畅性和通顺性，无需总结或者输出其他内容，文章内容如下：
文章内容：
```
{text}
```
无需输出其他内容，保持原文内容输出是：
"""
            prompt = prompt
            ans = get_ans(prompt)
            ans = ans.replace("\n\n","\n")
            with open(os.path.join("new_data", file[:-4] + ".txt"), "a+", encoding="utf-8") as f:
                f.write(ans)
                f.write("\n")
