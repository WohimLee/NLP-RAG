
import json
from zhipuai import ZhipuAI
from tqdm import tqdm


codes = ['2-', '4-', '5-', '6-']
filters = ["成为", "是啥", "是什么"]

def get_answer(prompt):
    client = ZhipuAI(api_key="0f56bcd3ce36d22b5b6564de4faeebfe.nvc4AlZb8rw1WhFG") # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-0520",  # 填写需要调用的模型名称
        messages=[
            {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
            {"role": "user", "content": prompt}
        ],
        stream = True,
        max_tokens = 1024,
        temperature = 0.95,
        top_p = 0.7
    )
    result = ''
    for chunk in response:
        result += chunk.choices[0].delta.content
    return result
    
def post_process(json_file):
    res = []
    with open(json_file, "r", encoding='utf-8') as f:
        jobs = json.load(f)
        for job in tqdm(jobs, desc="Processing json"):
            if any(f in job["instruction"] or f in job["output"] for f in filters):
                continue
            res.append(job)
    with open("jobs_qa-res.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
        
if __name__ == "__main__":
    # file_path = "all_jobs-test.json"
    # file_path = "data/all_jobs-4th.json"
    
    file_path = "data/all_jobs-raw.json"
    
    all_qa = []
    with open(file_path, "r", encoding="utf-8") as file:
        jobs = json.load(file)
        for job in tqdm(jobs, desc="Porecessing json"):
            if job["职业代号"][:2] not in codes:
                continue
            del job["职业代号"]
            prompt = f'''根据以下文本内容构建20条问答对数据集，要求如下：
Q的要求：结合所给文本模拟用户提出自己的需求；非常口语化；简短，10个字以内
A的要求：结合所给文本内容，给用户推荐相关的职业，不是要成为的职业，而是推荐相关为用户提供服务;不限字数，尽量详细
文本内容：
{job}
生成的QA对使用以下格式存储下来：
[{{"Q":"xxx", "A":"xxx"}}, {{"Q":"xxx", "A":"xxx"}}]'''
            answer = get_answer(prompt)
            try:
                qa_list = json.loads(answer)   
                for qa in qa_list:
                    Q, A = qa["Q"], qa["A"]
                    data = {
                        "instruction": Q,
                        "input": "",
                        "output": A
                    }
                    all_qa.append(data)    
            except:
                pass
            n = len(all_qa)
            if n !=0 and n % 5 == 0:
                with open("jobs_qa.json", "w", encoding="utf-8") as file:
                    json.dump(all_qa, file, indent=2, ensure_ascii=False)
                    print(f"[INFO]: Save {n} QAs to jobs_qa.json")


    json_file = 'jobs_qa.json'
    post_process(json_file)