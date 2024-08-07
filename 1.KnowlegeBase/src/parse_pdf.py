import fitz  # PyMuPDF
import re
import json

from tqdm import tqdm


signs = ['0', '1', '2', '4', '5', '6', '7', '8', '9', '-']

def extract_job_descriptions(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    all_jobs = []
    current_job = {}
    all_lines = []
    description_idx = -1
    content_idx = 9999
    for page_num in tqdm(range(len(doc)), desc="Processing PDF pages"):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        lines = text.split('\n')
        all_lines.extend(lines)

    for idx, line in tqdm(enumerate(all_lines), desc="Prodessing all lines"):
        if content_idx < idx and all_lines[idx+2] == '-':
            job_content = "".join(all_lines[content_idx: idx+1])
            current_job["职业代号"] = job_code
            current_job["职业名称"] = job_title
            current_job["职业描述"] = job_description
            current_job["职业内容"] = job_content
            all_jobs.append(current_job.copy())
            content_idx += 9999
            continue
        
        if line.startswith("\u3000") and (all_lines[idx+1] not in signs):
            job_title = line[1:]
            job_code = "".join(all_lines[idx-10:idx])
            description_idx = idx+1
            continue

        if line == "主要工作任务:":
            job_description = "".join(all_lines[description_idx:idx])
            content_idx = idx + 1
            continue

    # Save to JSON file
    output_file = 'all_jobs-4th.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_jobs, f, ensure_ascii=False, indent=4)
        print(f"Save all_jobs to {output_file}")

# Path to the PDF file
pdf_path = '../data/2022职业大典-第四类.pdf'
# pdf_path = '../data/中华人民共和国职业分类大典.pdf'

job_descriptions = extract_job_descriptions(pdf_path)

# # Display the results
# for job_code, job_info in job_descriptions.items():
#     print(f"Job Code: {job_code}")
#     print(f"Title: {job_info['title']}")
#     print(f"Description: {job_info['description']}\n")
