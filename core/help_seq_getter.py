import os
from sentence_transformers import SentenceTransformer
import time
import requests
import re
import json
import pandas as pd
# from enum import Enum

# class InstructionType(Enum):
#     RED = 1
#     GREEN = 2
#     BLUE = 3
    
help_get_flag = False  # 启用帮助文档 True / False
BGE_flag = False  # 启用BGE_model  True / False
new_database_flag = False

if help_get_flag and BGE_flag:
    LLM_resort_flag = False  # True / False
    top_k_number = 1
    task_name = "任务内容微调"  # 要输入的任务
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置环境变量
    st_time_bge = time.time()
    BGE_model = SentenceTransformer('BAAI/bge-large-zh')  # Initialize BGE_Model
    print('bge-large-zh ✓ 本次导入model用时：', round(time.time()-st_time_bge, 2), '秒')

def baidu_gpt(given_seq, selected_answers):
    def get_access_token(API_KEY, SECRET_KEY):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))

    API_KEY = "****************"  # 换成你自己的 不要偷看！
    SECRET_KEY = "****************"  # 换成你自己的 不要偷看！
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token(API_KEY, SECRET_KEY)

    user_input = f'我想要在手机上完成“{given_seq}”的任务，有以下帮助文档可以查看：\n{";".join([f"id: {index + 1} 内容：{answer}" for index, answer in enumerate(selected_answers)])}，我现在应该查看哪个帮助文档，告诉我序号就行'
    payload = json.dumps({
    "messages": [
        {
            "role": "user",
            "content": "您需要扮演一个筛选器的角色，我现在有一个任务需要完成，现在有几个帮助语句可以使用，请您帮我把最合适的帮助语句挑选出来，只需要告诉我此帮助语句的id，如果你认为没有合适的帮助语句，请回答'id: 0'"
        },
        {
            "role": "assistant",
            "content": "好的，我将回答'id: \{合适的帮助语句id\}'，若没有合适的帮助语句，我将回答'id: 0'"
        },
        {
            "role": "user",
            "content": user_input
        },
    ]
    })

    headers = {
        'Content-Type': 'application/json'
    }    
    response = requests.request("POST", url, headers=headers, data = payload)

    response_json = response.json()
    result = response_json.get("result")  # 提取"result"字段的内容 
    
    # print(f"\n文心一言回答：{result}\n")

    id_content = None
    match = re.search(r'id[:：] (\d+)', result)
    if match:
        id_content = int(match.group(1))
    else:
        print("Failure to correctly answer!")

    return id_content

def get_top_similarity_seq(given_seq, helpfile_path, instruction_type = 2, pre_command="为这个句子生成表示以用于检索相关文章：", top_k = 1):
    # helpfile_path = '/data4/tangyt/GUI-Detection-Grouping/help_table_nopbd.csv'
    # # helpfile_path = '/data4/tangyt/GUI-Detection-Grouping/help_table.csv'
    qa_table = pd.read_csv(helpfile_path).to_dict('records')
    
    given_seq = given_seq.rstrip('。')
    encoded_question = BGE_model.encode([given_seq], normalize_embeddings=True)  # Encode the given question

    # Initialize a list to store top similarities
    top_similarities = []

    for row in qa_table:
        if pd.notna(row["帮助提问"]) and pd.notna(row["帮助语句内容"]):
            if pd.isna(row["应用名称"]) and pd.notna(previous_app_name):
                row["应用名称"] = previous_app_name

            instruction = f'{row["帮助提问"]}{row["帮助语句内容"]}'

            encoded_row_instruction = BGE_model.encode([instruction], normalize_embeddings=True)
            similarity = encoded_question @ encoded_row_instruction.T

            # print(f'Similarity for row: {similarity}, Content: {instruction}')  #Debugging line

            # Append the similarity score along with the instruction
            top_similarities.append((similarity, row["帮助提问"],row["帮助语句内容"],row[task_name]), instruction)

        previous_app_name = row["应用名称"]

    # Sort the list based on similarity in descending order
    top_similarities.sort(key=lambda x: x[0], reverse=True)

    # Select the top-k results
    top_k_results = top_similarities[:top_k]

    # Extract relevant information from the top-k results
    selected_similarities = [result[0] for result in top_k_results]
    selected_questions = [result[1] for result in top_k_results]
    selected_answers = [result[2] for result in top_k_results]
    selected_tasks = [result[3] for result in top_k_results]
    
    selected_instructions = [result[4] for result in top_k_results]

    if LLM_resort_flag:
        id_content = baidu_gpt(given_seq, selected_instructions)
    else:
        id_content = 1

    if id_content and id_content != 0:
        selected_question = selected_questions[id_content-1]
        selected_answer = selected_answers[id_content-1]
        selected_task = selected_tasks[id_content-1]
        selected_similarity = selected_similarities[id_content-1]
    else:
        return None,None,None,None
    
    return selected_question, selected_answer, selected_task, round(float(selected_similarity), 3)

def help_seq_get(task_string, helpfile_path='data\help_table\help_table.csv' if not new_database_flag else 'data\help_table\demonstration_database.csv'):
    if BGE_flag:
        selected_question, selected_answer, selected_task, selected_similarity = get_top_similarity_seq(task_string, helpfile_path, top_k = top_k_number)

        # Check if any matching answer is found
        if selected_answer is not None:
            print(" \nBGE 任务语句向量化 相似度比较结果：")
            print("Selected Similarity:", selected_similarity)
            print("Selected Question:", selected_question)
            print("Selected Answer:", selected_answer)   
            print("\n")
        else:
            print("\nNo matching answer found.\n")

        if new_database_flag:
            return "如何"+ selected_question+ "?", selected_answer  
        else:
            return selected_question, selected_answer
    
    ##----------------原来的，直接用表格中的任务内容进行相等匹配-------------0
    else:
        data = pd.read_csv(helpfile_path)

        # 使用DataFrame过滤数据
        filtered_data = data[
            # (data['领域范围'] == desired_domain) &
            # (data['应用名称'] == desired_app) &
            (data['任务内容'] == task_string)
        ]

        # 获取帮助语句内容
        help_question = filtered_data['帮助提问'].values[0] if not filtered_data.empty else None
        help_content = filtered_data['帮助语句内容'].values[0] if not filtered_data.empty else None
        return help_question, help_content

