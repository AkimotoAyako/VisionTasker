import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import json  
import requests
import http.client
import time
from core.command_processor import CommandProcessor
from core.LLm_history import Global_LLm_history
from core.Config import SCREEN_jSON_PATH 


def use_LLM(gpt_choice, object_message_humanword):
    if gpt_choice == 'baidu':
        baiduLLM = BaiduLLM(object_message_humanword)
        return baiduLLM.result_2_order()
    elif gpt_choice == 'openai':
        chatgptLLM = ChatGptLLM(object_message_humanword)
        return chatgptLLM.result_2_order()
    elif gpt_choice == 'azure_openai':
        azurechatgptLLM = AzureChatGptLLM(object_message_humanword)
        return azurechatgptLLM.result_2_order()
    elif gpt_choice == 'chatglm':
        chatglmLLM = ChatGlmLLM(object_message_humanword)
        return chatglmLLM.result_2_order()

class LLM:
    def __init__(self, object_message) -> None:
        self.user_input = str(json.loads(object_message))
        
    def get_llm_result(self):
        result = "default"
        return result
    
    def result_2_order(self):
        user_input = self.user_input
        start_time = time.time()
        result = self.get_llm_result()
        # -------------------update llm msg history-----------------------------
        if "此次任务的帮助文档信息" in user_input: # 不记录帮助文档信息和界面信息
            user_input = user_input.split("，此次任务的帮助文档信息")[0]  + "。"
        else:
            user_input = user_input.split("，当前界面有以下按钮")[0]  + "。"

        Global_LLm_history.clear_previous_one_record()
        Global_LLm_history.add_user_input(user_input) #记录user_input

        # -------------------根据产生的执行器命令类型来判断是否需要记录LLm_result-----------------------------
        print("\n*----------------------------- Execution -------------------------------*")
        print('🧐: Oh, I got it! Here we go!\n')

        command_processor = CommandProcessor(result, SCREEN_jSON_PATH)
        order_list = command_processor.generate_exe_command()

        for order in order_list:
            if order['action'] == 'default':
                Global_LLm_history.clear_previous_one_record() #本次user_input不记录
            elif order['action'] == 'tap':
                Global_LLm_history.add_LLM_output(f"A：tap_action：点击['{order['button_content']}']。")
            elif order['action'] == 'keyboard_input':  
                Global_LLm_history.add_LLM_output(f"A：input_action：在输入框：['{order['button_content']}']输入（“{order['data']['input_text']}”）并回车。")
                    
        Global_LLm_history.save_conversation_to_file()
        return order_list

class BaiduLLM(LLM):
    def get_access_token(self, API_KEY, SECRET_KEY):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))
            
    def get_llm_result(self):
        BAIDU_API_KEY = "xxxxxxxx"
        BAIDU_SECRET_KEY = "xxxxxxxx"
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + self.get_access_token(BAIDU_API_KEY, BAIDU_SECRET_KEY)

        Global_LLm_history.add_user_input(self.user_input)
        # Global_LLm_history.save_conversation_to_file()

        headers = {
            'Content-Type': 'application/json'
        }    
        response = requests.request("POST", url, headers = headers, data = Global_LLm_history.get_payload())
        response_json = response.json()
        
        tokens_infor = response_json.get("usage")
        result = response_json.get("result")  # 提取"result"字段的内容 
        print(f"\nErnie Answer：{result}\n")
        
        return result

class ChatGptLLM(LLM):
    def get_llm_result(self):
        Global_LLm_history.add_user_input(self.user_input)

        headers = {
        'Authorization': 'xxxxxxxx xxxxxxxx',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
        }
        conn = http.client.HTTPSConnection("oa.api2d.net")
        conn.request("POST", "/v1/chat/completions", Global_LLm_history.get_payload(), headers)
        
        res = conn.getresponse()
        data = res.read()
        response_data = json.loads(data.decode("utf-8"))
        result = str(response_data['choices'][0]['message']['content'])  # 提取"result"字段的内容
        total_tokens = str(response_data['usage']['total_tokens'])
        print(f"\nChatGpt Answer：{result}\n")

        return result

class AzureChatGptLLM(LLM):
    def get_llm_result(self):
        url = "https://ui-gpt-4.openai.azure.com/openai/deployments/gpt-35-turbo-16k/chat/completions?api-version=2023-07-01-preview"    
        headers = {
        'Content-Type': 'application/json',
        'api-key': 'xxxxxxxx'
        }   
        Global_LLm_history.add_user_input(self.user_input)
        
        response = requests.request("POST", url, headers = headers, data = Global_LLm_history.get_payload())

        data = json.loads(response.text)

        total_tokens = data['usage']['total_tokens']
        result = data['choices'][0]['message']['content']
        print(f"\nAzureChatGpt Answer：{result}\n")
        
        return result

# 本地部署的chatglm
class ChatGlmLLM(LLM):
    def chatglm_convert_prompt(self, payload):
        payload = json.loads(payload)
        prompt = ""  
        for message in payload["messages"]:  
            if message["role"] == "system":  
                prompt += f"\n\n系统：{message['content']}"  
            elif message["role"] == "user":  
                prompt += f"\n\n用户：{message['content']}" 
            elif message["role"] == "assistant":  
                prompt += f"\n\nChatGLM3-6B：{message['content']}"  
        return prompt 
    
    def call_model(self, prompt):
        url = "http://127.0.0.1:8590/chat"
        query = {"human_input": prompt}
        _headers = {"Content_Type": "application/json"}
        with requests.session() as sess:
            resp = sess.post(url,
                    json=query,
                    headers=_headers,
                    timeout=60)
        if resp.status_code == 200:
            resp_json = resp.json()
            predictions = resp_json["response"]
            return predictions
        else:
            return "请求模型失败"
         
    def get_llm_result(self):
        url = "https://ui-gpt-4.openai.azure.com/openai/deployments/gpt-35-turbo-16k/chat/completions?api-version=2023-07-01-preview"    
        headers = {
        'Content-Type': 'application/json',
        'api-key': 'xxxxxxxx'
        }   
        Global_LLm_history.add_user_input(self.user_input)

        prompt = self.chatglm_convert_prompt(Global_LLm_history.get_payload())
        result = self.call_model(prompt)
        # result = json.loads(result)
        print(f"\nChatglm Answer: {str(result)}\n")
        return result            


if __name__ == '__main__':
    from core.screenshot_translator import ScreenshotTranslator

    output_filename = 'data/screenshot/screenshot.json'
    # 检查文件是否存在  
    if os.path.exists(output_filename):  
        with open(output_filename, 'r', encoding='utf-8') as f:    
            f = f.read()  
            object_message_json_data = json.loads(f)
    else:  
        print(f"file {output_filename} do not exist")

    # 将屏幕信息json文件 转换成 易阅读的字符串
    ScreenshotTranslatorTest = ScreenshotTranslator('screenshot/screenshot.json')
    humanword = ScreenshotTranslatorTest.json2humanword()
    message = "Q：在12306应用中预订11月5日从西安到北京的高铁票" 

    taped_element_content = '在此处输入'
    tapped_button_str = '，现在已经点击了按钮' + repr(taped_element_content)+ '后'

    # tapped_button_str = ''

    task_str = str(message)
    task_str = task_str.split('Q：')[-1]
    object_message_humanword = f'Q：{task_str}{tapped_button_str}，当前界面有以下按钮：{humanword}'  
    print(object_message_humanword)
    # object_message = 
    object_message_humanword = json.dumps(object_message_humanword)

    #选择gpt-api   baidu/ openai/ azure_openai
    gpt_choice = 'azure_openai'

    if gpt_choice == 'baidu':
        baiduLLM = BaiduLLM(object_message_humanword)
        baiduLLM.result_2_order()
    elif gpt_choice == 'openai':
        chatgptLLM = ChatGptLLM(object_message_humanword)
        chatgptLLM.result_2_order()
    elif gpt_choice == 'azure_openai':
        azurechatgptLLM = AzureChatGptLLM(object_message_humanword)
        azurechatgptLLM.result_2_order()
    elif gpt_choice == 'chatglm':
        chatglmLLM = ChatGlmLLM(object_message_humanword)
        chatglmLLM.result_2_order()
