import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import json  
import requests


from core_gpt4v.command_processor import CommandProcessor
from core_gpt4v.LLm_history import GPT4V_history
from core_gpt4v.Config import SCREEN_jSON_PATH, img_4gpt4_out_path, save_path_new

def use_LLM(gpt_choice, object_message_humanword):
    if gpt_choice == 'azure_openai':
        azurechatgptLLM = Gpt4VLLM(object_message_humanword)
        return azurechatgptLLM.result_2_order()


class LLM:
    def __init__(self, object_message) -> None:
        self.user_input = str(json.loads(object_message))
        
    def get_llm_result(self):
        result = "default"
        return result
    
    def result_2_order(self):
        user_input = self.user_input
        result = self.get_llm_result()
        # -------------------update llm msg history-----------------------------
        # if "此次任务的帮助文档信息" in user_input: # 不记录帮助文档信息和界面信息
        #     user_input = user_input.split("，此次任务的帮助文档信息")[0]  + "。"
        # else:
        #     user_input = user_input.split("，当前界面有以下按钮")[0]  + "。"
        # 一次请求实际上会有1次记录
        GPT4V_history.clear_previous_records(del_num = 1)
        GPT4V_history.add_user_textinput(user_input) #记录user_input

        # -------------------根据产生的执行器命令类型来判断是否需要记录LLm_result-----------------------------

        # command_processor = CommandProcessor(result, SCREEN_jSON_PATH)
        # order_list = command_processor.generate_exe_command()

        # for order in order_list:
        #     if order['action'] == 'default':
        #         GPT4V_history.clear_previous_one_record() #本次user_input不记录
        #     elif order['action'] == 'tap':
        #         GPT4V_history.add_LLM_output(f"A：tap_action：id=<SOI>{order['button_content']}<EOI>。")
        #     elif order['action'] == 'keyboard_input':  
        #         GPT4V_history.add_LLM_output(f"A：input_action：id=<SOI>{order['button_content']}<EOI>，输入（“{order['data']['input_text']}”）并回车。")
        GPT4V_history.add_LLM_output(result)
        GPT4V_history.save_conversation_to_file()
        # return order_list
        return ["ok"]

class Gpt4VLLM(LLM):
    def get_llm_result(self):
        GPT4V_history.add_user_input(self.user_input, img_path = save_path_new)
        GPT4V_history.save_conversation_to_file()
        GPT4V_KEY = "xxxxxxxx"
        GPT4V_ENDPOINT = "https://gpt-turbo-4.openai.azure.com/openai/deployments/gpt-4-vision/chat/completions?api-version=2024-02-15-preview"

        headers = {
            "Content-Type": "application/json",
            "api-key": GPT4V_KEY,
        }        
        try:
            # response = requests.post(GPT4V_ENDPOINT, headers=headers, json= global_vars.get_payload())
            response = requests.post(GPT4V_ENDPOINT, headers=headers, json = json.loads(GPT4V_history.get_payload()))
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")
        
        data = response.json()
        # print(data)
        total_tokens = data['usage']['total_tokens']
        print(f"total_tokens: {total_tokens}\n")

        result = data['choices'][0]['message']['content']
        print(f"GPT4V answer: {result}\n")  
        
        return result


if __name__ == '__main__':
    # message = "Q：我想找到我最近浏览过的内容"
    # message = "Q：给“咖啡因从咖啡果中来”发送“hello, 小卞555”"
    # message = "Q：在12306应用中预订11月1日从西安到北京的高铁票"
    # message = "Q：在饿了么中清空购物车"
    # message = "Q：搜索葛东琪的歌曲“悬溺”，播放并且收藏"
    message = "去进入“我的”界面" 
    
    object_message_humanword = f'Q：{message}'  
    print(object_message_humanword)

    #选择gpt-api   baidu/ openai/ azure_openai
    gpt_choice = 'azure_openai'

    if gpt_choice == 'azure_openai':
        azurechatgptLLM = Gpt4VLLM(json.dumps(object_message_humanword))
        orderlist = azurechatgptLLM.result_2_order()
        print(orderlist[0])

