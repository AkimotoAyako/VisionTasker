import json 
import base64 
import os
#选择gpt-api   baidu/  azure_openai / chatglm
gpt_choice = 'azure_openai'
#选择language   zh / en
language = 'zh'

from core.LLm_history import LLmHistory
from core_gpt4v.Config import img_4gpt4_out_path,save_path_new, LLM_history_PATH

    
class LLMHistory_GPt4V(LLmHistory):
    def __init__(self, language, gpt_choice):
        self.language = language
        self.gpt_choice = gpt_choice
        self.global_payload = self.set_preprompt()  # 设置初始值为 store_payload
        
    def save_conversation_to_file(self, task_str = "llm_history", LLM_history_folder = LLM_history_PATH):
        data = json.loads(self.global_payload)
        hisory = data["messages"]
        history_path = LLM_history_folder + "/" + task_str + ".json" 
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        json.dump(hisory, open(history_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
   
    def set_preprompt(self):    
        if self.language == 'zh':
            message_list = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "您需要扮演一个智能助手的角色，现在我需要在手机应用上完成指定任务，但不清楚如何操作，\
告诉我在手机界面上对哪个元素进行点击操作或者输入特定内容以完成该任务。"
                        }
                ]
                },
                #1
                { 
                    "role": "user",
        #             "content": "您需要扮演一个智能助手的角色，现在我需要在手机应用上完成指定任务，但不清楚如何操作，\
        # 告诉我在手机界面上对哪个元素进行点击操作或者输入特定内容以完成该任务。我的提问是Q，您的回答是A。我将执行您给的操作指令并把界面反馈给您（每次1个Q），您再告诉我下一步（每次1个A）。您只需要提供一步的操作命令."
                    "content": [
                        {
                            "type": "text",
                            "text": "我的提问是Q，您的回答是A。我将执行您给的操作指令并把界面反馈给您（每次1个Q），您再告诉我下一步（每次1个A）。您每次只需要提供一步的操作命令."
                        }
                ]
                },
                {
                    "role": "assistant",
                    # "content": "好的，我将扮演智能助手的角色帮助你完成指定任务，每次回答我只提供一步的操作命令。"
                    "content": [
                        {
                            "type": "text",
                            "text": "好的，我将扮演智能助手的角色帮助你完成指定任务，你每次给我提供任务信息和界面，每次回答我只提供一步的操作命令。"
                        }
                ]
                },
                #2
                { 
                    "role": "user",
        #             "content": "每次回答我一步，你需要根据任务要求、已经点击过的控件、键盘输入过的内容，\
        # 谨慎地决定操作指令中的值：xxx, yyy。xxx应为我提供的控件完整内容。如果我为你提供了帮助文档的信息，请您的回答首先以帮助文档为参考。"
                    "content": [
                        {
                            "type": "text",
                            "text": "每次回答我一步，你需要根据任务要求，\
谨慎地决定操作指令中的值：xxx, yyy。xxx应为已提供的控件id名称，yyy为应该在输入栏内输入的文字信息。"
                        }
                ]
                },
                {
                    "role": "assistant",
        #             "content": "好的，我每次只回答一步，并且会根据任务要求、已经点击过的控件、键盘输入过的内容，或者你提供的帮助文档信息，\
        # 谨慎地决定操作指令中的值：xxx, yyy。xxx应为你提供的控件完整内容。如果你提供了帮助文档的信息，我的回答将首先以帮助文档为参考。"
                    "content": [
                        {
                            "type": "text",
                            "text": "好的，我每次只回答一步，并且会根据任务要求，\
谨慎地决定操作指令中的值：xxx, yyy。xxx应为已提供的控件名称，yyy为应该在输入栏内输入的文字信息。"
                        }
                ]
                },
                { 
                    "role": "user",
                    # "content": "你需要根据任务要求和当前界面的所有控件信息，注意当前界面可能不是初始界面，可能已经完成了一些操作，你要判断接下来应该进行什么操作，提供一个规范的操作命令。"
                    "content": [
                        {
                            "type": "text",
                            "text": "你需要根据任务要求和当前界面的所有控件信息，注意当前界面可能不是初始界面，可能已经完成了一些操作，你要判断接下来应该进行什么操作，提供一个规范的操作命令。"
                        }
                ]
                },
                {
                    "role": "assistant",
                    # "content": "好的，我一定会根据任务要求和当前界面的所有控件信息，为您提供一个规范的操作命令。"
                    "content": [
                        {
                            "type": "text",
                            "text": "好的，我一定会根据任务要求和当前界面的所有控件信息，为您提供一个规范的操作命令。"
                        }
                ]
                },
                #3
                { 
                    "role": "user",
        #             "content": "若进行点击操作，则务必用该格式进行回答：A：tap_action: id=<SOI>xxx<EOI>。\
        # 若进行输入内容操作，则在输入框中进行操作，则务必用该格式进行回答： A：input_action: id=<SOI>xxx<EOI> 输入（“yyy”）并回车。\
        # xxx应为已提供的控件id，yyy为应该输入的文字信息，不要说下一步，不重复你提供的界面信息，不要有多余的解释。"
                    "content": [
                        {
                            "type": "text",
                            "text": "若进行点击操作，则务必用该格式进行回答：A：点击xxx。\
若进行输入内容操作，则在输入框中进行操作，则务必用该格式进行回答： A：在输入框xxx内输入（“yyy”）并回车。\
xxx应为已提供的控件名称，yyy为应该在输入栏内输入的文字信息。"
                        }
                ]
                },
                {
                    "role": "assistant",
            #         "content": "好的，若进行点击操作，则我会用该格式进行回答：A：tap_action: 点击['xxx']。\
            # 若进行输入内容操作，则在输入框中进行操作，我会用该格式进行回答：A：input_action: 在输入框: ['xxx']输入（“yyy”）并回车。\
            # xxx, yyy均为文字。xxx应为已提供的控件完整内容。"
                    # "content": "好的，我明白了两种类型的操作命令，我会按格式输出操作命令。"
                    "content": [
                        {
                            "type": "text",
                            "text": "好的，我明白了两种类型的操作命令，我会按格式输出操作命令。第一种为点击操作 A：点击xxx。第二种为在搜索框输入文字的输入操作为 A：在输入框xxx内输入（“yyy”）并回车。"
                        }
                ]
                },
                #5
                #8
                { 
                    "role": "user",
                    # "content": "如果您在界面中没有找到要点击的控件，请回复我：A：无操作；如果您判定当前界面已经完成了任务，请回复我：A：任务已完成。"
                    # "content": "您每次只需提供一步的操作命令，只需要一条input_action命令或者tap_action命令，如果您判定当前界面已经完成了任务，请回复我：A：end_action: 任务已完成。"
                    "content": [
                        {
                            "type": "text",
                            "text": "您每次只需提供一步的操作命令，如果您判定当前界面已经完成了任务，请回复我：A：任务已完成。"
                        }
                ]
                },
                {
                    "role": "assistant",
                    # "content": "好的，如果我在界面中没有找到要点击的控件，我将回复：A：无操作；\
                    #     如果我判定当前界面已经完成了任务，我将回复：A：任务已完成。请告诉我您需要完成什么任务？"
                    # "content": "好的，我每次只提供一步、一种操作命令，如果我判定当前界面已经完成了任务，我将回复：A：end_action: 任务已完成。请告诉我您需要完成什么任务？"
                    "content": [
                        {
                            "type": "text",
                            "text": "好的，我每次只提供一步、一种操作命令，如果我判定当前界面已经完成了任务，我将回复：A：任务已完成。请告诉我您需要完成什么任务？"
                        }
                ]
                }
            ]
        elif self.language == 'en':
            message_list = [
                #1
                { 
                    "role": "user",
                    "content": "You need to act as an intelligent assistant. Now, I need to complete a specific task on a mobile app but am unsure how to proceed. Please tell me which element to click on or what specific content to input on the mobile interface to accomplish the task. My question is Q, and your answer is A. I will execute the instructions you provide and give you feedback on the interface (one Q at a time), and you can then instruct me on the next step (one A at a time). You only need to provide one step of the operation command."
                },
                {
                    "role": "assistant",
                    "content": "I will act as an intelligent assistant to help you complete the specified task. Each time you ask, I will provide only one step of the operation command."
                },
                #2
                { 
                    "role": "user",
                    "content": "Each time you answer me, you need to carefully determine the values in the operation command based on task requirements, previously clicked controls, and keyboard input: xxx, yyy. xxx should be the complete content of the control I provide. If I have given you information from the help document, please consider it as a reference in your response."
                },
                {
                    "role": "assistant",
                    "content": "I will only answer one step at a time, and I will carefully determine the values in the operation command based on task requirements, previously clicked controls, keyboard input, or information you provide from the help document: xxx, yyy. xxx should be the complete content of the control you provide. If you provide information from the help document, my response will prioritize the help document as a reference."
                },
                { 
                    "role": "user",
                    "content": "You need to consider the task requirements and all control information on the current interface. Note that the current interface may not be the initial one, and some operations may have already been completed. You should assess what action to take next and provide a standardized operation command."
                },
                {
                    "role": "assistant",
                    "content": "I will definitely provide you with a standardized operation command based on the task requirements and all control information on the current interface."
                },
                #3
                { 
                    "role": "user",
                    "content": "If performing a tap action, be sure to answer in this format: A: tap_action: Click['xxx']. If performing a content input action in an input box, be sure to answer in this format: A: input_action: In the input box:['xxx'] type(“yyy”) and press Enter. xxx, yyy are both text, and xxx should be the complete content of the provided control. Avoid mentioning the next step, do not repeat the interface information you provided, and avoid unnecessary explanations."
                },
                {
                    "role": "assistant",
                    "content": "I understand the two types of operation commands, and I will output the commands in the specified format."
                },
                #5
                { 
                    "role": "user",
                    "content": "Here is an example of a tap action:\n\
        Q: I want to go back to the main page. Let's think step by step. The current interface is as follows:\n\
        First row - From left to right, the buttons are:\n\
        ['Forward or skip to the right, play, or YouTube']; Input box: ['Search for videos, anime, or creators']; ['Search'];\n\
        Second row - ['ADB Keyboard (ON)']\n\
        A: tap_action: Click['Forward or skip to the right, play, or YouTube']."
                },
                {
                    "role": "assistant",
                    "content": "Okay, if performing a tap action, I will answer in this format: A: tap_action: Click['xxx']. Now, please provide me with an example of a content input action."
                },
                #6
                { 
                    "role": "user",
                    "content": "Here is an example of a content input action:\n\
        Q: I want to watch 'Jay Chou Concert.' Let's think step by step. The current interface is as follows:\n\
        First row - From left to right, the buttons are:\n\
        ['Mirror flip icon key']; Input box: ['Default text: Lady Gaga']; ['Search, magnify key']; ['si3']; ['Mail, message symbol key'];\n\
        Second row - From left to right, the controls are:\n\
        ['Home']; ['Translation, or text-with-image key']; ['Member Purchase']; ['My']; ['Plus key'];\n\
        A: input_action: In the input box: ['Default text: Lady Gaga'] type('Jay Chou Concert') and press Enter."
                },
                {
                    "role": "assistant",
                    "content": "Okay, if performing a content input action, I will answer in this format: A: input_action: In the input box: ['xxx'] type('yyy') and press Enter."
                },
            ]

        if self.gpt_choice == 'baidu':
            dict_message = {
            "messages": message_list,
            }
        elif self.gpt_choice == 'chatglm':
            dict_message = {
            "messages": message_list,
            }
        elif self.gpt_choice == 'azure_openai':
            dict_message = {
            "messages": message_list,
            "temperature": 0.7,
            "top_p": 0.90,
            "max_tokens": 4096

            }
            
        store_payload = json.dumps(dict_message)
        return store_payload

    def clear_previous_records(self, del_num):  
        payload = json.loads(self.global_payload) 
        for i in range(del_num):
            payload["messages"].pop()
        self.global_payload = json.dumps(payload) 
         
    def add_user_input(self, user_input, id_img_path = img_4gpt4_out_path, img_path = save_path_new):
        
        # encoded_id_image = base64.b64encode(open(id_img_path, 'rb').read()).decode('ascii') 
        encoded_image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii') 
         
        payload = json.loads(self.global_payload) 
    
        payload["messages"].append({  
            "role": "user",  
            "content": [
                {
                    "type": "image_url",
                    "image_url":
                        {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                },
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        })
#         payload["messages"].append({  
        
#             "role": "assistant",  
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "好的，我已经接收到这次相应的原图，请您再传入原图对应的标有每个控件id号的图片(控件的上方标注着id号)。"
#                 }
#             ]
#         })
#         payload["messages"].append({ 
#             "role": "user",  
#             "content": [
#                 {
#                     "type": "image_url",
#                     "image_url":
#                         {
#                             "url": f"data:image/png;base64,{encoded_id_image}"
#                         }
#                 },
#                 {
#                     "type": "text",
#                     "text": user_input + "。请你根据任务内容和此次输入的两张图片，给我一个规范的操作命令。注意，点击操作：A：tap_action: id=<SOI>xxx<EOI>；\
# 输入内容操作： A：input_action: id=<SOI>xxx<EOI>，输入（“yyy”）并回车。xxx应为已提供的控件id，是一个整数，yyy为应该在输入栏内输入的文字信息。"
#                 }
#             ]
#         })  
    
        # 将payload转换为JSON字符串  
        self.global_payload = json.dumps(payload)  
        
    def add_user_textinput(self, user_input):
        payload = json.loads(self.global_payload) 
    
        payload["messages"].append({  
            "role": "user",  
            "content": [
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        })  
    
        # 将payload转换为JSON字符串  
        self.global_payload = json.dumps(payload) 
            
    def add_LLM_output(self, model_response):  
        payload = json.loads(self.global_payload)

        # 将模型响应添加到payload中  
        payload["messages"].append({  
            "role": "assistant",  
            "content": model_response  
        })  
    
        self.global_payload = json.dumps(payload)  

GPT4V_history = LLMHistory_GPt4V(language, gpt_choice)