import os
import json  
#选择gpt-api   baidu/  azure_openai / chatglm
gpt_choice = 'baidu'
#选择language   zh / en
language = 'zh'
from core.Config import LLM_history_PATH
class LLmHistory:
    def __init__(self, language, gpt_choice):
        self.language = language
        self.gpt_choice = gpt_choice
        self.global_payload = self.set_preprompt()  # 设置初始值为 store_payload
        
    def save_conversation_to_file(self, task_str = "llm_history", LLM_history_folder = LLM_history_PATH):
        data = json.loads(self.global_payload)
        hisory = data["messages"]
        history_path = LLM_history_folder + task_str + ".json"
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        json.dump(hisory, open(history_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
  
    def set_payload(self, value):
        self.global_payload = value

    def get_payload(self):
        return self.global_payload
    
    def set_preprompt(self):    
        if self.language == 'zh':
            message_list = [
                #1
                { 
                    "role": "user",
                    "content": """您需要扮演一个智能助手的角色，现在我需要在手机应用上完成指定任务，但不清楚如何操作，\
        告诉我在手机界面上对哪个元素进行点击操作或者输入特定内容以完成该任务。我的提问是Q，您的回答是A。我将执行您给的操作指令并把界面反馈给您（每次1个Q），您再告诉我下一步（每次1个A）。您只需要提供一步的操作命令."""
                },
                {
                    "role": "assistant",
                    "content": "好的，我将扮演智能助手的角色帮助你完成指定任务，每次回答我只提供一步的操作命令。"
                },
                #2
                { 
                    "role": "user",
                    "content": "每次回答我一步，你需要根据任务要求、已经点击过的控件、键盘输入过的内容，\
        谨慎地决定操作指令中的值：xxx, yyy。xxx应为我提供的控件完整内容。如果我为你提供了帮助文档的信息，请您的回答首先以帮助文档为参考。"
                },
                {
                    "role": "assistant",
                    "content": "好的，我每次只回答一步，并且会根据任务要求、已经点击过的控件、键盘输入过的内容，或者你提供的帮助文档信息，\
        谨慎地决定操作指令中的值：xxx, yyy。xxx应为你提供的控件完整内容。如果你提供了帮助文档的信息，我的回答将首先以帮助文档为参考。"
                },
                { 
                    "role": "user",
                    "content": "你需要根据任务要求和当前界面的所有控件信息，注意当前界面可能不是初始界面，可能已经完成了一些操作，你要判断接下来应该进行什么操作，提供一个规范的操作命令。"
                },
                {
                    "role": "assistant",
                    "content": "好的，我一定会根据任务要求和当前界面的所有控件信息，为您提供一个规范的操作命令。"
                },
                #3
                { 
                    "role": "user",
                    "content": "若进行点击操作，则务必用该格式进行回答：A：tap_action: 点击['xxx']。\
        若进行输入内容操作，则在输入框中进行操作，则务必用该格式进行回答： A：input_action: 在输入框:['xxx']输入（“yyy”）并回车。\
        xxx, yyy均为文字，xxx应为已提供的控件完整内容，不要说下一步，不重复你提供的界面信息，不要有多余的解释。"
                },
                {
                    "role": "assistant",
            #         "content": "好的，若进行点击操作，则我会用该格式进行回答：A：tap_action: 点击['xxx']。\
            # 若进行输入内容操作，则在输入框中进行操作，我会用该格式进行回答：A：input_action: 在输入框: ['xxx']输入（“yyy”）并回车。\
            # xxx, yyy均为文字。xxx应为已提供的控件完整内容。"
                    "content": "好的，我明白了两种类型的操作命令，我会按格式输出操作命令。"
                },
                #5
                { 
                    "role": "user",
                    "content": "以下为点击操作的例子：\n\
        Q：我想返回主页面，当前界面如下：\n\
        第1行 - 从左到右为以下按钮:\n\
        ['向右的前进或跳过，播放，或YouTube']； 输入框: ['搜索视频、番剧或up主']； ['搜索']；\n\
        第2行 -  ['ADB Keyboard (ON)']\n \
        A：tap_action: 点击['向右的前进或跳过，播放，或YouTube']。"
                },
                {
                    "role": "assistant",
                    "content": "好的，若进行点击操作，则我会用该格式进行回答：A：tap_action: 点击['xxx']。请再为我提供输入内容操作的示例。"
                },
                #6
                { 
                    "role": "user",
                    "content": "以下为输入内容操作的例子：\n\
        Q：我想看“周杰伦演唱会”，当前界面如下：\n\
        第1行 - 从左到右为以下按钮:\n\
        ['镜像翻转图标键']； 输入框: ['默认文本: chill kill解析']； ['搜索、放大键']； ['si3']； ['邮件、信息符号键']；\n\
        第2行 -  从左到右为以下控件:\n\
        ['标题：7 首页 动态']； ['7']； ['主页、房子或家庭键']； ['翻译，或带字图片键']； ['会员购']； ['我的']； ['加号键']； \
        A：input_action: 在输入框: ['默认文本: chill kill解析']输入（“周杰伦演唱会”）并回车。"
                },
                {
                    "role": "assistant",
                    "content": "好的，若进行输入内容操作，则我会用该格式进行回答：A：input_action: 在输入框: ['xxx']输入（“yyy”）并回车。请问还有别的示例吗？"
                },
                #7
                { 
                    "role": "user",
                    "content": "Q：发送“你好”，当前界面有以下按钮：\n\
        第1行 -  从左到右为以下按钮: ['向右的前进或跳过，播放，或YouTube_']； ['咖啡因从咖啡果中来']； ['省略号、更多或溢出菜单']；\n\
        第2行 -  ['10月22日晚上19：31']\n\
        第3行 -  从左到右为以下按钮: ['音频、客服键']； 输入框: ['默认文本：']； ['笑脸，用于表情包列表、评分良好或喜欢']； ['+']；\n\
        A：input_action: 在输入框: ['默认文本：']输入（“你好”）并回车。"
                },
                {
                    "role": "assistant", 
                    "content": "好的，若进行点击操作，则我会用该格式进行回答：A：tap_action: 点击['xxx']；\
        若进行输入内容操作，则我会用该格式进行回答：A：input_action: 在输入框: ['xxx']输入（“yyy”）并回车。\
        我会注意xxx, yyy均为文字，xxx应为你提供的控件完整内容，且不说下一步，不重述你提供的界面信息，不说多余的解释。"
                },
                #8
                { 
                    "role": "user",
                    # "content": "如果您在界面中没有找到要点击的控件，请回复我：A：无操作；如果您判定当前界面已经完成了任务，请回复我：A：任务已完成。"
                    "content": "您每次只需提供一步的操作命令，只需要一条input_action命令或者tap_action命令，如果您判定当前界面已经完成了任务，请回复我：A：end_action: 任务已完成。"
                },
                {
                    "role": "assistant",
                    # "content": "好的，如果我在界面中没有找到要点击的控件，我将回复：A：无操作；\
                    #     如果我判定当前界面已经完成了任务，我将回复：A：任务已完成。请告诉我您需要完成什么任务？"
                    "content": "好的，我每次只提供一步、一种操作命令，如果我判定当前界面已经完成了任务，我将回复：A：end_action: 任务已完成。请告诉我您需要完成什么任务？"
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

            "model": "gpt-35-turbo-16k",  # gpt-3.5-turbo
            "temperature": 0,  # 0.7
            "max_tokens": 800,
            "top_p": 0,  # 0.95
            "frequency_penalty": 0,
            "presence_penalty": 0,

            "messages": message_list,
            }
            
        store_payload = json.dumps(dict_message)
        return store_payload
    
    def clear_previous_one_record(self):  
        payload = json.loads(self.global_payload) 
        payload["messages"].pop()
        self.global_payload = json.dumps(payload) 
         
    def add_user_input(self, user_input):  
        payload = json.loads(self.global_payload) 
    
        payload["messages"].append({  
            "role": "user",  
            "content": user_input  
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

Global_LLm_history = LLmHistory(language, gpt_choice)