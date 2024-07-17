import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import json
import re

from core_gpt4v.Config import SCREEN_jSON_PATH, screenshot_height

class CommandProcessor: #LLM to Executor
    def __init__(self, command, json_file_path ):
        self.data = self.get_screenshot_jsondata(json_file_path)
        self.command = command
        
    def get_screenshot_jsondata(self, file_path):
        if os.path.exists(file_path):  
            with open(file_path, 'r', encoding='utf-8') as f:  
                data = f.read()  
                object_message_json_data = json.loads(data)  # 解析JSON数据  
        else:  
            print(f"文件{file_path}不存在") 
            
        return object_message_json_data
    
    def find_element_with_id(self, data, target_id): #根据id查找位置
        with open(SCREEN_jSON_PATH, 'r', encoding= 'utf-8') as json_file:
            data = json.load(json_file)
            json_r = screenshot_height/800
            for compo in data['compos']:
                if compo['id'] == int(target_id):
                    position = [
                        int(compo['position']['row_min'] * json_r),
                        int(compo['position']['row_max'] * json_r),
                        int(compo['position']['column_min'] * json_r),
                        int(compo['position']['column_max'] * json_r)
                    ]
                    return position # position：list
        
        # 如果未找到匹配的元素，返回False
        return False

    def calculate_center(self, location): #计算区域中心点坐标
        # status_bar_height = 0
        # center_x = int((location['left'] + location['right']) / 2  )
        # center_y = int((location['top'] + location['bottom']) / 2  - status_bar_height)
        # return center_x, center_y
        status_bar_height = 0
        center_x = int((location[2] + location[3]) / 2  )
        center_y = int((location[0] + location[1]) / 2  - status_bar_height)
        return [center_x, center_y]
    
    def check_llmcommand_format(self, command): #检查llm的命令格式
        """
        检查llm的命令格式
        
        Parameters:
        param1 - llm给的命令
        
        Returns:
        命令类型
        格式化后的命令
        
        """
        command_type = None
        formated_command = command
        
        command = str(command).replace("\n", "")
        # match = re.search(r'A[:：]\s*(\w+)_action[:：]id=<SOI>(.*?)<EOI>(?:[。]|$)', command)
        match = re.search(r'A[:：]\s*(\w+)_action[:：](.*?)(?:[。]|$)', command)
        if match:
            command_type = match.group(1)
            if command_type != "end":
                first_instruction = f"A：{command_type}_action：{match.group(2)}"
                print("提取出的指令: ", first_instruction)
                formated_command = first_instruction 
            else:
                print("提取出的指令: A：end_action：任务已完成")
        else:
            print("没有按照格式要求正确回答操作类型")
        return command_type, formated_command
   
    def match_component_content(self, command_type, formated_command): #正则匹配出命令中的控件内容
        compo_id = None
        # if  command_type == "input":
        #     component_quote_patterns = [r"输入框:\[\'(.*?)\'\]", r"输入框: \[\'(.*?)\'\]" , r"输入框:\'(.*?)\'", r"输入框\[\'(.*?)\'\]"]
        # if  command_type == "tap":
        #     component_quote_patterns = [r"\[\'(.*?)\'\]", r"\[\'(.*?)\_\'\]", r"\[(.*?)\]", r"\['(.*?)\]", r"\「(.*?)\」",  r"\‘(.*?)\’", r'\“(.*?)\”', r'\【(.*?)\】', r"点击'(.*?)'", r"点击(.*?)，", r"点击(.*?)。"]
        component_quote_patterns =  [r"id=<SOI>(.*?)<EOI>"]
        for pattern in component_quote_patterns:
            compo_match = re.search(pattern, str(formated_command))
            if compo_match:
                break  # 一旦找到匹配，终止循环

        if compo_match :  
            compo_id = compo_match.group(1)  
            if isinstance(int(compo_id), int):
                print(f"控件id是: {compo_id}") 
            else:
                print("检测到非法id格式，此控件id不是int类型的整数") 
        else:
            print("未正则匹配到gpt回答的ID") 

        return compo_id

    def match_input_textmsg(self, command_type, formated_command): #正则匹配出input命令中的待输入文本内容
        text_content = None
        if command_type == "input":
            input_text_patterns = [r"输入（“(.*?)”）", r"输入“(.*?)”", r"输入（'(.*?)'）"]
            for pattern in input_text_patterns:
                text_match = re.search(pattern, str(formated_command))
                if text_match:
                    break  # 一旦找到匹配，终止循环

            if text_match :     
                text_content = text_match.group(1) 
                print(f"input_text_content: {text_content}")
            else:
                print("gpt判定为input_action，但gpt未回答Input_text")
        
        return text_content
                
    def find_compo_center(self, compo_id):  # 找出控件中心点
        tapcenter_x, tapcenter_y = None, None  
        target_id = compo_id # 要查找的元素的ID 

        target_id_list = [f"{target_id}"]
        for target_id in target_id_list:
            element_location = self.find_element_with_id(self.data, target_id)   # 查找目标元素的位置, 文字内容  
            if element_location:  
                # print(f"Element {target_id} 的位置信息为: {element_location}")  
                # print(f"Element {target_id} 的文字内容信息为: {repr(element_text_content)}\n") 
                tapcenter_x, tapcenter_y = self.calculate_center(element_location)
                break  
        return tapcenter_x, tapcenter_y, element_location
    
    def generate_exe_command(self):  # 产生执行器所需命令
        command = self.command
        command_type, formated_command = self.check_llmcommand_format(command)
        if command_type == "end":
            # print("提取出的指令: A：end_action：任务已完成")
            order_list = [{'action': 'default', 'reason': '任务已完成', 'command': command}]  
            return order_list      
        if command_type == None:
            # print("没有按照格式要求正确回答操作类型")
            order_list = [{'action': 'default', 'reason': '没有按照格式要求正确回答操作类型', 'command': command}]  
            return order_list
 
        compo_id = self.match_component_content(command_type, formated_command)
        if compo_id == None:
            order_list = [{'action': 'default', 'reason': '未正则匹配到gpt回答的ID','command': formated_command}]  
            return order_list 
 
        tapcenter_x, tapcenter_y, element_location = self.find_compo_center(compo_id)
        if not tapcenter_x:
            print(f"界面中不存在id为{compo_id}的控件")
            order_list = [{'action': 'default', 'reason': f"界面中不存在id为{compo_id}的控件",'command': formated_command}] 
            return order_list 
        
        if command_type == "tap": 
            action_data = {'action': 'tap', 'data': {'tap_point': [tapcenter_x, tapcenter_y]}, 'button_content': compo_id, 'element_location': element_location, 'command': formated_command} 
            
        if command_type == "input":
            text_content = self.match_input_textmsg(command_type, formated_command)
            if text_content: # 如果有输入内容
                action_data = {'action': 'keyboard_input', 'data': {'tap_point': [tapcenter_x, tapcenter_y], 'input_text': text_content}, 'button_content': compo_id, 'element_location': element_location, 'command': formated_command} 
        order_list = [action_data]  
    
        # print(order_list)  
        return order_list

if __name__ == '__main__':

    command = "A：input_action：id=<SOI>1<EOI> 输入（'电池管理'）并回车。"
    print(command)
    
    CommandProcessorTest = CommandProcessor(command, SCREEN_jSON_PATH)
    order_list = CommandProcessorTest.generate_exe_command()
    print(order_list[0])