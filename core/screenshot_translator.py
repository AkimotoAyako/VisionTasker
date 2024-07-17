import json
import os
from io import StringIO

class ScreenshotTranslator: 
    def __init__(self, file_path='screenshot/screenshot.json', output_text_file_path='screenshot/screenshot.txt') -> None:
        self.jsondata = self.get_screenshot_jsondata(file_path)
        self.output_humanword = StringIO()
        self.output_text_file_path = output_text_file_path
        
    def get_screenshot_jsondata(self, file_path):
        if os.path.exists(file_path):  
            with open(file_path, 'r', encoding='utf-8') as f:  
                data = f.read()  
                object_message_json_data = json.loads(data)  # 解析JSON数据  
        else:  
            print(f"文件{file_path}不存在") 
            
        return object_message_json_data
    
    def save_output_humanword_to_file(self, file_path):
        output_text = self.output_humanword.getvalue()
        with open(file_path, 'w') as file:
            file.write(output_text)

    def alignment_judge(self, element): # 判断当前区域的排列方式  "alignment: v"
        alignment_parts = element.split(":")
        if len(alignment_parts) == 2:
            key, value = alignment_parts
            key = key.strip()  # 去除两边的空格
            value = value.strip()  # 去除两边的空格
            if value == 'h':
                value = 'horizontally'
            elif value == 'v' or value == 'n':
                value = 'vertically'
            return value
        else:
            return "Invalid alignment format"

    def process_element(self, element, level = 0, line_number = 1, alignment = 'vertically', block_convert_flag = False, child_flag = False):
        if 'id' in element and 'class' in element:
            element_id = element['id']
            element_class = element['class']
            
            child_text = repr(element.get('text_content', None))

            position_info = f"\n第{line_number}列" if alignment == 'horizontally' else f"\n第{line_number}行"
            line_info = f"{position_info} - "
            indentation = "    " * level
            
            # output_str = f"{indentation}{line_info}id: {element_id}, Class: {element_class}, Text: {child_text}\n"
            if element_id.startswith('l') :
                c_alignment = self.alignment_judge('alignment:' + element['list_alignment'])
                align_str = "从左到右" if c_alignment == 'horizontally' else "从上到下"
                # output_str = f"{indentation}{line_info} 这是一个列表，{align_str}为以下按钮:\n"
                output_str = f"{indentation}{line_info}  {align_str}为以下控件:\n"

            elif element_id.startswith('b'):          
                if block_convert_flag:
                    c_alignment = 'horizontally' if alignment=='vertically' else 'vertically'
                else:
                    c_alignment = alignment
                align_str = "从左到右" if c_alignment == 'horizontally' else "从上到下"

                # output_str = f"{indentation}{line_info} 这是一个块，{align_str}为以下按钮:\n"
                output_str = f"{indentation}{line_info} {align_str}为以下控件:\n"

            else :
                # output_str = f"{indentation}{line_info} [{element_id.split('-')[-1]}-{child_text}]"
                # output_str = f"{indentation}{line_info} [{repr(child_text)}]"
                output_str = f"{indentation}{line_info} [{child_text}]"
                if element.get("sub_class") == 'edittext' or element.get("sub_class") == 'autocompletetextview':
                    output_str = f"{indentation}{line_info}输入框: [{child_text}]"

            #处理2级节点
            if child_flag : 
                indentation = " "
                if element_id.startswith('b') or element_id.startswith('l'):
                    output_str = ''
                else:
                    # output_str = f"{indentation}[{element_id.split('-')[-1]}-{child_text}]\n"
                    # output_str = f"{indentation}[{repr(child_text)}]"
                    output_str = f"{indentation}[{child_text}]；"

                    if element.get("sub_class") == 'edittext' or  element.get("sub_class") == "autocompletetextview":
                        output_str = f"{indentation}输入框: [{child_text}]；"


            self.output_humanword.write(output_str)

            # Block or compos or text  处理子项
            if 'children' in element: 
                # for line_index, child in enumerate(element['children']):
                #         line_number = process_element(child, level + 1, line_index + 1, output_buffer)
                if element_class == 'Block': # 如果是Block 那么子块的布局会更换方向
                    if block_convert_flag:
                        c_alignment = 'horizontally' if alignment=='vertically' else 'vertically'
                    else:
                        c_alignment = alignment
                        
                    for line_index, child in enumerate(element['children']):
                        line_number = self.process_element(child, level + 1, line_index + 1, alignment=c_alignment, child_flag = True)

                else:  # 如果是compos or text
                    for line_index, child in enumerate(element['children']):
                        line_number = self.process_element(child, level + 1, line_index + 1,  alignment= alignment, child_flag = True)

            # List  处理子项
            if 'list_items' in element: 
                list_items = element['list_items']
                # list_alignment = 'list_alignment: ' + element['list_alignment']
                # list_alignment_str = f"{indentation}   - {alignment_judge(list_alignment)}\n"

                c_alignment = self.alignment_judge('alignment:' + element['list_alignment'])
                for line_index, list_item in enumerate(list_items):
                    for list_item_element in list_item:
                        line_number = self.process_element(list_item_element, level + 1, line_index + 1, alignment=c_alignment, child_flag = True)           
                # if output_buffer:
                #     # output_buffer.write(list_alignment_str)

        return line_number

    def json2humanword(self, imgname = "screenshot"):
        for line_index, element in enumerate(self.jsondata):
            if isinstance(element, str): # json中第一个元素为string，如"alignment: v"，用于描述排列方式
                alignment = self.alignment_judge(element)
            self.process_element(element, level=0, line_number=line_index, alignment=alignment, block_convert_flag = True)

        output_text = self.output_humanword.getvalue()  # 获取存储的输出字符串
        
        # file_folder = "/data4/tangyt/GUI-Detection-Grouping/run_single_test/humanword"
        # os.mkdir(file_folder) if not os.path.exists(file_folder) else None
        # file_path = f'{file_folder}/{imgname}.txt'
        # save_output_buffer_to_file(output_text, file_path)
        # self.save_output_humanword_to_file(self.output_text_file_path)
        self.output_humanword.close()  # 关闭StringIO对象

        return output_text

if __name__ == '__main__':
    output_filename = 'screenshot/screenshot.json'  # 读取JSON文件

    ScreenshotTranslatorTest = ScreenshotTranslator(output_filename)
    output_text = ScreenshotTranslatorTest.json2humanword()
    print(output_text)

