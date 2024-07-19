import json
import cv2
from skimage.metrics import structural_similarity as ssim
import shutil
import pandas as pd

from core.process_img_script import process_img
from core.command_operator import *
from core.LLm_history import Global_LLm_history, language, gpt_choice
from core.screenshot_translator import ScreenshotTranslator
from core.LLM_api import use_LLM
from core.help_seq_getter import help_get_flag, help_seq_get
from core.Config import *
import logging
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)

taskdf = pd.read_excel(TaskTable_PATH, keep_default_na=False)
input_ProcessImgModel = True
if input_ProcessImgModel:
    # 导入模型 开机仅一次即可
    import core.import_models as import_models
    model_ver, model_det, model_cls, preprocess, ocr = import_models.import_all_models \
        (alg,
            # model_path_yolo='pt_model/yolo_s_best.pt',
            accurate_ocr = accurate_ocr,
            model_path_yolo='pt_model/yolo_mdl.pt',
            model_path_vins_dir='pt_model/yolo_vins_',
            model_ver='14',
            model_path_vins_file='_mdl.pt',
            model_path_cls='pt_model/clip_mdl.pth'
            )


def are_images_similar(save_path_old, save_path_new, threshold=0.90, threshold_roi=0.97, roi=None): #图片相似度阈值 
    
    img1 = cv2.imread(save_path_old)# 读取图片
    img2 = cv2.imread(save_path_new)

    # Check if dimensions are different
    if img1.shape != img2.shape:
        shutil.copy(save_path_new, save_path_old)
        return False  # 表示图片更新了
    
    if roi is not None:
        x, y, w, h = roi
        img1_roi = img1[y:y + h, x:x + w]
        img2_roi = img2[y:y + h, x:x + w]
        gray_img1_roi = cv2.cvtColor(img1_roi, cv2.COLOR_BGR2GRAY)
        gray_img2_roi = cv2.cvtColor(img2_roi, cv2.COLOR_BGR2GRAY)
        similarity_index_roi, _ = ssim(gray_img1_roi, gray_img2_roi, full=True)
    else:
        similarity_index_roi = 1

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  
    similarity_index, _ = ssim(gray_img1, gray_img2, full=True)
    shutil.copy(save_path_new, save_path_old)
    return similarity_index > threshold and similarity_index_roi > threshold_roi

def eleLoc_2_roi(element_location): #element_location = {'left': 42, 'right': 838, 'top': 2036, 'bottom': 2185}
    roi = (
        element_location['left'] - 10,
        element_location['top'] - 10 ,
        element_location['right'] - element_location['left'] + 30,
        element_location['bottom'] - element_location['top'] + 30
    )
    return roi

def client_main():
    
    # 初始化准备
    global task_str
    global tapped_button_str
    global invalid_button_str
    global taped_element_content
    global taped_element_roi
    global default_reason
    
    shutil.copy(save_path_default, save_path_old)
    first_flag = True
    task_str = ""
    default_reason = ""
    invalid_button_str = [""]
    taped_element_content = ""
    taped_element_roi = None
    tapped_button_str = ["，现在已经进行了的操作有"]


    print("\n*--------------------------------- START --------------------------------------*")

    message = "Q：" + input("🥰: Hi, I'm VisionTasker. What can I do for you~")  # Go to Alipay to analyze my spending in annual

    # message = "Q：" + message
    if message.startswith("Q"):  # 处理以 "任务为:" 开头的消息
        task_str = str(message)
        task_str = task_str.split('Q：')[-1]
        print("👂 User:", task_str)
        response = "🧐: OK, let me help you 👌"
        print(response)

        # 若是再次输入m，则重置此次任务的所有信息
        if len(tapped_button_str) > 1:
            print("Enter the task content again, and the operation record is available, reset the task, please return to the initial interface")
            shutil.copy(save_path_default, save_path_old)
            default_reason = ""
            invalid_button_str = [""]
            taped_element_content = ""
            taped_element_roi = None
            tapped_button_str = ["，现在已经进行了的操作有"]
            Global_LLm_history.__init__(language, gpt_choice)
            # 根据任务内容获取帮助文档信息
        if not help_get_flag:
            help_str = ''
        else:
            help_question, help_content = help_seq_get(task_str)
            help_str = f"此次任务的帮助文档信息：{help_question} {help_content}" if help_question and help_content else ''
            print(help_str)

    else:
        # 处理其他消息
        print("Received a different message")

    
    try:
        while True:
            time.sleep(5)
            if not first_flag:
                print('🧐: Move on!')
            first_flag = False
            print("\n*---------------------- SCREEN UNDERSTANDING ---------------------------*")
            # 截图 or 长截图
            if longscreenshot_flag:
                capture_longscreenshot(save_path_new)
            else:
                capture_screenshot(save_path_new)

            print('🧐: Hmmm, I can watch your screen now...')
            # 根据图片是否更新，判断上次点击的控件的可点击性（没更新就是不可点击）
            screenshot_update_flag = not(are_images_similar(save_path_old, save_path_new, roi = taped_element_roi))
            if (not screenshot_update_flag) and taped_element_content != '' and not default_reason: # 图片没更新 + 不是在发送了m之后 + 不是过程中出现了 界面不存在该元素或回答格式不对
                tapped_button_str.pop()  # 删去上次点击的控件信息
                # 删去llm上次的input和output信息
                Global_LLm_history.clear_previous_one_record()
                Global_LLm_history.clear_previous_one_record()

                invalid_button_str.append(f"['{taped_element_content}']")  # 记录这个无效的button
            else:
                invalid_button_str = ['']  # 重置无效控件列表
            print('🧐: OK, let me take a closer look......')
            result_js = process_img(label_path_dir, save_path_old, output_root, layout_json_dir, high_conf_flag,
                        alg, clean_save, plot_show, ocr_save_flag, model_ver, model_det, 
                        model_cls, preprocess, pd_free_ocr=ocr, ocr_only=ocr_output_only, lang=language, accurate_ocr=accurate_ocr)


            # 存储json文件
            json.dump(result_js, open(SCREEN_jSON_PATH, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

            # 将屏幕信息json文件 转换成 易阅读的字符串
            print('🧐: Haha, time for LLM!')
            print("\n*--------------------------- TASK PLANNING -----------------------------*")

            ScreenshotTranslatorTest = ScreenshotTranslator(SCREEN_jSON_PATH)
            humanword = ScreenshotTranslatorTest.json2humanword()

            # 提示llm的注意事项
            if len(invalid_button_str) == 1:  # =1代表初始，即没有无效控件
                invalid_button_string = '。'
                if default_reason.startswith("界面中不存在"):
                    invalid_button_string = f"。注意{default_reason}。"
                    default_reason = ""
            else: #存在无效控件，即不可点击的控件
                invalid_button_string = ''.join(map(str, invalid_button_str)) #无效控件提醒
                invalid_button_string = f"。注意控件{invalid_button_string}不可点击。"
                if default_reason.startswith("界面中不存在"):
                    invalid_button_string = f"。注意控件{invalid_button_string}不可点击,{default_reason}。"
                    default_reason = ""

            # 整合要送入llm的信息
            if len(tapped_button_str) == 1: # =1代表初始，即没有点击过控件
                object_message_humanword = f'Q：{task_str}，{help_str}当前界面有以下按钮：{humanword}{invalid_button_string}'
            else:
                tapped_button_string = ''.join(map(str, tapped_button_str))
                object_message_humanword = f'Q：{task_str}{tapped_button_string}，{help_str}当前界面有以下按钮：{humanword}{invalid_button_string}'
            print(object_message_humanword)

            object_message_humanword = json.dumps(object_message_humanword)
            order_list = use_LLM(gpt_choice, object_message_humanword)


            for order in order_list:
                if order['action'] != 'default':  # llm 返回有效命令
                    # 若此次点击的控件与上次点击控件不同，此次有效
                    if taped_element_content != order['button_content']:
                        taped_element_content = order['button_content'] # 点击的控件内容
                        taped_element_roi = eleLoc_2_roi(order['element_location']) # 点击的位置

                        # 记录操作历史
                        if order['action'] == 'keyboard_input':
                            input_text_content = order['data']['input_text']
                            tapped_button_str.append(f'（在输入框:[{repr(taped_element_content)}]输入了{repr(input_text_content)}并回车）')
                        elif order['action'] == 'tap':
                            tapped_button_str.append(f'（点击[{repr(taped_element_content)}]）')
                else:
                    default_reason = order['reason']
                    print("Please send the screenshot again and GPT will regenerate the operation command")

            response = json.dumps(order_list, ensure_ascii=False)  # 转换order_list为JSON格式

            print(f"Execute: {response}")  # 处理消息并生成回复
            response = eval(response)


            if isinstance(response, str):
                # print("response 是字符串类型")
                pass
            elif isinstance(response, list):
                order_list = response
                operator(order_list)  # 传递服务器返回的order_list中的每个操作
                print("\n*---------------------------- √ ONE STEP COMPLETED ----------------------------*\n\n\n")

            else:
                print("response 是其他类型")
                
        
    finally:
        print(f"client_main发生异常")


if __name__ == "__main__":

    client_main()
