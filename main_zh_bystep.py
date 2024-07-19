import json
import cv2
from skimage.metrics import structural_similarity as ssim
import shutil
import pandas as pd
import time
#库文件
from core.process_img_script import process_img
from core.command_operator import *
from core.LLm_history import Global_LLm_history, language, gpt_choice
from core.screenshot_translator import ScreenshotTranslator
from core.LLM_api import use_LLM
from core.help_seq_getter import help_get_flag, help_seq_get
from core.Config import *
taskdf = pd.read_excel(TaskTable_PATH, keep_default_na = False) 
input_ProcessImgModel = True
if input_ProcessImgModel:
    # 导入模型 开机仅一次即可
    import GPUtil
    gpus = GPUtil.getGPUs()
    now_gpu = gpus[0]
    if now_gpu.memoryFree/now_gpu.memoryTotal > 0.9:
        print('你的装备是：', now_gpu.name, '现在使用的内存：', now_gpu.memoryFree, '/', now_gpu.memoryTotal, '哈哈没人和你抢')
    else:
        print('你的装备是', now_gpu.name, '内存使用情况：', now_gpu.memoryFree, '/', now_gpu.memoryTotal, '好像有别的程序在用 显卡发出尖锐爆鸣')
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

def get_time(f):

    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

def are_images_similar(save_path_old, save_path_new, threshold=0.90, threshold_roi=0.97, roi=None): #图片相似度阈值 
    
    img1 = cv2.imread(save_path_old)# 读取图片
    img2 = cv2.imread(save_path_new)

    # Check if dimensions are different
    if img1.shape != img2.shape:
        print("图片尺寸不同，不是同一张图")
        shutil.copy(save_path_new, save_path_old)
        return False  # 表示图片更新了
    
    if roi is not None:
        x, y, w, h = roi
        img1_roi = img1[y:y + h, x:x + w]
        img2_roi = img2[y:y + h, x:x + w]
        gray_img1_roi = cv2.cvtColor(img1_roi, cv2.COLOR_BGR2GRAY)
        gray_img2_roi = cv2.cvtColor(img2_roi, cv2.COLOR_BGR2GRAY)
        similarity_index_roi, _ = ssim(gray_img1_roi, gray_img2_roi, full=True)
        print(f"局部区域相似度: {similarity_index_roi}")
    else:
        print(f"未指定比较的局部区域")
        similarity_index_roi = 1

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)# 将图片转换为灰度图
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  
    similarity_index, _ = ssim(gray_img1, gray_img2, full=True)# 计算结构相似性指数
    print(f"图片相似度: {similarity_index}")

    if similarity_index > threshold and similarity_index_roi > threshold_roi: # 判断相似度是否大于阈值
        print(f"图片相似度大于{threshold}，局部区域相似度大于{threshold_roi}，是同一张图")
    else:
        print(f"图片相似度不足{threshold}，或局部区域相似度不足{threshold_roi}，不是同一张图")
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
    task_str = ""
    default_reason = ""
    invalid_button_str = [""]
    taped_element_content = ""
    taped_element_roi = None
    tapped_button_str = ["，现在已经进行了的操作有"]
    
    try:
        while True:
            command = input("发送消息或图片 (m/i): ") 
            
            #发送任务信息
            if command.startswith('m'):
                #-----------------begin: input task content directly--------------
                message = "Q：" + input("你好，我是VisionTasker。请输入任务内容：") 
                # 例如：找到西安交通大学的官方微博并关注
                #-----------------end: input task content directly--------------

                #-----------------begin: use task.xlsx--------------
                # tasklist = taskdf['任务内容'].tolist()  # 任务列表
                # inputarray = command.split(" ")
                # if len(inputarray) == 1:
                #     num = input("请输入任务序号：(序号需要大于3)")
                # else:
                #     num = inputarray[1]
                # while not num.isdigit() or tasklist[int(num) - 2] == '' :
                #     num = input("请输入有效的任务序号：(序号需要大于3)")
                # message = "Q：" + tasklist[int(num) - 2]  
                #-----------------end: use task.xlsx--------------
                if message.startswith("Q"):          
                    task_str = str(message)
                    task_str = task_str.split('Q：')[-1]
                    response = "已收到任务内容：" + task_str
                    print(response)
                    
                    # 若是再次输入m，则重置此次任务的所有信息
                    if len(tapped_button_str) > 1:
                        print("再次输入任务内容，且有了操作记录，重置此次任务，请回到初始界面")
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
                    
            #发送截图指示
            elif command.startswith('i'):
                start_time = time.time()
                #截图 or 长截图
                if longscreenshot_flag:
                    capture_longscreenshot(save_path_new)
                else:
                    capture_screenshot(save_path_new)


                # 根据图片是否更新，判断上次点击的控件的可点击性（没更新就是不可点击）
                screenshot_update_flag = not(are_images_similar(save_path_old, save_path_new, roi = taped_element_roi))  
                if (not screenshot_update_flag) and taped_element_content != '' and not default_reason: # 图片没更新 + 不是在发送了m之后 + 不是过程中出现了 界面不存在该元素或回答格式不对
                    tapped_button_str.pop() # 删去上次点击的控件信息
                    # 删去llm上次的input和output信息
                    Global_LLm_history.clear_previous_one_record()
                    Global_LLm_history.clear_previous_one_record()

                    invalid_button_str.append(f"['{taped_element_content}']") # 记录这个无效的button
                else:
                    invalid_button_str = [''] #重置无效控件列表

                result_js = process_img(label_path_dir, save_path_old, output_root, layout_json_dir, high_conf_flag,
                            alg, clean_save, plot_show, ocr_save_flag, model_ver, model_det, model_cls, preprocess, 
                             pd_free_ocr=ocr, ocr_only=ocr_output_only)

                
                # 存储json文件
                json.dump(result_js, open(SCREEN_jSON_PATH, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
                
                # 将屏幕信息json文件 转换成 易阅读的字符串
                ScreenshotTranslatorTest = ScreenshotTranslator(SCREEN_jSON_PATH)
                humanword = ScreenshotTranslatorTest.json2humanword()
                end_time = time.time()
                print("（截图 + 控件检测分组 + 转为自然语言描述） 耗时: {:.2f}秒".format(end_time - start_time))           
                
                # 提示llm的注意事项
                if len(invalid_button_str) == 1: # =1代表初始，即没有无效控件
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
                print("\n" + object_message_humanword)

                object_message_humanword = json.dumps(object_message_humanword)

                # 使用大语言模型生成json格式的操作命令, 创建一个线程来调用 connect_gpt 函数

                order_list = use_LLM(gpt_choice, object_message_humanword)

                for order in order_list:
                    if order['action'] != 'default': # llm 返回有效命令
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
                        print("请再发送一次截图，GPT将重新生成操作命令")

                response = json.dumps(order_list, ensure_ascii=False)  # 转换order_list为JSON格式
              
                print(f"回复客户端消息: {response}")  # 处理消息并生成回复
                response = eval(response)
                  
             
            else:
                print("无效的命令")
                continue
            

            # 执行命令
            if isinstance(response, list):
                start_time = time.time()
                order_list = response
                operator(order_list)  # 传递服务器返回的order_list中的每个操作
                end_time = time.time()
                print("（LLM获取输出后到执行动作） 耗时: {:.2f}秒".format(end_time - start_time))
                print("*---------------------------执行命令完毕-----------------------------------*\n") 
            else:
                print("response 是其他类型")
                
        
    finally:
        print(f"client_main发生异常")


if __name__ == "__main__":

    client_main()
