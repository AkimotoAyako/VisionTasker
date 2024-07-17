# 配置文件

# 选项
high_conf_flag = False  # 是否使用高支持度阈值 对于系统应用可以提升加快速度
clean_save = False  # 是否只按照路径要求输出layout的json
plot_show = False  # 是否显示图片
if clean_save:
    plot_show = False
ocr_save_flag = 'save'  # ocr省钱模式 用于反复调整时 直接使用已保存的ocr结果 部分文件支持
ocr_output_only = False  # 新增：只输出ocr结果 不要所有ip
alg = 'yolo'  # yolo / detr / vins 三种算法
accurate_ocr = False  # 是否使用高精度版OCR
lang = 'zh'  # en / zh  # 输出语言选择
workflow_only = False     # 只输出json和整体流程图
ocr = None
gpt4_mode = True  # 使用gpt4v模式时不导入clip模型 我还不会把参数从随便一个文件传进来所以你自己在这改
gpt4_text_size = 0.35 # 给gpt4v的带控件id图片 标注字体大小
gpt4_text_thickness = 2  # 给gpt4v的带控件id图片 标注字体粗细 整数类型

# 路径
    # label_path_dir = 'D:/UI_datasets/other_codes/GUI-Detection-Grouping/clip_labels/'  # 176类分类注释文件存放地址
    # output_root = 'D:/UI_datasets/other_codes/GUI-Detection-Grouping/outputs/'  # 如果不是clean输出模式 完整版的输出文件存放的文件夹
    # layout_json_dir = 'D:/UI_datasets/other_codes/GUI-Detection-Grouping/outputs/clean_json'  # clean模式下 最终识别结果json输出的文件夹
label_path_dir = 'pt_model/clip_labels/'  # 176类分类注释文件存放地址
# img_path = '/data4/bianyiheng/task1/0.png'  # 检测图片路径
save_path_old = 'data/screenshot/screenshot_old.png'
save_path_new = 'data/screenshot/screenshot_new.png'
save_path_default = 'data/screenshot/screenshot_default.png'
output_root = 'data/outputs/'  # 如果不是clean输出模式 完整版的输出文件存放的文件夹
layout_json_dir = 'clean_json'  # clean模式下 最终识别结果json输出的文件夹
# SCREEN_jSON_PATH = "data/screenshot/screenshot.json" #截图信息的json文件存放路径
# XML_file_PATH = "data/screenshot/"#截图信息的XML文件存放路径(测试CHI的方法)
img_4gpt4_out_path = 'data/screenshot/screenshot_gpt4.png'  # 给gpt4的图片输出路径 仅给该文件夹下的脚本修改了该用法
SCREEN_jSON_PATH = "data/screenshot/screenshot_gpt4.json"
# IMAGE_PATH = "data/screenshot/screenshot_gpt4.png"
LLM_history_PATH = "core_gpt4v/LLM_history"
TaskTable_PATH = "data/task.xlsx"


    
    
# 手机参数
screenshot_height = 2244
screenshot_width = 1080

choose_phone = "真机huawei"  #"真机huawei"/"pad"/"真机redmi"
if choose_phone == "真机huawei":
    DeviceName = "CLB7N18719001673" #USB连接
    DefaultInputKeyboard = "com.huawei.ohos.inputmethod/com.android.inputmethod.latin.LatinIME"
elif choose_phone == "pad":
    DeviceName = "ALNYUN3829G02254" #USB连接
    DefaultInputKeyboard = "com.baidu.input_hihonor/com.baidu.input_honor.ImeService"
elif choose_phone == "真机redmi":
    DeviceName = "2d6e3364" #USB连接
    DefaultInputKeyboard = "com.huawei.ohos.inputmethod/com.android.inputmethod.latin.LatinIME"