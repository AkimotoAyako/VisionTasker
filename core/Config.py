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
accurate_ocr = False  # 是否使用高精度版OCR, 低精度的是开源paddleocr，高精度的是调用paddle的ocr api
lang = 'zh'  # en / zh  # 输出语言选择
workflow_only = False     # 只输出json和整体流程图

# 路径

label_path_dir = 'pt_model/clip_labels/'  # 176类分类注释文件存放地址
# img_path = '/data4/bianyiheng/task1/0.png'  # 检测图片路径
save_path_old = 'data/screenshot/screenshot_old.png'
save_path_new = 'data/screenshot/screenshot_new.png'
save_path_default = 'data/screenshot/screenshot_default.png'
output_root = 'data/outputs/'  # 如果不是clean输出模式 完整版的输出文件存放的文件夹
layout_json_dir = 'clean_json'  # clean模式下 最终识别结果json输出的文件夹
SCREEN_jSON_PATH = "data/screenshot/screenshot.json" #截图信息的json文件存放路径
XML_file_PATH = "data/screenshot/"#截图信息的XML文件存放路径(测试CHI的方法)
img_4gpt4_out_path = 'data/screenshot/screenshot_gpt4.png'  # 给gpt4的图片输出路径 仅给该文件夹下的脚本修改了该用法
LLM_history_PATH = "data/LLM_history"
TaskTable_PATH = "data/task.xlsx"

# adb
choose_phone = "真机huawei"  #"真机huawei"/"pad"/"真机redmi"
if choose_phone == "真机huawei":
    DeviceName = "CLB7N18719001673" #USB连接 cmd输入"adb devices"获得DeviceName
    DefaultInputKeyboard = "com.huawei.ohos.inputmethod/com.android.inputmethod.latin.LatinIME"
elif choose_phone == "pad":
    DeviceName = "ALNYUN3829G02254" #USB连接
    DefaultInputKeyboard = "com.baidu.input_hihonor/com.baidu.input_honor.ImeService"
elif choose_phone == "真机redmi":
    DeviceName = "2d6e3364" #USB连接
    DefaultInputKeyboard = "com.huawei.ohos.inputmethod/com.android.inputmethod.latin.LatinIME"
ADBKeyboard = "com.android.adbkeyboard/.AdbIME"
# 键盘输入需要安装ADBKeyboard.apk(网上找)
# 输入法设置： 

# 1.获取当前设备有效的输入法列表
# adb shell ime list -s
# 注意： -s并不是已安装的所有输入法，而是安装并已勾选的输入法。
# 启用方式：
# 系统设置>>通用>>语言和输入法>>勾选输入法

# 2.检查可用的全部虚拟键盘：
# adb shell ime list -a

# 3.获取当前设备有效输入法的详细信息
# adb -s ALNYUN3829G02254 shell ime list

# 4.查看当前正在使用的输入法
# 当我们封装切换时，必须先知道这个输入法的默认名称，以下时获取方式：
# adb  -s CLB7N18719001673 shell settings get secure default_input_method