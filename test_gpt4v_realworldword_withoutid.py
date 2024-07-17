import json
import cv2
from skimage.metrics import structural_similarity as ssim
import shutil
import GPUtil
import pandas as pd
#库文件
from core.process_img_4gpt4_script import process_img_4gpt4_script
from core_gpt4v.command_operator import capture_longscreenshot, capture_screenshot, longscreenshot_flag, operator
from core_gpt4v.LLm_history import GPT4V_history, language, gpt_choice
from core_gpt4v.LLM_api import use_LLM
# from core_gpt4v.help_seq_getter import help_get_flag, help_seq_get
from core_gpt4v.Config import *

# input_ProcessImgModel = True
# if input_ProcessImgModel:
#     # 导入模型 开机仅一次即可
#     import GPUtil
#     gpus = GPUtil.getGPUs()
#     now_gpu = gpus[0]
#     if now_gpu.memoryFree/now_gpu.memoryTotal > 0.9:
#         print('你的装备是：', now_gpu.name, '现在使用的内存：', now_gpu.memoryFree, '/', now_gpu.memoryTotal, '哈哈没人和你抢')
#     else:
#         print('你的装备是', now_gpu.name, '内存使用情况：', now_gpu.memoryFree, '/', now_gpu.memoryTotal, '好像有别的程序在用 显卡发出尖锐爆鸣')
#     import core.import_models as import_models
#     # model_ver, model_det, model_cls, preprocess = import_models.import_all_models \
#     #     (alg,
#     #         # model_path_yolo='pt_model/yolo_s_best.pt',
#     #         model_path_yolo='pt_model/yolo_mdl.pt',
#     #         model_path_vins_dir='pt_model/yolo_vins_',
#     #         model_ver='14',
#     #         model_path_vins_file='_mdl.pt',
#     #         model_path_cls='pt_model/clip_mdl.pth',
#     #         gpt4v_mode=gpt4_mode
#     #         )
#     print(gpt4_mode)
#     model_ver, model_det, model_cls, preprocess, ocr = import_models.import_all_models \
#         (alg, accurate_ocr=accurate_ocr,
#             # model_path_yolo='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/yolo_s_best.pt',
#             model_path_yolo='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/yolo_mdl.pt',
#             model_path_vins_dir='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/yolo_vins_',
#             model_ver='14',
#             model_path_vins_file='_mdl.pt',
#             model_path_cls='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/clip_mdl.pth',
#             gpt4v_mode=gpt4_mode
#             )

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
        element_location[2] - 10,
        element_location[0] - 10 ,
        element_location[3] - element_location[2] + 30,
        element_location[1] - element_location[0] + 30
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
            
            #发送任务信息   “m xx”  m + 空格（一个或多个）+ 任务在表中的序号
            if command.startswith('m'):
                #region
                # --------------影音视听---------------
                # message = "我想看去哔哩哔哩看“守护解放西4”" # b站
                # message = "去哔哩哔哩购买邀请码" # b站
                # message = "去哔哩哔哩看“守护解放西4”并将第一集缓存到手机里" # b站
                # message = "去哔哩哔哩的创作中心查看稿件“神龙大侠之旅结束段”的数据" # b站
                # message = "查看哔哩哔哩离线缓存的视频" # b站

                # message = "播放陈奕迅的“十年”，并且收藏" # QQ音乐
                # message = "去QQ音乐中将弹唱版的“姑娘别哭泣”添加到自建歌单“测试”中" # QQ音乐
                # message = "开启QQ音乐的定时关闭，设为45分钟" # QQ音乐
                # message = "去QQ音乐查看我收藏的歌曲" # QQ音乐

                # message = "去Spotify搜索热门华语音乐歌单" # Spotify
                # message = "去Spotify查看我的歌单" # Spotify
                # message = "去Spotify中播放艺人团体“Red Velvet”的歌曲" # Spotify
                # message = "去Spotify中查看我喜欢的歌曲并播放" # Spotify

                # message = "在喜马拉雅找一个儿童的睡前故事播放" # 喜马拉雅
                # message = "去喜马拉雅中开启定时关闭功能，并选择播完3集声音" # 喜马拉雅
                # message = "开启喜马拉雅中的哄睡模式" # 喜马拉雅
                # message = "去喜马拉雅中查看我已过期的优惠券" # 喜马拉雅
                # message = "去喜马拉雅中查看我购买过的内容" # 喜马拉雅

                # message = "去哔哩哔哩漫画领取大会员的漫画福利券" # 哔哩哔哩漫画
                # message = "去哔哩哔哩漫画领取漫画福利券" # 哔哩哔哩漫画
                # message = "关闭哔哩哔哩漫画的自动购买下一话漫画的功能" # 哔哩哔哩漫画
                # message = "去哔哩哔哩漫画看我缓存下来的漫画" # 哔哩哔哩漫画
                # message = "去哔哩哔哩漫画看我的已购漫画" # 哔哩哔哩漫画
                # message = "去哔哩哔哩漫画查看国漫榜" # 哔哩哔哩漫画    

                # --------------聊天社交---------------
                # message = "去小红书搜索男生冬季穿搭并点赞第一个作品" # 小红书
                # message = "去小红书看我的浏览记录" # 小红书
                # message = "去小红书查看优惠券" # 小红书
                # message = "去小红书查看收礼记录" # 小红书

                # message = "去微信给“咖啡因从咖啡果中来”发消息，提醒她吃早饭" # 微信
                # message = "去看看我的微信朋友圈" # 微信
                # message = "去微信中发一条朋友圈“早上好”，且仅自己可见" # 微信
                # message = "去微信中发一条朋友圈，现拍一张照片，配文“早上好”，且仅自己可见" # 微信
                # message = "去微信面对面建群" # 微信
                # message = "去微信中进入聊天界面，添加“线条小狗第1弹”表情包" # 微信
                # message = "去微信中切换账号" # 微信
                # message = "去微信中打开贝壳找房" # 微信

                # message = "查看微博中所有@我的消息" # 微博
                # message = "查看微博热搜" # 微博
                # message = "去微博查看我发过的评论" # 微博
                # message = "找到西安交通大学的官方微博并关注" # 微博

                # --------------出行导航---------------
                # message = "我要用高德地图看用公交地铁到西安北站的路线" # 高德地图
                # message = "去高德地图中看卫星地图" # 高德地图
                # message = "去高德地图搜索西安北站并收藏" # 高德地图
                # message = "去高德地图将收藏的西安北站设置备注名为高铁站" # 高德地图

                # message = "去携程旅行预定西安北站附近的酒店双床房" # 携程旅行
                # message = "去携程旅行预定12月14日到12月15日西安永宁门附近的民宿" # 携程旅行
                # message = "去携程旅行看看北京的旅游景点" # 携程旅行
                # message = "去携程旅行查别人帮我订的机票" # 携程旅行
                # message = "去携程旅行查看我的旅行路线" # 携程旅行

                # message = "去铁路12306预订11月11日从西安北站到北京西站的火车票" # 铁路12306
                # message = "去铁路12306查看车站大屏" # 铁路12306
                # message = "去铁路12306查看12月27日从长沙到重庆的火车票" # 铁路12306
                # message = "去铁路12306申请临时身份证明" # 铁路12306
                # message = "去铁路12306查看12月6日的G1938次高铁、西安北到徐州东的高铁外卖订餐界面" # 铁路12306
                # message = "去铁路12306查询上海客服中心电话并拨打" # 铁路12306
                # message = "去铁路12306查看待评价的餐饮订单" # 铁路12306

                # message = "去移动交通大学预约班车，早上6：20从思源中心出发的那一趟" # 移动交通大学
                # message = "去移动交通大学查看横向合同立项情况" # 移动交通大学
                # message = "去移动交通大学预约健身房" # 移动交通大学
                
                # message = "在航旅纵横预订11月20日从西安到澳门的往返机票" # 航旅纵横
                # message = "在航旅纵横查看我待出行的行程"  # 航旅纵横
                # message = "去航旅纵横查询CA8888航班"  # 航旅纵横
                # message = "去航旅纵横将我的行程导入日历中" # 航旅纵横
                # message = "去航旅纵横查看上海浦东机场的机场大屏" # 航旅纵横
                # message = "去航旅纵横查看我的权益包" # 航旅纵横
                # message = "去航旅纵横添加个人的护照信息" # 航旅纵横

                # --------------购物消费---------------
                # message = "去淘宝中搜索罗技键盘并选择第一件商品" # 淘宝
                # message = "查看我淘宝中收藏的宝贝" # 淘宝
                # message = "删除淘宝订单的第一条退款记录" # 淘宝
                # message = "去淘宝给所有订单添加价保" # 淘宝

                # message = "全选盒马购物车中的商品并删除" # 盒马
                # message = "清空盒马的购物车" # 盒马
                # message = "去盒马购买大闸蟹" # 盒马
                # message = "去盒马查看我开过的发票" # 盒马
                # message = "去盒马添加收货地址" # 盒马

                # message = "去拼多多搜索发卡，并选择品牌商品" # 拼多多
                # message = "去拼多多中查看待处理的退款订单" # 拼多多
                # message = "去拼多多中查看我的举报投诉的处理进度" # 拼多多
                # message = "查看我拼多多中收藏的商品" # 拼多多

                # --------------摄影摄像---------------
                # message = "去美图秀秀现拍一张照片并添加滤镜“深夜食堂”" # 美图秀秀
                # message = "去美图秀秀制作一张一寸照" # 美图秀秀
                # message = "去美图秀秀设置我的个人水印" # 美图秀秀
                # message = "去美图秀秀设置不允许他人保存我的图片" # 美图秀秀
                
                # message = "去轻颜查看我用过的特效" # 轻颜
                # message = "将轻颜的闪光灯设置打开" # 轻颜
                # message = "去轻颜将男生妆容适配关闭" # 轻颜
                # message = "" # 

                # --------------金融理财---------------
                # message = "支付宝中查看收款记录" # 支付宝
                # message = "将支付宝余额提现" # 支付宝
                # message = "支付宝中帮我查看本月账单的支出情况" # 支付宝
                # message = "支付宝中帮我交10元电费" # 支付宝
                # message = "在支付宝中取消淘宝的免密支付" # 支付宝

                # message = "查看零钱金额" # 微信
                # message = "打开微信收款码" # 微信
                # message = "用微信交供暖费" # 微信
                # message = "在微信中暂停使用付款码支付功能" # 微信
                
                # message = "" # 
                # --------------美食娱乐---------------
                # message = "去饿了么查找3km内的奶茶店" # 饿了么
                # message = "用饿了么看看炸鸡的外卖，销量优先" # 饿了么
                # message = "去饿了么中新增收货地址" # 饿了么         
                # message = "去饿了么看看我的优惠券" # 饿了么
                # message = "我想吃肯德基的甜品两件套" # 饿了么
                # message = "去饿了么中开通小额免密支付" # 饿了么

                # message = "在美团中到黄焖鸡米饭店点一份大份微辣黄焖鸡米饭" # 美团
                # message = "去美团中查看我的收藏" # 美团
                # message = "去美团中查看3公里内的KTV" # 美团
                # message = "去美团中查看我退款的订单" # 美团

                # message = "去大麦查看我看过的演出票" # 大麦
                # message = "去大麦添加新的观演人" # 大麦
                # message = "去大麦看看我附近的脱口秀的票" # 大麦
                # message = "开启大麦的积分过期提醒" # 大麦
                # message = "去大麦把我看过的德云社演出的官方电子纪念票存入相册" # 大麦

                # --------------体育运动---------------
                # message = "查看/设置运动步数？" # keep
                # message = "去Keep开始行走记录" # keep
                # message = "去Keep查看我的运动装备" # keep
                # message = "去Keep查看“八段锦”系列的课程" # keep
                # message = "去Keep中开启体态评估" # keep

                # message = "去训记中查看二头肌的动作" # 训记
                # message = "去训记中查看我的身体数据" # 训记
                # message = "去训记中观看俯身飞鸟动作演示" # 训记
                # message = "去训记中查看训练的数据统计" # 训记

                # message = "去华为运动健康设置跑步的心率上限值为198次/分钟" # 华为运动健康
                # message = "" # 

                # --------------学习资讯---------------
                # message = "用网易有道词典翻译一下“deep learning”" # 网易有道词典
                # message = "去网易有道词典背今天的单词”" # 网易有道词典
                # message = "去网易有道词典下载离线的牛津词典" # 网易有道词典

                # message = "去微信读书将西游记添加到书架中" # 微信读书
                # message = "查看我微信读书的阅读时长" # 微信读书
                # message = "去微信读书设置允许使用音量键翻页" # 微信读书

                # message = "查看今日头条的头条热榜" # 今日头条
                # message = "去查看今日头条的浏览历史" # 今日头条
                # message = "打开今日头条的新闻推送" # 今日头条

                # message = "搜索问题“CS ranking in China”，并打开第一个热门回答" # 知乎
                # message = "去知乎中查看我收藏的内容" # 知乎
                # message = "去知乎中查看我赞过的内容" # 知乎
                
                # --------------效率办公--------------
                # message = "将网易邮箱大师中的所有未读邮件标为已读" # 网易邮箱大师
                # message = "去网易邮箱大师设置为收取全部邮件" # 网易邮箱大师

                # message = "在腾讯会议中开一个快速会议，打开视频，并使用个人会议号" # 腾讯会议

                # message = "去系统应用“日历”中给圣诞节添加提醒事项“买礼物”" # 日历
                # message = "在日历中搜索“设置”" # 日历

                # message = "去qq邮箱打开收件箱并查看每日悦读订阅的内容" # qq邮箱
                # message = "去qq邮箱给631080500@qq.com写邮件" # qq邮箱

                # message = "去手机应用“信息”中将查看通知信息" # 信息
                # message = "去系统应用“信息”中删除所有骚扰信息" # 信息

                # message = "给联系人中的阿里巴巴钉钉客服打电话" # 电话

                # message = "将联系人中的阿里巴巴钉钉客服的二维码信息分享给微信的文件传输助手" # 联系人
                
                # message = "在备忘录中的待办添加事项“周日去拔牙”" # 备忘录
                # message = "在备忘录中查看我收藏的笔记" # 备忘录
                
                # message = "打开一个chrome浏览器的无痕浏览窗口" # chrome
                # message = "清除Chrome浏览器的Cookie数据" # chrome
                # message = "将Chrome浏览器的默认搜索引擎设置为Bing" # chrome

                # --------------便捷生活--------------
                # message = "去中国移动充20元话费" # 中国移动
                # message = "去中国移动查询我订阅的套餐" # 中国移动

                # message = "去菜鸟查看我有几个待取包裹" # 菜鸟裹裹
                # message = "去菜鸟开启丰巢小程序的取件授权" # 菜鸟裹裹
                # message = "去菜鸟添加家人账号" # 菜鸟裹裹
                
                # message = "打开支付宝的公交乘车码" # 支付宝
                # message = "去支付宝交电费" # 支付宝
                # message = "去移动交通大学查看我的安全邮箱" # 支付宝
                # message = "去支付宝中的“校园派”小程序中给我的校园卡充值" # 支付宝

                # message = "去系统应用“天气”中查看北京天气" # 天气

                # message = "去系统应用“时钟”中新建一个上午9点半的闹钟" # 时钟
                # message = "去系统应用“时钟”查看纽约的时间" # 时钟

                

                # --------------功能设置--------------
                # message = "开启个人移动WLAN热点" # 设置
                # message = "打开蓝牙并连接airpods" # 设置
                # message = "去设置中将时间设为24小时制"  # 设置
                # message = "去设置中打开电池管理" # 设置

                # message = "我想要将微信界面更改为英文界面" # 微信
                # message = "去微信中在我的发现页添加“搜一搜”功能" # 微信

                # message = "查看支付宝账号的邮箱信息" # 支付宝
                # message = "支付宝的深色模式设置为跟随系统" # 支付宝
                # message = "去支付宝中设置不可通过转账页面添加我" # 支付宝

                # message = "去手机应用“天气”中查看上海市静安区的天气" # 华为天气

                # message = "在联系人界面设置卞艺衡的工作单位为交大" # 联系人
                #endregion
                taskdf = pd.read_excel(TaskTable_PATH, keep_default_na = False) 
                tasklist = taskdf['任务内容'].tolist() # 任务列表
                inputarray = command.split(" ")
                if len(inputarray) == 1:
                    num = input("请输入任务序号：")
                else:
                    num = inputarray[1]
                while not num.isdigit() or tasklist[int(num) - 2] == '':
                        num = input("请输入有效的任务序号：")
                message = "Q：" + tasklist[int(num) - 2]
                # message = "Q：" + message
                if message.startswith("Q"):# 处理以 "任务为:" 开头的消息                   
                    task_str = str(message)
                    task_str = task_str.split('Q：')[-1]
                    response = "已收到任务内容：" + task_str
                    print(response)

                    # 若是再次输入m，则重置此次任务的所有信息
                    if len(tapped_button_str) == 1:
                        # # invalid_button_str = ['']
                        # # taped_element_content = ''
                        # # tapped_button_str = store_tapped_button_str
                        # print("您再次输入了任务内容，且有了操作记录，现在会重置此次任务，请回到初始界面")
                        shutil.copy(save_path_default, save_path_old)
                        default_reason = ""
                        invalid_button_str = [""]
                        taped_element_content = ""
                        taped_element_roi = None
                        tapped_button_str = ["，现在已经进行了的操作有"]
                        GPT4V_history.__init__(language, gpt_choice) 
                                              
                    # 根据任务内容获取帮助文档信息
                    # if not help_get_flag:
                    #     help_str = ''
                    # else:
                    #     help_question, help_content = help_seq_get(task_str)
                    #     help_str = f"此次任务的帮助文档信息：{help_question} {help_content}" if help_question and help_content else ''
                    #     print(help_str)

                else:
                    # 处理其他消息
                    print("Received a different message")  
                    
            #发送截图指示
            elif command.startswith('i'):
                
                #截图 or 长截图
                if longscreenshot_flag:
                    capture_longscreenshot(save_path_new)
                else:
                    capture_screenshot(save_path_new)


                # 根据图片是否更新，判断上次点击的控件的可点击性（没更新就是不可点击）
                # screenshot_update_flag = not(are_images_similar(save_path_old, save_path_new, roi = taped_element_roi))  
                # if (not screenshot_update_flag) and taped_element_content != '' and not default_reason: # 图片没更新 + 不是在发送了m之后 + 不是过程中出现了 界面不存在该元素或回答格式不对
                #     tapped_button_str.pop() # 删去上次点击的控件信息
                #     # 删去llm上次的input和output信息
                #     GPT4V_history.clear_previous_one_record()
                #     GPT4V_history.clear_previous_one_record()

                #     invalid_button_str.append(f"{taped_element_content} ") # 记录这个无效的button
                # else:
                #     invalid_button_str = [''] #重置无效控件列表

             
                # process_img_4gpt4_script(label_path_dir, save_path_old, output_root, layout_json_dir, high_conf_flag, alg,
                #             clean_save, plot_show, ocr_save_flag, model_ver, model_det, pd_free_ocr=ocr,
                #             ocr_only=ocr_output_only, workflow_only=workflow_only, accurate_ocr=accurate_ocr, lang=lang, 
                #             img_4gpt4_out_path=img_4gpt4_out_path, gpt4_text_size=gpt4_text_size, gpt4_text_thickness=gpt4_text_thickness)



                # 提示llm的注意事项
                # if len(invalid_button_str) == 1: # =1代表初始，即没有无效控件
                #     invalid_button_string = '。'
                #     if default_reason.startswith("界面中不存在"):
                #         invalid_button_string = f"。注意{default_reason}。"
                #         default_reason = ""
                # else: #存在无效控件，即不可点击的控件
                #     invalid_button_string = ''.join(map(str, invalid_button_str)) #无效控件提醒
                #     invalid_button_string = f"。注意id为{invalid_button_string}的控件不可点击。"
                #     if default_reason.startswith("界面中不存在"):
                #         invalid_button_string = f"。注意id为{invalid_button_string}的控件不可点击,{default_reason}。"
                #         default_reason = ""

                # 整合要送入llm的信息
                # if len(tapped_button_str) == 1: # =1代表初始，即没有点击过控件
                #     object_message_humanword = f'Q：{task_str}{invalid_button_string}'
                # else:
                #     tapped_button_string = ''.join(map(str, tapped_button_str))
                #     object_message_humanword = f'Q：{task_str}{tapped_button_string}{invalid_button_string}'
                   
                # object_message_humanword = f'Q：{task_str}{invalid_button_string}'
                object_message_humanword = f'Q：{task_str}'
                
                print("\n" + object_message_humanword)

                # 使用大语言模型生成json格式的操作命令, 调用 connect_gpt 函数
                order_list = use_LLM(gpt_choice, json.dumps(object_message_humanword))

                # for order in order_list:
                #     if order['action'] != 'default': # llm 返回有效命令
                #         # 若此次点击的控件与上次点击控件不同，此次有效
                #         if taped_element_content != order['button_content']:
                #             taped_element_content = order['button_content'] # 点击的控件内容
                #             taped_element_roi = eleLoc_2_roi(order['element_location']) # 点击的位置
                            
                #             # 记录操作历史
                #             if order['action'] == 'keyboard_input':
                #                 input_text_content = order['data']['input_text']
                #                 tapped_button_str.append(f'（在id为{repr(taped_element_content)}的输入框内输入了{repr(input_text_content)}并回车）')
                #             elif order['action'] == 'tap':
                #                 tapped_button_str.append(f'（点击了id为{repr(taped_element_content)}的控件）')
                #     else:
                #         default_reason = order['reason']
                #         print("请再发送一次截图，GPT将重新生成操作命令")

                response = json.dumps(order_list, ensure_ascii=False)  # 转换order_list为JSON格式
              
                print(f"回复客户端消息: {response}")  # 处理消息并生成回复
                response = eval(response)

                        
            else:
                print("无效的命令")
                continue

            if isinstance(response, str):
                # print("response 是字符串类型")
                pass
            elif isinstance(response, list):
                order_list = response
                # operator(order_list)  # 传递服务器返回的order_list中的每个操作
                print("*---------------------------执行命令完毕-----------------------------------*\n") 
            else:
                print("response 是其他类型")
                
        
    finally:
        print(f"client_main发生异常")


if __name__ == "__main__":

    client_main()
