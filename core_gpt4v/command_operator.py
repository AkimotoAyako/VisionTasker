import time
import subprocess
import re
import os
import cv2  
import numpy as np
import base64

from core_gpt4v.Config import DefaultInputKeyboard, DeviceName
 

screenshotPath = "screenshot/"

longscreenshot_flag = False  # 长截图  True / False

# existing_driver = None  # 假设没有现有的驱动程序

status_bar_height = 85
screen_size_height = 1920
screen_size_width = 1080
total_height = 1920
actual_workable_screen_size_height = 1920

duration_time_ms = 2500
wo_status_bar_ratio = 0.965


def execute_adb(adb_command):
    subprocess.run(adb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
def simulate_enter_key(keycode):
    try:
        # 执行 adb shell input keyevent 66 命令
        adb_command = f"adb -s DeviceName shell input keyevent {str(keycode)}"
        execute_adb(adb_command)
        # subprocess.run(["adb", "-s", DeviceName, "shell", "input", "keyevent", str(keycode)], check=True)
        # print("Simulated Enter key event successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Failed to simulate Enter key event.")

def send_text_by_adbkeyboard(text):   
    chars = ",".join([str(ord(char)) for char in text])# 将文本转换为以逗号分隔的Unicode代码点
    adb_command = f"adb -s {DeviceName} shell am broadcast -a ADB_INPUT_CHARS --eia chars \'{chars}\'"# 构建 adb 命令  
    print(adb_command)
    try:
        execute_adb(adb_command)
        time.sleep(0.5)       
        print(f"Send texts successfully: {text}")
    except subprocess.CalledProcessError as e:
        print(f"Error sending text: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def perform_action(action_data):
    global screen_size_height
    global status_bar_height
    global screen_size_width
    global total_height
    global actual_workable_screen_size_height
    action_type = action_data['action']

    if action_type == 'swipe':
        swipe_data = action_data['data']
        start_x, start_y = swipe_data['start']
        start_y = screen_size_height-1 if start_y >= screen_size_height else start_y 
        end_x, end_y = swipe_data['end']
        end_y = screen_size_height-1 if end_y >= screen_size_height else end_y 
        duration_ms = swipe_data.get('duration_ms', 800)  # 默认滑动持续时间为 800 毫秒
        # swipe_action = TouchAction(driver)
        # swipe_action.press(x=start_x, y=start_y).wait(duration_ms).move_to(x=end_x, y=end_y).release().perform()
        # time.sleep(0.5)  # 可选，添加等待时间以确保操作完成
        
        swipe_command = f"adb -s {DeviceName} shell input swipe {start_x} {start_y} {end_x} {end_y} {duration_ms}"
        execute_adb(swipe_command)
        time.sleep(0.5)
        
    elif action_type == 'longpress':
        tap_data = action_data['data']
        tap_x, tap_y = tap_data['tap_point']
        # tap_y = screen_size_height-1 if tap_y + status_bar_height >= screen_size_height else tap_y + status_bar_height
        tap_y = screen_size_height-1 if tap_y  >= screen_size_height else tap_y 
        print(f"tap_y:{tap_y}")

        duration_ms = 1500
        longpress_command = f"adb -s {DeviceName} shell input swipe {tap_x} {tap_y} {tap_x+1} {tap_y+1} {duration_ms}"
        execute_adb(longpress_command)
        time.sleep(0.2)

    elif action_type == 'tap':
        tap_data = action_data['data']
        tap_x, tap_y = tap_data['tap_point']
        if not longscreenshot_flag:
            tap_y = screen_size_height-1 if tap_y >= screen_size_height else tap_y

        else:#长截图下的点击操作
            screen_height_without_statusbar = int(screen_size_height * wo_status_bar_ratio)

            # defaultroute=[int(screen_size_height/12 *5), int(screen_size_height/12 *3)]
            defaultroute=[int(screen_size_height/24 *13), int(screen_size_height/24 *11)]

            swipe_distance = defaultroute[0] - defaultroute[1]
            supported_distance = total_height - screen_size_height
            blank_zone = int(total_height - (1 -wo_status_bar_ratio ) * screen_size_height)
            source_tap_y = tap_y
            while tap_y  >= screen_height_without_statusbar:#坐标大于当前界面的非底部导航栏区域
                if tap_y  <= screen_size_height:
                    # if source_tap_y<blank_zone: #坐标处于当前界面的导航栏中，实际上是在长截图的非底部导航栏的区域中，需要再滑动一下；否则直接点击
                        # TouchAction(driver).press(x=screen_size_width/2, y=defaultroute[0]).wait(duration_time_ms).move_to(x=screen_size_width/2, y=defaultroute[0]-(tap_y-screen_height_without_statusbar)).release().perform()
                        supported_distance -= (tap_y-screen_height_without_statusbar)
                        tap_y = screen_height_without_statusbar-1 if tap_y < actual_workable_screen_size_height else actual_workable_screen_size_height-1
                else: #坐标大于当前界面
                    swipe_distance_y2bottom = tap_y-screen_size_height +1
                    swipe_distance_shortest = total_height-source_tap_y +1
                    min_swipe_distance = min(swipe_distance_y2bottom, swipe_distance_shortest,supported_distance)
                    # min_swipe_distance = min(swipe_distance_y2bottom, supported_distance)
                    if source_tap_y + swipe_distance < total_height:  #实际操作中滑动可以达到设定距离       
                        # TouchAction(driver).press(x=screen_size_width/2, y=defaultroute[0]).wait(duration_time_ms).move_to(x=screen_size_width/2, y=defaultroute[1]).release().perform()
                        supported_distance -= swipe_distance
                        tap_y -= swipe_distance
                    elif source_tap_y + swipe_distance_y2bottom < total_height:#否则滑动距离为 到上个界面底部的距离
                        # TouchAction(driver).press(x=screen_size_width/2, y=defaultroute[0]).wait(duration_time_ms).move_to(x=screen_size_width/2, y=defaultroute[0]-swipe_distance_y2bottom).release().perform()
                        tap_y -= swipe_distance_y2bottom              
                    else:
                        # TouchAction(driver).press(x=screen_size_width/2, y=defaultroute[0]).wait(duration_time_ms).move_to(x=screen_size_width/2, y=defaultroute[0]-min_swipe_distance).release().perform()
                        supported_distance -= min_swipe_distance
                        tap_y -= min_swipe_distance

        print(f"tap_y:{tap_y}")

        # TouchAction(driver).tap(x=tap_x, y=tap_y).perform()
        # time.sleep(0.2)  # 可选，添加等待时间以确保操作完成
        tap_command = f"adb -s {DeviceName} shell input tap {tap_x} {tap_y}"
        execute_adb(tap_command)
        time.sleep(0.2)
        
    elif action_type == 'keyboard_input':
        input_data = action_data['data']
        text_to_input = input_data['input_text']  # 要输入的文本内容
        tap_x, tap_y = input_data['tap_point']
        # tap_y = screen_size_height-1 if tap_y + status_bar_height >= screen_size_height else tap_y + status_bar_height
        tap_y = screen_size_height-1 if tap_y >= screen_size_height else tap_y

        tap_command = f"adb -s {DeviceName} shell input tap {tap_x} {tap_y}"
        execute_adb(tap_command) 
        time.sleep(0.5)

        adb_command_enable_adb_ime = f"adb -s {DeviceName} shell ime enable com.android.adbkeyboard/.AdbIME"
        adb_command_change2adbkeyboard = f"adb -s {DeviceName} shell ime set com.android.adbkeyboard/.AdbIME"
        # subprocess.run(adb_command_enable_adb_ime, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)  # enable adbkeyboard
        os.system(adb_command_enable_adb_ime)  # enable adbkeyboard
        time.sleep(0.3)
        # subprocess.run(adb_command_change2adbkeyboard, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)  # 切换为adbkeyboard
        os.system(adb_command_change2adbkeyboard)  # 切换为adbkeyboard
        time.sleep(0.3)
        # tap_action.tap(x=tap_x, y=tap_y).perform()
        # time.sleep(0.2)

        send_text_by_adbkeyboard(text_to_input)

        enter_command = f"adb -s {DeviceName} shell input keyevent 66"
        execute_adb(enter_command) #enable adbkeyboard
        time.sleep(0.2)
        
        adb_command_change_IME_default = f"adb -s {DeviceName} shell ime set {DefaultInputKeyboard}"
        execute_adb(adb_command_change_IME_default) #切换为拼音输入法 
        time.sleep(0.2)  # 可选，添加等待时间以确保操作完成
        # simulate_enter_key(66)

    else:
        print(f"不支持的操作类型: {action_type}")

def screenshot_adb(screenshot_path):
    screenshot_command = f"adb -s {DeviceName} shell screencap -p /sdcard/screenshot.png"
    pull2local_command = f"adb -s {DeviceName} pull /sdcard/screenshot.png {screenshot_path}"
    delete_sd_command = f"adb -s {DeviceName} shell rm /sdcard/screenshot.png"
    execute_adb(screenshot_command)
    time.sleep(0.2)
    execute_adb(pull2local_command) 
    time.sleep(0.2)
    execute_adb(delete_sd_command)  
    time.sleep(0.1)
    
def capture_screenshot(screenshot_path):
    global screen_size_height
    global status_bar_height 
    # if driver : screen_size = driver.get_window_size() 
    # screen_size_height = screen_size["height"]
    # print('ScreenShot Size: ' + str(screen_size["width"]) +',' + str(screen_size["height"]))

    try:
        screenshot_adb(screenshot_path)

        screenshotsize = cv2.imread(screenshot_path).shape
        print(screenshotsize)
        screen_size_height = screenshotsize[0]
        # status_bar_height = screenshotsize[0]-screen_size["height"]

        print(f"截图已保存至 {screenshot_path}")
    except Exception as e:
        print(f"截图失败: {str(e)}")
 
    # print("screenshot session has been stopped.")

def stitch(img1, img2, ratio):
    if img1 is None or img2 is None:
        raise FileNotFoundError
    if img1.shape != img2.shape:
        raise ValueError("images do not have the same width!")

    # ignore both sides 20 pixels
    # there maybe some border or scroll bar
    # img1 = img1[0 : img1.shape[0], 20 : img1.shape[1] - 20, 0 : img1.shape[2]]
    # img2 = img2[0 : img2.shape[0], 20 : img2.shape[1] - 20, 0 : img2.shape[2]]
    img1 = img1[0 : img1.shape[0], 0 : img1.shape[1], 0 : img1.shape[2]]
    img2 = img2[0 : img2.shape[0], 0 : img2.shape[1], 0 : img2.shape[2]]

    if img1.shape[2] == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        gray2 = img2

    h, w = gray1.shape

    thresh1 = cv2.adaptiveThreshold(
        gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    thresh2 = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )

    height = h
    sub = thresh1 - thresh2
    sub = cv2.medianBlur(sub, 3)
    sub = sub // 255
    thresh = w // 10

    # ignore top, bottom 3% field
    # there maybe some border or scroll bar
    min_height = h * wo_status_bar_ratio
    for i in range(h - 1, 0, -1):
        if np.sum(sub[i]) > thresh and height < min_height:
            # print(np.sum(sub[i]))
            break
        height = height - 1
    block = sub.shape[0] // ratio

    templ = gray1[height - block : height,]
    if templ.shape[0] < block:
        print("templ shape is too small", templ.shape)
        return 0, 1, 0

    res = cv2.matchTemplate(gray2, templ, cv2.TM_SQDIFF_NORMED)
    mn_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # print("thresh, shape ,height, block", thresh, sub.shape, height, block)
    # print("mn_val, max_val, min_loc, max_loc", mn_val, max_val, min_loc, max_loc)

    # Considering imaging compression and other error, 90% similarity is required for a match
    # And it could be adjusted if stitch badly
    errorRate = 0.1
    if mn_val < errorRate:
        return 1, (height - block) / h, min_loc[1] / h
    else:
        return 0, 1, 0

def drawImgList(imgDirPath):
    images = []
    stitchRate = []

    files = os.listdir(imgDirPath)
    if len(files) < 2:
        raise ValueError("The number of pictures is less than 2!")
    for i in range(len(files)):
        image = cv2.imread(imgDirPath + "/" + files[i])
        images.append(image)

    h, w, c = images[0].shape
    for i in range(len(files) - 1):
        print("Stitching [{}/{}] images".format(i + 1, len(files) - 1))
        result, bottom, top = stitch(images[i], images[i + 1], 15)
        if result == 0:
            # print("Something was wrong with the stitching! Try to changes the ratio.")
            bottom = h
            top = h
        elif result == 1:
            bottom = int(bottom * h)
            top = int(top * h)
        stitchRate.append([bottom, top])

    for i in range(len(files) - 1)[::-1]:
        bottom, top = stitchRate[i]
        images[i + 1] = images[i + 1][top:]
        images[i] = images[i][0:bottom]

    return cv2.vconcat(images)

def capture_longscreenshot(screenshot_path, maxSwipe=2, defaultroute=[1500, 500]):
    global screen_size_height
    global status_bar_height
    global screen_size_width
    global total_height
    global actual_workable_screen_size_height

    # screen_size = driver.get_window_size()
    # actual_workable_screen_size_height = screen_size["height"]

    screenshot_adb(screenshotPath + "test" + ".png")
    testimage = cv2.imread(screenshotPath + "test" + ".png")  # 读取图像并检查是否成功
    if testimage is not None:
        screen_size_height, screen_size_width, channels = testimage.shape
        print(f"\nsingle screenshot size: {testimage.shape}\n")

    defaultroute=[int(screen_size_height/3 *2), int(screen_size_height/3)]
    
    try:
        filename = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        dirpath = screenshotPath + filename
        os.mkdir(dirpath)
        # printScreen(dirpath)
        # driver.save_screenshot(dirpath + "/" + str(int(time.time())) + ".png")
        screenshot_adb(dirpath + "/" + str(int(time.time())) + ".png")
        for i in range(maxSwipe):
            swipe_command = f"adb -s {DeviceName} shell input swipe {screen_size_width/2} {defaultroute[0]} {screen_size_width/2} {defaultroute[1]} 1000"
            execute_adb(swipe_command)
            # time.sleep(0.2)
            screenshot_adb(dirpath + "/" + str(int(time.time())) + ".png")
            
            
        imgStitch = drawImgList(dirpath)
        cv2.imwrite(screenshot_path, imgStitch)
        for i in range(maxSwipe):
            # swipeScreen([defaultroute[1], defaultroute[0]])
            # TouchAction(driver).press(x=screen_size_width/2, y=defaultroute[1]).wait(1000).move_to(x=screen_size_width/2, y=defaultroute[0]).release().perform()
            swipe_command = f"adb -s {DeviceName} shell input swipe {screen_size_width/2} {defaultroute[1]} {screen_size_width/2} {defaultroute[0]} 1000"
            execute_adb(swipe_command)
            time.sleep(0.2)
            
        print(f"截图已保存至 {screenshot_path}")
        if imgStitch is not None:
            total_height, width, channels = imgStitch.shape
            print(f"\ntotal screenshot size: {imgStitch.shape}\n")
    except Exception as e:
        print(f"截图失败: {str(e)}")

def operator(order_list): 
    global screen_size_height
    global status_bar_height
    global screen_size_width
    global total_height
    global actual_workable_screen_size_height
    #返回默认的无操作时
    for order in order_list:
        if order['action'] == 'default':
            print(f"GPT给出无效命令，当前无操作，请再发送一次截图")
            return 0       

    # screen_size = driver.get_window_size()

    # print('Operating Screen Size: ' + str(screen_size["width"]) +', ' + str(screen_size["height"]))

    for order in order_list:
        perform_action(order)

if __name__=="__main__":
    # driver = start_appium_session()

    swipe_action_data = {'action': 'swipe', 'data': {'start': [500, 1400], 'end': [500, 400], 'duration_ms': 800}}
    tap_action_data = {'action': 'tap', 'data': {'tap_point': [536, 2712]}}
    longpress_action_data = {'action': 'longpress', 'data': {'tap_point': [575, 670]}}
    keyinput_action_data = {
        'action': 'keyboard_input',
        'data': {
            # 'tap_point': [500, 150],  # 指定要在哪个位置执行键盘输入
            # 'tap_point': [539, 354],
            'tap_point': [542, 502],
            
            'input_text': '中文'  # 要输入的文本内容
            # 'text': 'CAT'  # 要输入的文本内容
        }
    }  
    # order_list = [swipe_action_data, tap_action_data]

    order_list = [keyinput_action_data]
    # order_list = [{'action': 'tap', 'data': {'tap_point': [536, 2784]}, 'button_content': '切换账号', 'element_location': {'left': 422, 'right': 651, 'top': 2753, 'bottom': 2816}, 'command': "A：tap_action： 点击['切换账号']"}]

    image_path = 'screenshot/screenshot_new.png'
    # image_path = image_path.replace('\\','/')  # 要发送的图片路径
    # if longscreenshot_flag:
    #     capture_longscreenshot(image_path)
    # else:
    #     capture_screenshot(image_path)
    # capture_longscreenshot(image_path)
    operator(order_list)


