import subprocess
from core.Config import XML_file_PATH, DeviceName
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
def get_uidump():
    subprocess.run(['adb', "-s", DeviceName,'shell', 'uiautomator', 'dump', '/sdcard/screenshot.xml'])

    subprocess.run(['adb', "-s", DeviceName,'pull', '/sdcard/window_uidump.xml', XML_file_PATH])

get_uidump()