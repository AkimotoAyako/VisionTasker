# -*- coding: utf-8 -*-
import element.detect_text.ocr as ocr
from element.detect_text.Text import Text
import cv2
import json
import time
import os
from os.path import join as pjoin
import base64
import requests


def save_detection_json(file_path, texts, img_shape, clean_save=False):
    # /data4/bianyiheng/task1/outputs/ocr/1.json
    output = {'img_shape': img_shape, 'texts': []}
    for text in texts:
        c = {'id': text.id, 'content': text.content}
        loc = text.location
        c['column_min'], c['row_min'], c['column_max'], c['row_max'] = loc['left'], loc['top'], loc['right'], loc[
            'bottom']
        c['width'] = text.width
        c['height'] = text.height
        output['texts'].append(c)
    if not clean_save:
        f_out = open(file_path, 'w', encoding='utf-8')
        json.dump(output, f_out, indent=4, ensure_ascii=False)
    return output


def visualize_texts(org_img, texts, shown_resize_height=None, show=False, write_path=None, clean_save=False):
    img = org_img.copy()
    for text in texts:
        text.visualize_element(img, line=2)

    img_resize = img
    if shown_resize_height is not None:
        img_resize = cv2.resize(img, (int(shown_resize_height * (img.shape[1] / img.shape[0])), shown_resize_height))

    if show:
        cv2.imshow('texts', img_resize)
        cv2.waitKey(0)
        cv2.destroyWindow('texts')
    if write_path is not None and not clean_save:
        cv2.imwrite(write_path, img_resize)
    return img_resize


def text_sentences_recognition(texts):
    '''
    Merge separate words detected by Google ocr into a sentence 通过迭代的方式尝试合并相邻的文本块，直到无法再合并为止
    '''
    changed = True  # 初始化为真 跟踪本迭代中是否发生了合并 如果之前有发生过文本合并 则需要再过一次 直到不能再合并为止
    while changed:  # 不断尝试合并文本
        changed = False
        temp_set = []  # 临时存储要合并的文本
        for text_a in texts:  # 每一个文本都会和之前扔过去集合里的全部比一次
            merged = False  #
            for text_b in temp_set:  # 判断是不是一行
                if text_a.is_on_same_line(text_b, 'h', bias_justify=0.2 * min(text_a.height, text_b.height),
                                          bias_gap=0.5 * max(text_a.word_width, text_b.word_width)):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:  # 如果没有合并就原封不动放过去
                temp_set.append(text_a)
        texts = temp_set.copy()

    for i, text in enumerate(texts):
        text.id = i  # 给文本编号
    return texts


def merge_intersected_texts(texts):
    '''
    Merge intersected texts (sentences or words)
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_intersected(text_b, bias=2):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return texts


def text_cvt_orc_format(ocr_result):
    texts = []  # 简单格式转换
    if ocr_result is not None:
        for i, result in enumerate(ocr_result):
            error = False
            x_coordinates = []
            y_coordinates = []
            text_location = result['boundingPoly']['vertices']
            content = result['description']
            for loc in text_location:
                if 'x' not in loc or 'y' not in loc:
                    error = True
                    break
                x_coordinates.append(loc['x'])
                y_coordinates.append(loc['y'])
            if error: continue
            location = {'left': min(x_coordinates), 'top': min(y_coordinates),
                        'right': max(x_coordinates), 'bottom': max(y_coordinates)}
            texts.append(Text(i, content, location))
    return texts


def text_filter_noise(texts):
    valid_texts = []
    for text in texts:
        if len(text.content) <= 1 and text.content.lower() not in ['日', '一', '二', '三', '四', '五', '六', '我', '0', '1',
                                                                   '2', '3', '4', '5', '6', '7', '8', '9', '订']:
            continue
        valid_texts.append(text)
    return valid_texts


def text_detection(input_file='../data/input/30800.jpg', ocr_root='../data/output/ocr', ocr_save='',
                   clean_save=False, accurate_ocr=True, ocr=None, show=False):
    start = time.time()
    name = input_file.replace('\\', '/').split('/')[-1][:-4]

    if accurate_ocr:  # 启动ocr识别高精度版！
        # 在这里写入高精度ocr
        API_KEY = "xxxxxxxx"  # username 用户名
        SECRET_KEY = "xxxxxxxx"  # key 密码

        pic_dir = input_file
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token=" + str(
            requests.post(url, params=params).json().get("access_token"))
        f = open(pic_dir, 'rb')
        img = base64.b64encode(f.read())
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        payload = {
            "image": img,
            "probability": 'true'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        result_all = response.text
        result_all = eval(result_all)
        # print(result_all)

        # 结果变形
        ocr_result = []
        for one_line in result_all["words_result"]:
            if one_line['probability']['average'] >= 0.84:  # 结果置信度过滤
                left_b, right_b, top_b, bottom_b = one_line['location']['left'], one_line['location']['left'] + \
                                                   one_line['location']['width'], \
                                                   one_line['location']['top'], one_line['location']['top'] + \
                                                   one_line['location']['height']
                curr_line = {
                    'description': one_line['words'],
                    'boundingPoly':
                        {'vertices':
                             [{'x': left_b, 'y': top_b},  # 左上
                              {'x': right_b, 'y': top_b},  # 右上
                              {'x': right_b, 'y': bottom_b},  # 右下
                              {'x': left_b, 'y': bottom_b}  # 左下
                              ]
                         }
                }
                ocr_result.append(curr_line)
    else:
        result = ocr.ocr(input_file, cls=True)
        line = result[0]
        ocr_result = []
        for one_line in line:
            curr_line = {
                'description': one_line[1][0],
                'boundingPoly':
                    {'vertices':
                         [{'x': one_line[0][0][0], 'y': one_line[0][0][1]},
                          {'x': one_line[0][1][0], 'y': one_line[0][1][1]},
                          {'x': one_line[0][2][0], 'y': one_line[0][2][1]},
                          {'x': one_line[0][3][0], 'y': one_line[0][3][1]}
                          ]
                     }
            }
            ocr_result.append(curr_line)

    texts = text_cvt_orc_format(ocr_result)  # 转换格式 [text] (text自定义）
    texts = merge_intersected_texts(texts)  # 合并相交的文字
    texts = text_filter_noise(texts)  # 文字去噪声 去掉部分单个字
    texts = text_sentences_recognition(texts)  # 迭代句子合并识别
    img = cv2.imread(input_file)
    res_img = visualize_texts(img, texts, shown_resize_height=800, show=show,
                              write_path=pjoin(ocr_root, name + '.jpg'), clean_save=clean_save)  # 文本识别可视化+存图

    text_js = save_detection_json(pjoin(ocr_root, name + '.json'), texts, img.shape, clean_save=clean_save)  # 存文本识别的json

    return res_img, text_js  # ndarray 800,467,3
