# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np
from os.path import join as pjoin
import os
import time
import shutil

from element.detect_merge.Element import Element
import element.detect_compo.lib_ip.ip_preprocessing as pre


# uied图像的生成！ 真的版本
def show_elements(org_img, eles, show=False, win_name='element', wait_key=0, shown_resize=None, line=2, draw_rec=True, gpt4_text_size=0.55, gpt4_text_thickness=2):
    color_map = {'Text': (196, 114, 68), 'Compo': (49, 125, 237), 'Block': (0, 255, 0), 'Text Content': (255, 0, 255)}  # bgr顺序 文本藏蓝色 控件橙色
    img = org_img.copy()
    for ele in eles:
        color = color_map[ele.category]
        ele.visualize_element(img, color, line, draw_rec=draw_rec, gpt4_text_size=gpt4_text_size, gpt4_text_thickness=gpt4_text_thickness)
    img_resize = img
    if shown_resize is not None:
        img_resize = cv2.resize(img, shown_resize)
    if show:
        cv2.imshow(win_name, img_resize)
        cv2.waitKey(wait_key)
        if wait_key == 0:
            cv2.destroyWindow(win_name)
    return img_resize


def save_elements(output_file, elements, img_shape, clean_save=False):  # uied
    # uied
    components = {'compos': [], 'img_shape': img_shape}
    for i, ele in enumerate(elements):
        c = ele.wrap_info()
        # c['id'] = i
        components['compos'].append(c)
    if not clean_save:
        json.dump(components, open(output_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    return components


def reassign_ids(elements):
    for i, element in enumerate(elements):
        element.id = i


def refine_texts(texts, img_shape):
    refined_texts = []
    for text in texts:
        # remove potential noise
        if text.height / img_shape[0] < 0.075:
            refined_texts.append(text)
    return refined_texts

# ocr结果段落合并
def merge_text_line_to_paragraph(elements):
    texts = []
    non_texts = []
    for ele in elements:  # 筛选出elements里面的text控件
        if ele.category == 'Text':
            texts.append(ele)
        else:
            non_texts.append(ele)

    changed = True
    while changed:  # changed的作用：合并文本元素，直到不能再合并为止
        changed = False
        temp_set = []  # 存储合并后的文本元素
        for text_a in texts:
            # print('text_a: ', text_a.text_content)
            merged = False
            for text_b in temp_set:  # 每次和已经放进temp_set的比
                # print('text_b: ', text_b.text_content)
                max_line_gap = max(min(text_a.height, text_b.height) // 2, 7)
                inter_area, _, _, _ = text_a.calc_intersection_area(text_b, bias=(0, max_line_gap))
                if inter_area > 0 and (abs(text_a.col_min - text_b.col_min) < 5 or abs(text_a.col_max - text_b.col_max) < 5 or abs((text_a.col_max-text_a.width/2) - (text_b.col_max-text_b.width/2)) < 5):  # 加上对齐约束
                    text_b.element_merge(text_a)
                    # print('merged b:', text_b.text_content)
                    merged = True
                    changed = True
                    break
            if not merged:  # 如果再没有可以合并的了 那么才把它放入处理过的结果里
                temp_set.append(text_a)
        texts = temp_set.copy()
    return non_texts + texts


def refine_elements(compos, texts, intersection_bias=(2, 2), containment_ratio=0.7):
    '''
    1. remove compos contained in text
    2. remove compos containing text area that's too large
    3. store text in a compo if it's contained by the compo as the compo's text child element
    1. 移除文本中包含的 compos
    2. 删除包含过大文本区域的 compos
    3. 将包含在 compo 中的文本作为 compo 的文本子元素存储在 compo 中
    '''
    elements = []  # 处理后的元素
    contained_texts = []  # 包含文本
    for compo in compos:  # 对每一个控件遍历
        is_valid = True  # 初始化 控件有效
        text_area = 0  # 跟踪控件中文本总面积
        text_in_compo = ''
        if compo.area == 0:
            continue
        for text in texts:
            inter, iou, ioa, iob = compo.calc_intersection_area(text, bias=intersection_bias)  # 计算交集
            if inter > 0:  # ioa分母是控件 iob分母是文本
                # the non-text is contained in the text compo
                if ioa >= 0.5:  # 控件完全包含在文本中（控件的70%被占了） 控件无效 停止对该控件对应文本的检查
                    is_valid = False  # 判决：扔掉控件 留文本
                    break
                text_area += inter  # 控件被占没有到0.7的 先计算累积文本占用面积
                # the text is contained in the non-text compo
                if iob >= containment_ratio and compo.category != 'Block':  # 如果控件占据了单个文本的很大部分（大于0.8）/控件包含了文本 且不是block
                    contained_texts.append(text)  # 判决：保留控件 扔掉文本
                    text_in_compo += text.text_content + ' '
        if is_valid and text_area / compo.area < containment_ratio:  # 如果控件有效 且 文本总面积与控件单独面积比不算很大
            # for t in contained_texts:
            #     t.parent_id = compo.id
            # compo.children += contained_texts
            compo.text_content = text_in_compo[:-1]
            elements.append(compo)  # 保留控件

    # elements += texts
    for text in texts:  # 保留除了刚刚留下的控件的文本
        if text not in contained_texts:
            elements.append(text)
    return elements


def check_containment(elements):
    for i in range(len(elements) - 1):
        for j in range(i + 1, len(elements)):
            relation = elements[i].element_relation(elements[j], bias=(2, 2))
            if relation == -1:
                elements[j].children.append(elements[i])
                elements[i].parent_id = elements[j].id
            if relation == 1:
                elements[i].children.append(elements[j])
                elements[j].parent_id = elements[i].id


def remove_top_bar(elements, img_height):  # 去除顶部任务栏
    new_elements = []
    max_height = img_height * 0.05
    min_row = img_height * 0.03
    for ele in elements:
        if ele.row_min < min_row and ele.height < max_height:
            continue
        new_elements.append(ele)
    return new_elements


def remove_bottom_bar(elements, img_height):  # 去除底部导航栏
    new_elements = []
    max_height = img_height * 0.05
    min_row = img_height * 0.95
    for ele in elements:
        # parameters for 800-height GUI
        if ele.row_min > min_row and ele.height <= max_height:
            continue
        new_elements.append(ele)
    return new_elements


def compos_clip_and_fill(clip_root, org, compos):
    def most_pix_around(pad=6, offset=2):
        '''
        determine the filled background color according to the most surrounding pixel
        '''
        up = row_min - pad if row_min - pad >= 0 else 0
        left = col_min - pad if col_min - pad >= 0 else 0
        bottom = row_max + pad if row_max + pad < org.shape[0] - 1 else org.shape[0] - 1
        right = col_max + pad if col_max + pad < org.shape[1] - 1 else org.shape[1] - 1
        most = []
        for i in range(3):
            val = np.concatenate((org[up:row_min - offset, left:right, i].flatten(),
                            org[row_max + offset:bottom, left:right, i].flatten(),
                            org[up:bottom, left:col_min - offset, i].flatten(),
                            org[up:bottom, col_max + offset:right, i].flatten()))
            most.append(int(np.argmax(np.bincount(val))))
        return most

    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)

    bkg = org.copy()
    cls_dirs = []
    for compo in compos:
        cls = compo['class']
        if cls == 'Background':
            compo['path'] = pjoin(clip_root, 'bkg.png')
            continue
        c_root = pjoin(clip_root, cls)
        c_path = pjoin(c_root, str(compo['id']) + '.jpg')
        compo['path'] = c_path
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)

        position = compo['position']
        col_min, row_min, col_max, row_max = position['column_min'], position['row_min'], position['column_max'], position['row_max']
        cv2.imwrite(c_path, org[row_min:row_max, col_min:col_max])
        # Fill up the background area
        cv2.rectangle(bkg, (col_min, row_min), (col_max, row_max), most_pix_around(), -1)
    cv2.imwrite(pjoin(clip_root, 'bkg.png'), bkg)


def merge(img_path, compo_path, text_path, merge_root=None, is_paragraph=True, ocr_only=True, is_remove_bar=True, clean_save=False, 
          text_js={}, ip_js={}, show=False, wait_key=0, draw_rec=True, img_4gpt4_out_path='data/screenshot/screenshot_gpt4.png', gpt4_text_size=0.55, gpt4_text_thickness=2):
    if (clean_save and not ocr_only) or not draw_rec:
        text_json = text_js
        compo_json = ip_js
    elif not clean_save and ocr_only:
        text_json = json.load(open(text_path, 'r'))
        org, grey = pre.read_img(img_path, 800)
        img_shape = [org.shape[0], org.shape[1], org.shape[2]]
        compo_json = {"img_shape": img_shape, 'compos': []}
    elif clean_save and ocr_only:
        text_json = text_js
        org, grey = pre.read_img(img_path, 800)
        img_shape = [org.shape[0], org.shape[1], org.shape[2]]
        compo_json = {"img_shape": img_shape, 'compos': []}
    else:
        compo_json = json.load(open(compo_path, 'r'))  # 从两个JSON文件中加载元素信息：compo_path 中包含图像组件（compos），text_path 中包含文本元素（texts）
        text_json = json.load(open(text_path, 'r'))

    # load text and non-text compo
    ele_id = 0
    compos = []  # 把文本和非文本组件加入列表
    for compo in compo_json['compos']:
        element = Element(ele_id, (compo['column_min'], compo['row_min'], compo['column_max'], compo['row_max']), compo['class'], compo['sub_class'])
        compos.append(element)
        ele_id += 1
    texts = []
    for text in text_json['texts']:
        element = Element(ele_id, (text['column_min'], text['row_min'], text['column_max'], text['row_max']), 'Text', 'Text', text_content=text['content'])
        texts.append(element)
        ele_id += 1
    if compo_json['img_shape'] != text_json['img_shape']:  # 如果图像的形状（尺寸）不匹配，它会根据比例调整文本元素的大小，以使它们与组件元素具有相同的图像形状
        resize_ratio = compo_json['img_shape'][0] / text_json['img_shape'][0]
        for text in texts:
            text.resize(resize_ratio)

    # check the original detected elements
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (compo_json['img_shape'][1], compo_json['img_shape'][0]))
    # 显示检测到的元素 查看初始布局
    # show_elements(img_resize, texts + compos, show=show, win_name='all elements before merging', wait_key=wait_key)

    # refine elements 各种修正
    texts = refine_texts(texts, compo_json['img_shape'])
    elements = refine_elements(compos, texts)  # 控件和文本对齐操作
    if is_remove_bar and min(cv2.imread(img_path).shape[0], cv2.imread(img_path).shape[1])<1200:  # 去除顶部底部bar
        elements = remove_top_bar(elements, img_height=compo_json['img_shape'][0])
        # elements = remove_bottom_bar(elements, img_height=compo_json['img_shape'][0])
    if is_paragraph:  # 段落合并
        elements = merge_text_line_to_paragraph(elements)
    reassign_ids(elements)  # 元素id更新
    check_containment(elements)  # 包含关系更新
    board = show_elements(img_resize, elements, show=show, win_name='elements after merging', wait_key=wait_key, draw_rec=draw_rec, gpt4_text_size=gpt4_text_size, gpt4_text_thickness=gpt4_text_thickness)

    # save all merged elements, clips and blank background 保存文件
    name = img_path.replace('\\', '/').split('/')[-1][:-4]
    if draw_rec:
        components = save_elements(pjoin(merge_root, name + '.json'), elements, img_resize.shape, clean_save=clean_save)  # json输出
    else:  # 给gpt4的无框输出-json
        components = save_elements(img_4gpt4_out_path[:-4] + '.json', elements, img_resize.shape, clean_save=clean_save)  # json输出

    if not clean_save and draw_rec:  # 普通输出
        cv2.imwrite(pjoin(merge_root, name + '.jpg'), board)  # 图输出
    elif not draw_rec:  # 给gpt4的无框输出-图片
        cv2.imwrite(img_4gpt4_out_path, board)  # 图输出

    # if not clean_save:
    #     print('[Merge Completed] Input: %s Output: %s' % (img_path, pjoin(merge_root, name + '.jpg')))
    # else:
    #     print('[Merge Completed] Input: %s Output: clean_save no output' % img_path)  # 打印信息

    return board, components
