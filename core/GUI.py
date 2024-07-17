# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import cv2
import os
import json
import numpy as np
import time
from os.path import join as pjoin
import element.detect_text.text_detection as text
import element.detect_compo.ip_region_proposal as ip
import element.detect_merge.merge as merge
from layout.obj.Compos_DF import ComposDF
from layout.obj.Compo import *
from layout.obj.Block import *
from layout.obj.List import *
import layout.lib.draw as draw

class GUI:
    def __init__(self, img_file, det_img, line_img, compos_json_file=None, output_dir='data/output'):
        self.img_file = img_file
        self.img = cv2.imread(img_file)
        self.img_reshape = self.img.shape
        self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        self.file_name = img_file.replace('\\', '/').split('/')[-1][:-4]

        self.output_dir = output_dir
        self.ocr_dir = pjoin(self.output_dir, 'ocr') if output_dir is not None else None
        self.non_text_dir = pjoin(self.output_dir, 'ip') if output_dir is not None else None
        self.merge_dir = pjoin(self.output_dir, 'uied') if output_dir is not None else None
        self.layout_dir = pjoin(self.output_dir, 'layout') if output_dir is not None else None
        self.workflow_dir = pjoin(self.output_dir, 'workflow') if output_dir is not None else None

        self.compos_json = None  # {'img_shape':(), 'compos':[]} 存放记录检测到的组件的json
        self.compos_df = None    # dataframe for efficient processing
        self.compos = []         # list of Compo objects
        self.detection_result_img = {'text': None, 'non-text': None, 'merge': None}   # visualized detection result 可视化识别结果

        self.det_img = det_img    # 目标检测图片
        self.line_img = line_img  # 线框图片

        self.layout_result_img_all = None     # 输出图2（图1UIED）: 所有的控件（灰色框）+ 彩色 橙非文本蓝文本 上号下字
        self.layout_result_img_block = None      # 输出图3：block蓝色块 merge蓝色框 其他元素灰色 带号

        self.layout_result_img_group = None     # visualize group of compos with repetitive layout
        self.layout_result_img_pair = None      # visualize paired groups
        self.layout_result_img_list = None      # visualize list (paired group) boundary

        self.lists = []     # list of List objects representing lists
        self.blocks = []    # list of Block objects representing blocks

    def save_layout_result_imgs(self):  # 输出三个图片
        os.makedirs(self.layout_dir, exist_ok=True)
        cv2.imwrite(pjoin(self.layout_dir, self.file_name + '-group.jpg'), self.layout_result_img_group)
        cv2.imwrite(pjoin(self.layout_dir, self.file_name + '-pair.jpg'), self.layout_result_img_pair)
        cv2.imwrite(pjoin(self.layout_dir, self.file_name + '-list.jpg'), self.layout_result_img_list)
        # print('Layout recognition result images save to ', output_dir)

    def save_final_result_imgs(self, workflow_only=True):  # 输出三个图片
        os.makedirs(self.workflow_dir, exist_ok=True)
        if not workflow_only:
            os.makedirs(self.layout_dir, exist_ok=True)
            cv2.imwrite(pjoin(self.layout_dir, self.file_name + '-all_compos.jpg'), self.layout_result_img_all)
            cv2.imwrite(pjoin(self.layout_dir, self.file_name + '-block.jpg'), self.layout_result_img_block)
        reshape_w, reshape_h = self.detection_result_img['text'].shape[1], self.detection_result_img['text'].shape[0]
        cv2.imwrite(pjoin(self.workflow_dir, self.file_name + '-block.jpg'), self.layout_result_img_block) # 输出block
        cv2.imwrite(pjoin(self.workflow_dir, self.file_name + '-workflows.jpg'),
                    np.concatenate([cv2.resize(self.line_img, (reshape_w, reshape_h)),
                                    cv2.resize(self.det_img, (int(reshape_w + reshape_w * 0.25), reshape_h)),
                                    self.detection_result_img['text'], self.detection_result_img['non-text'],
                                    self.detection_result_img['merge'], self.layout_result_img_all, self.layout_result_img_block], axis=1))  # 输出结果拼图


    def save_layout_result_json(self, clean_save=False, layout_json_dir=''):  # block的 最后要用的json
        os.makedirs(self.layout_dir, exist_ok=True)
        js = ['alignment: v']
        for block in self.blocks:
            js.append(block.wrap_info())

        def update_location(node):
            if isinstance(node, list):
                # 如果是列表，遍历其中的每个元素
                for item in node:
                    update_location(item)
            elif isinstance(node, dict):
                # 如果是字典，检查是否存在"location"键
                if "location" in node:
                    # 更新"location"键中的四个值
                    location = node["location"]
                    height_resized = 800 / self.img.shape[0] * self.img.shape[1]
                    location["left"] = int(location["left"] / height_resized * self.img.shape[1])
                    location["right"] = int(location["right"] / height_resized * self.img.shape[1])
                    location["top"] = int(location["top"] / 800 * self.img.shape[0])
                    location["bottom"] = int(location["bottom"] / 800 * self.img.shape[0])
                # 遍历字典中的值
                for key, value in node.items():
                    update_location(value)

        # 调用函数来更新JSON数据
        update_location(js)
        if not clean_save:
            json.dump(js, open(pjoin(self.workflow_dir, self.file_name + '.json'), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        # else:  # 存储到指定路径，不会嵌套文件夹
        #     json.dump(js, open(pjoin(layout_json_dir, self.file_name + '.json'), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        # print('Layout recognition result json save to ', output_dir)
        return js

    def save_list(self):  # list的json 用处不大大
        os.makedirs(self.layout_dir, exist_ok=True)
        js = {'ui': self.file_name, 'list': [], 'multitab': []}
        for lst in self.lists:
            js['list'].append(lst.wrap_list_items())
        json.dump(js, open(pjoin(self.layout_dir, self.file_name + '-list.json'), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)  # list的

    def save_detection_result(self):  # 更新uied的json
        if not os.path.exists(pjoin(self.merge_dir, self.file_name + '.jpg')):
            os.makedirs(self.ocr_dir, exist_ok=True)
            os.makedirs(self.non_text_dir, exist_ok=True)
            os.makedirs(self.merge_dir, exist_ok=True)
            cv2.imwrite(pjoin(self.ocr_dir, self.file_name + '.jpg'), self.detection_result_img['text'])
            cv2.imwrite(pjoin(self.non_text_dir, self.file_name + '.jpg'), self.detection_result_img['non-text'])
            cv2.imwrite(pjoin(self.merge_dir, self.file_name + '.jpg'), self.detection_result_img['merge'])
        if not os.path.exists(pjoin(self.merge_dir, self.file_name + '.json')):
            json.dump(self.compos_json, open(pjoin(self.merge_dir, self.file_name + '.json'), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    def save_layout_result(self, clean_save=False, layout_json_dir='', workflow_only=True):
        if not clean_save and not workflow_only:
            self.save_detection_result()  # 更新uied里面的json和图片
            # self.save_layout_result_imgs()  # 保存layout图片
            self.save_final_result_imgs(workflow_only=workflow_only)  # 保存输出图片
            # self.save_list()  # 保存list
        elif workflow_only:
            self.save_final_result_imgs(workflow_only=workflow_only)  # 保存输出图片
        result_js = self.save_layout_result_json(clean_save=clean_save, layout_json_dir=layout_json_dir)  # 保存layoutjson
        return result_js

    '''
    *****************************
    *** GUI Element Detection ***
    *****************************
    '''
    def resize_by_height(self, img_resize_longest_side=800):
        height, width = self.img.shape[:2]
        # if height > width:
        width_re = int(img_resize_longest_side * (width / height))
        return img_resize_longest_side, width_re, self.img.shape[2]
        # else:
        #     height_re = int(img_resize_longest_side * (height / width))
        #     return height_re, img_resize_longest_side, self.img.shape[2]

    def detect_element(self, col_lines, row_lines, is_ocr=True, is_non_text=True, is_merge=True, bbox_detect=[], sub_class_detect=[],
                       img_resize_longest_side=800, ocr_save='', clean_save=False, ocr_only=True, workflow_only=True, ocr=None, 
                       accurate_ocr=True, show=False, draw_rec=True, img_4gpt4_out_path='data/screenshot/screenshot_gpt4.png', 
                       gpt4_text_size=0.55, gpt4_text_thickness=2):
        if workflow_only:
            clean_save = True
        if self.img_file is None:  # 看是不是读到照片了
            print('No GUI image is input')
            return
        # 在检测非文本元素的同时，按最长边调整图形用户界面图像的大小
        if img_resize_longest_side is not None:
            self.img_reshape = self.resize_by_height(img_resize_longest_side)  # 按比例缩放 现默认长边是800 889*520-->800*467
            self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
            resize_height = self.img_reshape[0]  # 缩放后高800
        else:
            self.img_reshape = self.img.shape
            self.img_resized = self.img.copy()
            resize_height = None
        text_js, ip_js = {}, {}
        key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': False,
                      'max-word-inline-gap': 10, 'max-line-ingraph-gap': 4, 'remove-ui-bar': True}  # 超参数？ 用于非文本compos检测

        if is_ocr:  # 如果要检测ocr就检测 否则从对应文件夹里直接读取之前检测过的图片
            if draw_rec:  # 非gpt4—ele-only版本 正常
                self.detection_result_img['text'], text_js = text.text_detection(self.img_file, self.ocr_dir, ocr_save=ocr_save, clean_save=clean_save, accurate_ocr=accurate_ocr, ocr=ocr, show=show)  # import element.detect_text.text_detection as text 改成pdpdocr的地方
            else:  # gpt4—ele-only版本 干净输出
                self.detection_result_img['text'], text_js = text.text_detection(self.img_file, self.ocr_dir,
                                                                                 ocr_save=ocr_save,
                                                                                 clean_save=True,
                                                                                 accurate_ocr=accurate_ocr, ocr=ocr,
                                                                                 show=show)  # import element.detect_text.text_detection as text 改成pdpdocr的地方
        elif os.path.isfile(pjoin(self.ocr_dir, self.file_name + '.jpg')):
            self.detection_result_img['text'] = cv2.imread(pjoin(self.ocr_dir, self.file_name + '.jpg'))

        if not ocr_only:  # 如果只要输出ocr的版本
            if is_non_text:
                if draw_rec:  # 非gpt4—ele-only版本 正常
                    self.detection_result_img['non-text'], ip_js = ip.compo_detection(self.img_file, self.non_text_dir, key_params, bbox_detect, sub_class_detect, col_lines, row_lines, resize_by_height=resize_height, clean_save=clean_save, show=show)  # import element.detect_compo.ip_region_proposal as ip
                else:  # gpt4—ele-only版本 干净输出
                    self.detection_result_img['non-text'], ip_js = ip.compo_detection(self.img_file, self.non_text_dir,
                                                                                      key_params, bbox_detect,
                                                                                      sub_class_detect, col_lines,
                                                                                      row_lines,
                                                                                      resize_by_height=resize_height,
                                                                                      clean_save=True,
                                                                                      show=show)  # import element.detect_compo.ip_region_proposal as ip
            elif os.path.isfile(pjoin(self.non_text_dir, self.file_name + '.jpg')):
                self.detection_result_img['non-text'] = cv2.imread(pjoin(self.non_text_dir, self.file_name + '.jpg'))

        if is_merge:  # 非文本和文本修正 区分 去重复
            os.makedirs(self.merge_dir, exist_ok=True)
            compo_path = pjoin(self.non_text_dir, self.file_name + '.json')
            ocr_path = pjoin(self.ocr_dir, self.file_name + '.json')
            self.detection_result_img['merge'], self.compos_json = merge.merge(self.img_file, compo_path, ocr_path, self.merge_dir,
                                                                               ocr_only=ocr_only, is_remove_bar=True, is_paragraph=True, 
                                                                               clean_save=clean_save, text_js=text_js, ip_js=ip_js, show=show, 
                                                                               draw_rec=draw_rec, img_4gpt4_out_path=img_4gpt4_out_path
                                                                               , gpt4_text_size=gpt4_text_size, gpt4_text_thickness=gpt4_text_thickness)  # import element.detect_merge.merge as merge
        return self.compos_json


    def load_detection_result(self):
        '''
        Load json detection result from json file
        '''
        self.compos_json = json.load(open(pjoin(self.merge_dir, self.file_name + '.json')))
        self.img_reshape = self.compos_json['img_shape']
        self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        self.draw_element_detection()

    def load_compos(self, compos):
        '''
        Load compos from objects: {'img_shape':(), 'compos':[]}
        '''
        self.compos_json = compos.copy()
        self.img_reshape = self.compos_json['img_shape']
        self.img_resized = cv2.resize(self.img, (self.img_reshape[1], self.img_reshape[0]))
        self.draw_element_detection()

    '''
    **************************
    *** Layout Recognition ***
    **************************
    '''
    # *** step1 ***
    def cvt_compos_json_to_dataframe(self):
        '''
        Represent the components using a Pandas DataFrame for the sake of processing
        为便于处理，使用 Pandas DataFrame 表示组件
        '''
        self.compos_df = ComposDF(json_data=self.compos_json, gui_img=self.img_resized.copy())

    # *** step2 ***
    def interface_interpretor(self, model_cls, preprocess, device, real_labels_dict, labels_for_read, col_lines, row_lines, lang='zh'):
        '''
        分组？阿尼哟
        '''
        # st_time = time.time
        self.compos_df.compos_dataframe.loc[(self.compos_df.compos_dataframe['sub_class'] == 'switch') & (
                    self.compos_df.compos_dataframe['text_content'].str.len() > 0), ['class', 'sub_class']] \
            = ['Text', 'Text']  # 一开始进来有文字的switch是错的
        self.compos_df.ori_compos_dataframe = self.compos_df.compos_dataframe.copy()
        self.compos_df.line_split(col_lines, row_lines)  # 一、根据线框检测规范group 规范block
        # print('line_split: ', (time.time()-st_time))
        # st_time = time.time()
        self.compos_df.q_search_recover(lang=lang)  # 二、修复误识别的搜索放大键
        # print('q_search_recover: ', (time.time()-st_time))
        # st_time = time.time()
        self.compos_df.compos_dataframe['pair_to'] = -1  # 先加一列
        self.compos_df.check_ratio_anno(lang=lang)  # 三、多选框+选项的注释
        self.compos_df.icon_cls_by_clip(self.img, model_cls, preprocess, device, real_labels_dict, labels_for_read, lang=lang)  # 四、在同一个block里 匹配图标和文字（横向和纵向） + CLIP找回独立图标含义
        # print('icon_cls_by_clip: ', (time.time()-st_time))
        # st_time = time.time()
        self.compos_df.get_back_edittext(lang=lang)  # 五、找回编辑框（输入框）
        # print('get_back_edittext: ', (time.time()-st_time))
        # st_time = time.time()
        self.compos_df.find_caption(lang=lang)  # 六、标题的发现 每个线框里第一高的无关联的text为标题 在text_content前面加标题
        # print('find_caption: ', (time.time()-st_time))
        # st_time = time.time()
        self.compos_df.find_selected_tab(self.img, lang=lang)  # 七、选项卡选中项发现 同一线框里有3~6个均匀分布的同行（center_row差距10以内）时触发
        # print('find_selected_tab: ', (time.time()-st_time))
        # st_time = time.time()
        self.compos_df.switch_name(lang=lang)  # 八、匹配switch名（可简化）
        # print('switch_name: ', (time.time()-st_time))
        # st_time = time.time()
        self.compos_df.other_sub_class_text(lang=lang)  # 九、这里加一个除了buttonicon和text 其他类的、没有文本内容的专属名称
        # print('other_sub_class_text: ', (time.time()-st_time))
        self.compos_df.compos_dataframe['text_content'] = self.compos_df.compos_dataframe['text_content'].str.replace('_', '')  # 把用于标记的下划线去掉
        self.compos_df.compos_dataframe['text_content'] = self.compos_df.compos_dataframe['text_content'].str.replace('\'', '')  # 把单引号去掉
        self.compos_df.compos_dataframe['text_content'] = self.compos_df.compos_dataframe['text_content'].str.replace('键键', '键')  # 去掉重复

    # *** step3 ***
    def cvt_groups_to_list_compos(self):
        '''
        Represent the recognized perceptual groups as List objects
        将已识别的感知组表示为列表对象
        '''
        df = self.compos_df.compos_dataframe

        self.lists = []
        self.compos = []

        # 处理未包含在列表中的其余组件
        for i in range(len(df)):
            compo_df = df.iloc[i]
            pair_to_id = str(compo_df['pair_to']) if compo_df['pair_to'] != -1 else ''  # 标注pair_to信息
            self.compos.append(Compo(compo_id='c-' + str(compo_df['id']), compo_class=compo_df['class'], compo_sub_class=compo_df['sub_class'], compo_df=compo_df, compo_pair_to=pair_to_id))
        # regard the list as a type of component in the GUI
        self.compos += self.lists


    def slice_hierarchical_block(self, clean_save=False):
        '''
        根据已识别的 Compos List 和线框信息 将图形用户界面划分为不同的层次块
        '''
        # 先排序
        self.compos_df.compos_dataframe['line_split'] = self.compos_df.compos_dataframe['line_split_c'].copy()
        self.compos_df.compos_dataframe['line_split_c'] = self.compos_df.compos_dataframe['line_split_c'].str.split('_', expand=True)[1].astype(int)
        self.compos_df.compos_dataframe['line_split_r'] = self.compos_df.compos_dataframe['line_split_r'].astype(int)
        self.compos_df.compos_dataframe.sort_values(['line_split_r', 'line_split_c'], inplace=True)

        if not clean_save:
            # 画图：记录block
            self.compos_df.compos_dataframe.insert(self.compos_df.compos_dataframe.shape[1], 'block_base', '')
            self.compos_df.compos_dataframe.insert(self.compos_df.compos_dataframe.shape[1], 'block_merge', '')

        # 1.遍历self.compos 得到array格式的line_split和line_merge_c信息
        line_split_all, line_merge_c_all = [], []
        for compo in self.compos:
            if compo.compo_id.split('-')[0] != 'c':
                line_split_all.append(self.compos_df.compos_dataframe.loc[compo.compo_df['id'].iloc[0], 'line_split'])
                line_merge_c_all.append(self.compos_df.compos_dataframe.loc[compo.compo_df['id'].iloc[0], 'line_merge_c'])
            else:
                line_split_all.append(self.compos_df.compos_dataframe.loc[compo.compo_df['id'], 'line_split'])
                line_merge_c_all.append(self.compos_df.compos_dataframe.loc[compo.compo_df['id'], 'line_merge_c'])
        line_split_all, line_merge_c_all = np.array(line_split_all), np.array(line_merge_c_all)

        # 2.根据line_split的排序来逐个block打包
        self.compos_df.compos_dataframe = self.compos_df.compos_dataframe.sort_values(['center_row', 'center_column'])
        block_id = 0
        line_split_uni = self.compos_df.compos_dataframe['line_split'].unique()
        blocks_all, blocks_for_merge = [], []
        for line_split in line_split_uni:
            compos_idx_in_block = np.where(line_split_all == line_split)[0]
            if len(compos_idx_in_block) != 0:
                block_compos = [self.compos[blocked_idx] for blocked_idx in compos_idx_in_block]  # 打包同一块内的
                blocks_for_merge.append(line_merge_c_all[compos_idx_in_block[0]])  # 记录本block是否需要合并
                if len(block_compos) > 1:  # 大于1才使用blocks打包
                    cr = [self.compos[blocked_idx].center_row for blocked_idx in compos_idx_in_block]
                    if (max(cr) - min(cr)) < 30:  # 差不多一行
                        direction = 'h'
                    else:
                        direction = 'v'
                    # 日历默认方向垂直
                    # ''.join(self.compos_df.compos_dataframe['text_content'].isin(['廿', 'calendars', 'date'])
                    if '廿' in ''.join(self.compos_df.compos_dataframe['text_content']) or 'calendars' in ''.join(self.compos_df.compos_dataframe['text_content'].str.lower()):
                        direction = 'v'
                    new_block = Block(id='b-' + str(block_id), compos=block_compos, slice_sub_block_direction=direction)
                    if not clean_save:
                        # 画图：记录基本block
                        self.compos_df.compos_dataframe.loc[self.compos_df.compos_dataframe['line_split'] == line_split, 'block_base'] = 'b-' + str(block_id)
                    blocks_all.append(new_block)
                    block_id = new_block.next_block_id

                else:
                    blocks_all.append(block_compos[0])

        # 3. block merge 合并部分！
        blocked_all_merged = []
        last_merge_id, merged_block_buffer = -1, []
        for bid in range(len(blocks_all)):
            curr_merge_id = blocks_for_merge[bid]  # 目前合并id更新
            if curr_merge_id == -1:  # 1. 当前块不需要合并
                if len(merged_block_buffer) != 0:  # 如果是前面累积了很多要合并的block的情况 那么在这里一次性送入block
                    new_block = Block(id='b-' + str(block_id), compos=merged_block_buffer, slice_sub_block_direction='v', merge=True)
                    blocked_all_merged.append(new_block)
                    if not clean_save:
                        # 画图：记录合并
                        self.compos_df.compos_dataframe.loc[self.compos_df.compos_dataframe['block_base'].isin(
                            [cp_bk.block_id for cp_bk in merged_block_buffer if 'block_id' in cp_bk.__dir__()]), 'block_merge'] = 'b-' + str(block_id)
                        self.compos_df.compos_dataframe.loc[[int(cp_bk.compo_id.split('-')[-1]) for cp_bk in merged_block_buffer
                                                             if 'compo_id' in cp_bk.__dir__()], 'block_merge'] = 'b-' + str(block_id)
                    block_id = new_block.next_block_id
                    merged_block_buffer = []  # 清空缓冲区
                blocked_all_merged.append(blocks_all[bid])  # 然后把当前的不要合并的block送入
            elif curr_merge_id != -1 and last_merge_id == curr_merge_id:  # 2.当前块需要接着合并
                merged_block_buffer.append(blocks_all[bid])  # 压入缓冲区
            elif curr_merge_id != -1 and last_merge_id != curr_merge_id:  # 3.当前块需要开启一次新的合并 且之前有不同的合并序号
                if len(merged_block_buffer) != 0:  # 如果是前面累积了很多要合并的block的情况 那么在这里也一次性送入block
                    new_block = Block(id='b-' + str(block_id), compos=merged_block_buffer, slice_sub_block_direction='v', merge=True)
                    blocked_all_merged.append(new_block)
                    if not clean_save:
                        # 画图：记录合并
                        self.compos_df.compos_dataframe.loc[self.compos_df.compos_dataframe['block_base'].isin(
                            [cp_bk.block_id for cp_bk in merged_block_buffer if 'block_id' in cp_bk.__dir__()]), 'block_merge'] = 'b-' + str(block_id)
                        self.compos_df.compos_dataframe.loc[[int(cp_bk.compo_id.split('-')[-1]) for cp_bk in merged_block_buffer
                                                             if 'compo_id' in cp_bk.__dir__()], 'block_merge'] = 'b-' + str(block_id)
                    block_id = new_block.next_block_id
                    merged_block_buffer = []  # 清空缓冲区
                merged_block_buffer.append(blocks_all[bid])  # 然后把当前要合并的block压入缓冲区
            last_merge_id = curr_merge_id  # 更新上一次合并id
        if len(merged_block_buffer) != 0:  # 如果循环结束缓冲区还有 那么把剩余的送完
            new_block = Block(id='b-' + str(block_id), compos=merged_block_buffer, slice_sub_block_direction='v', merge=True)
            blocked_all_merged.append(new_block)
            if not clean_save:
                # 画图：记录合并
                self.compos_df.compos_dataframe.loc[self.compos_df.compos_dataframe['block_base'].isin(
                    [cp_bk.block_id for cp_bk in merged_block_buffer if 'block_id' in cp_bk.__dir__()]), 'block_merge'] = 'b-' + str(block_id)
                self.compos_df.compos_dataframe.loc[[int(cp_bk.compo_id.split('-')[-1]) for cp_bk in merged_block_buffer
                                                     if 'compo_id' in cp_bk.__dir__()], 'block_merge'] = 'b-' + str(block_id)
        self.blocks = blocked_all_merged


    # entry method
    def recognize_layout(self, model_cls, preprocess, device, real_labels_dict, labels_for_read, col_lines, row_lines,
                         is_save=True, clean_save=False, layout_json_dir='', workflow_only=True, lang='zh', return_df=False):
        start_t = time.time()
        self.cvt_compos_json_to_dataframe()  # 识别元素装入df
        self.interface_interpretor(model_cls, preprocess, device, real_labels_dict, labels_for_read, col_lines, row_lines, lang=lang)  # 界面解析（NEW!）
        self.cvt_groups_to_list_compos()  # List ———— 把list格式化存到self.compos里
        self.slice_hierarchical_block(clean_save)  # Block ———— 层次识别 block识别嵌套识别
        # if not clean_save:
        #     self.get_layout_result_imgs()  # layout的图片输出
        if not clean_save:
            self.get_final_result_imgs()  # 结果图片输出 uied（已有）all所有控件信息 block划分
        result_js = []
        if is_save:
            result_js = self.save_layout_result(clean_save=clean_save, layout_json_dir=layout_json_dir, workflow_only=workflow_only)
        if not clean_save:
            print("[Layout Recognition Completed in %.3f s] Input: %s Output: %s" % (time.time() - start_t, self.img_file, pjoin(self.layout_dir, self.file_name + '.json')))
        else:
            print("[Layout Recognition Completed in %.3f s] Input: %s Output: %s" % (time.time() - start_t, self.img_file, pjoin(layout_json_dir, self.file_name + '.json')))
        # print(time.ctime(), '\n\n')
        if return_df:
            return self.compos_df.compos_dataframe
        else:
            return result_js

    '''
    *********************
    *** Visualization ***
    *********************
    '''

    # —————————————————新版最终可视化函数———————————————————————

    def get_final_result_imgs(self):  # 获得结果图片（总） 在这里画图
        self.layout_result_img_all = self.visualize_result_img_all()  # 输出图2（图1UIED）: 所有的控件（灰色框）+ 彩色 橙非文本蓝文本 上号下字
        self.layout_result_img_block = self.visualize_result_img_block()  # 输出图3：block蓝色块 merge蓝色线框 其他元素灰色 带号

    def visualize_result_img_all(self):
        # 因为有中文所以需要处理
        font_path = '/data4/bianyiheng/simhei.ttf'
        font_size = 12
        font = ImageFont.truetype(font_path, font_size)
        board = self.img_resized.copy()  # 新建画布 所有的控件（灰色框）+ 彩色 橙非文本蓝文本 上号下字
        top_add = 15
        board = cv2.copyMakeBorder(board, int(top_add), 0, 0, int(board.shape[1] * 0.4), cv2.BORDER_CONSTANT, value=(255, 255, 255))  # 加一个白边
        color_map = {'Text': (196, 114, 68), 'Compo': (49, 125, 237), 'Block': (196, 114, 68)}  # bgr顺序 文本藏蓝色 控件橙色 容器老蓝色
        removed_idx = list(set(self.compos_df.ori_compos_dataframe.index) - set(self.compos_df.compos_dataframe.index))  # 提取被删掉的部分
        # 1. 灰色的部分 被删掉的
        for ridx in removed_idx:
            compo = self.compos_df.ori_compos_dataframe.loc[ridx]
            board = cv2.rectangle(board, (compo.column_min, int(compo.row_min + top_add)), (compo.column_max, int(compo.row_max + top_add)), (127, 127, 127), 1)  # 删掉的画灰色
            board = cv2.putText(board, str(compo.id), (int(compo.column_min), int(compo.row_min - 3 + top_add)), cv2.FONT_HERSHEY_PLAIN, 0.6, (127, 127, 127))  # 标个号

        # 2. 彩色的部分 没有被删掉
        for idx in self.compos_df.compos_dataframe.index:
            compo = self.compos_df.compos_dataframe.loc[idx]
            board = cv2.rectangle(board, (compo.column_min, int(compo.row_min + top_add)), (compo.column_max, int(compo.row_max + top_add)), color_map[compo['class']], 2)  # 保留框画彩色
            board = cv2.putText(board, str(compo.id) + '-', (int(compo.column_min), int(compo.row_min - 3 + top_add)), cv2.FONT_HERSHEY_PLAIN, 0.6, color_map[compo['class']])  # 标个号
        # 这里有中文：需要换一个包 这里写彩色文本内容
        img_pil = Image.fromarray(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for idx in self.compos_df.compos_dataframe.index:
            compo = self.compos_df.compos_dataframe.loc[idx]
            draw.text((int(compo.column_min + 19), int(compo.row_min - 14 + top_add)), compo['text_content'], font=font, fill=color_map[compo['class']][::-1])
            # board = cv2.putText(board, compo.text_content, (int(compo.column_min + 20), int(compo.row_min - 2)), cv2.FONT_HERSHEY_PLAIN, 0.9, color_map[compo['class']])  # 标个内容
        board = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # 转回去
        board = cv2.resize(board, (board.shape[1], self.img_reshape[0]))
        # # 画出来看看
        # plt.figure(figsize=(9, 16))
        # plt.imshow(board[:, :, ::-1])
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
        return board

    def visualize_result_img_block(self):
        board = self.img_resized.copy()  # block蓝色块 merge蓝色线框 其他元素灰色 带号
        # board = cv2.copyMakeBorder(board, 0, 0, 0, int(board.shape[1] / 3), cv2.BORDER_CONSTANT, value=(255, 255, 255))  # 加一个白边
        # 1. 先画灰色 所有的控件：
        for idx in self.compos_df.compos_dataframe.index:
            compo = self.compos_df.compos_dataframe.loc[idx]
            board = cv2.rectangle(board, (compo.column_min, compo.row_min), (compo.column_max, compo.row_max), (127, 127, 127), 2)  # 删掉的画灰色
            board = cv2.putText(board, str(compo.id), (int(compo.column_min), int(compo.row_min - 2)), cv2.FONT_HERSHEY_PLAIN, 1, (127, 127, 127))  # 标个号
        # 2. block蓝色半透明框
        overlay = board.copy()  # 复制图层 再合并
        alpha = 0.2  # 透明度
        for block in self.compos_df.compos_dataframe['block_base'].unique():
            if block != '':
                block_df = self.compos_df.compos_dataframe.loc[self.compos_df.compos_dataframe['block_base'] == block]
                c_min, c_max, r_min, r_max = min(block_df['column_min']), max(block_df['column_max']), \
                                             min(block_df['row_min']), max(block_df['row_max'])
                cv2.rectangle(overlay, (c_min, r_min), (c_max, r_max), (251, 168, 3), -1)  # 在半透明图层画块
                cv2.putText(board, block, (int(c_max - 40), int(r_max + 12)), cv2.FONT_HERSHEY_PLAIN, 1, (251, 168, 3))  # 标个block号
        cv2.addWeighted(overlay, alpha, board, 1 - alpha, 0, board)
        # 3. merge的蓝色虚线
        for block in self.compos_df.compos_dataframe['block_merge'].unique():
            if block != '':
                block_df = self.compos_df.compos_dataframe.loc[self.compos_df.compos_dataframe['block_merge'] == block]
                c_min, c_max, r_min, r_max = min(block_df['column_min']), max(block_df['column_max']), \
                                             min(block_df['row_min']), max(block_df['row_max'])
                cv2.rectangle(board, (c_min, r_min), (c_max, r_max), (251, 168, 3), 2)
                cv2.putText(board, block, (int(c_max - 50), int(r_max + 15)), cv2.FONT_HERSHEY_PLAIN, 1.3, (251, 168, 3))  # 标个block号
        # # 画出来看看
        # plt.figure(figsize=(9, 16))
        # plt.imshow(board[:, :, ::-1])
        # plt.show()
        return board

    def visualize_final_imgs(self):
        I1 = cv2.resize(self.detection_result_img['merge'], (self.img_reshape[1], self.img_reshape[0]))   # 组pair和list的 仅显示函数
        I2 = cv2.resize(self.layout_result_img_all, (self.img_reshape[1], self.img_reshape[0]))
        I3 =cv2.resize(self.layout_result_img_block, (self.img_reshape[1], self.img_reshape[0]))
        img = np.concatenate([I1, I2, I3], axis=1)[:, :, ::-1]  # 拼一起显示 使用plt
        plt.figure(figsize=(27, 16))
        plt.imshow(img)
        plt.title(' All Compos            Filtered               Blocks', fontdict={'weight': 'normal', 'size': 60})
        plt.show()
    # ——————————————————组pair和list的————————————————————————

    def get_layout_result_imgs(self):  # 组pair和list的 绘图函数（总）
        self.layout_result_img_group = self.visualize_compos_df('group', show=False)  # 可视化group
        self.layout_result_img_pair = self.visualize_compos_df('group_pair', show=False)  # 可视化grouppair
        self.layout_result_img_list = self.visualize_lists(show=False)


    def visualize_compos_df(self, visualize_attr, show=True):  # 具体绘图group和group pair的 被：get_layout_result_imgs(self)组pair和list的 绘图函数（总）调用
        board = self.img_resized.copy()
        return self.compos_df.visualize_fill(board, gather_attr=visualize_attr, name=visualize_attr, show=show)

    def visualize_lists(self, show=True):  # list专用绘图函数
        board = self.img_resized.copy()
        for lst in self.lists:
            board = lst.visualize_list(board, flag='block')
        if show:
            cv2.imshow('lists', board)
            cv2.waitKey()
            cv2.destroyWindow('lists')
        return board

    def visualize_layout_recognition(self):
        I1 = cv2.resize(self.layout_result_img_group, (self.img_reshape[1], self.img_reshape[0]))   # 组pair和list的 仅显示函数 self.img_reshape
        I2 = cv2.resize(self.layout_result_img_pair, (self.img_reshape[1], self.img_reshape[0]))
        I3 = cv2.resize(self.layout_result_img_list, (self.img_reshape[1], self.img_reshape[0]))
        img = np.concatenate([I1, I2, I3], axis=1)[:, :, ::-1]  # 拼一起显示 使用plt
        plt.figure(figsize=(27, 16))
        plt.imshow(img)
        plt.title('  group               group_pair               list', fontdict={'weight': 'normal', 'size': 60})
        plt.show()

    def visualize_element_detection(self):
        I1 = cv2.resize(self.detection_result_img['text'], (self.img_reshape[1], self.img_reshape[0]))   # 文本非文本合并的 仅显示函数
        I2 = cv2.resize(self.detection_result_img['non-text'], (self.img_reshape[1], self.img_reshape[0]))
        I3 = cv2.resize(self.detection_result_img['merge'], (self.img_reshape[1], self.img_reshape[0]))
        img = np.concatenate([I1, I2, I3], axis=1)[:, :, ::-1]  # 拼一起显示 使用plt
        plt.figure(figsize=(27, 16))
        plt.imshow(img)
        plt.title('  text                 non-text                merge',
                  fontdict={'weight': 'normal', 'size': 60})
        plt.show()


    # ——————————————————文本非文本合并————————————————————————
    # 这里感觉也用不到
    def draw_element_detection(self, line=2):  # 文本非文本合并的 绘图函数
        board_text = self.img_resized.copy()
        board_nontext = self.img_resized.copy()
        board_all = self.img_resized.copy()
        colors = {'Text': (196, 114, 68), 'Compo': (49, 125, 237),
                  'Block': (0, 166, 166)}  # bgr顺序 文本藏蓝色 控件橙色
        for compo in self.compos_json['compos']:
            position = compo['position']
            if compo['class'] == 'Text':
                draw.draw_label(board_text, [position['column_min'], position['row_min'], position['column_max'], position['row_max']], colors[compo['class']], line=line)
            else:
                draw.draw_label(board_nontext, [position['column_min'], position['row_min'], position['column_max'], position['row_max']], colors[compo['class']], line=line)
            draw.draw_label(board_all, [position['column_min'], position['row_min'], position['column_max'], position['row_max']], colors[compo['class']], line=line)

        self.detection_result_img['text'] = board_text
        self.detection_result_img['non-text'] = board_nontext
        self.detection_result_img['merge'] = board_all

    # ————————————————————————以下可能用不到了————————————————————————

    def visualize_all_compos(self, show=True):
        board = self.img_resized.copy()
        for compo in self.compos:
            board = compo.visualize(board)
        if show:
            cv2.imshow('compos', board)
            cv2.waitKey()
            cv2.destroyWindow('compos')

    def visualize_block(self, block_id, show=True):
        board = self.img_resized.copy()
        self.blocks[block_id].visualize_sub_blocks_and_compos(board, show=show)

    def visualize_blocks(self, show=True):
        board = self.img_resized.copy()
        for block in self.blocks:
            board = block.visualize_block(board)
        if show:
            cv2.imshow('compos', board)
            cv2.waitKey()
            cv2.destroyWindow('compos')

    def visualize_container(self, show=True):
        board = self.img_resized.copy()
        df = self.compos_df.compos_dataframe
        containers = df[df['class'] == 'Block']
        for i in range(len(containers)):
            container = containers.iloc[i]
            children = df.loc[list(container['children'])]
            for j in range(len(children)):
                child = children.iloc[j]
                color = (0,255,0) if child['class'] == 'Compo' else (0,0,255)
                cv2.rectangle(board, (child['column_min'], child['row_min']), (child['column_max'], child['row_max']), color, 2)
            draw.draw_label(board, (container['column_min'], container['row_min'], container['column_max'], container['row_max']), (166,166,0), text='container')
        if show:
            cv2.imshow('container', board)
            cv2.waitKey()
            cv2.destroyWindow('container')
