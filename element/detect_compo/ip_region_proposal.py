import cv2
from os.path import join as pjoin
import time
import json
import numpy as np

import element.detect_compo.lib_ip.ip_preprocessing as pre
import element.detect_compo.lib_ip.ip_draw as draw
import element.detect_compo.lib_ip.ip_detection as det
import element.detect_compo.lib_ip.file_utils as file
import element.detect_compo.lib_ip.Component as Compo
from element.config.CONFIG_UIED import Config
C = Config()


def nesting_inspection(org, grey, compos, ffl_block):
    '''
    Inspect all big compos through block division by flood-fill
    :param ffl_block: gradient threshold for flood-fill
    :return: nesting compos
    '''
    nesting_compos = []
    for i, compo in enumerate(compos):
        if compo.height > 50:  # 对于每个大型组件
            replace = False
            clip_grey = compo.compo_clipping(grey)
            n_compos = det.nested_components_detection(clip_grey, org, grad_thresh=ffl_block, show=False)
            Compo.cvt_compos_relative_pos(n_compos, compo.bbox.col_min, compo.bbox.row_min)

            for n_compo in n_compos:
                if n_compo.redundant:
                    compos[i] = n_compo
                    replace = True
                    break
            if not replace:
                nesting_compos += n_compos
    return nesting_compos


def compo_detection(input_img_path, ip_root, uied_params, bbox_detect, sub_class_detect, col_lines, row_lines,
                    resize_by_height=800, clean_save=False, show=False, wai_key=0):

    start = time.time()  # 环节
    name = input_img_path.replace('\\', '/').split('/')[-1][:-4]
    # print(len(bbox_detect))
    # *** Step 1 *** pre-processing: read img -> get binary map 天才第一步：读图片->拿到二值化图 这一步通过梯度阈值、形态学运算确定了ROI（感兴趣的区域）
    org, grey = pre.read_img(input_img_path, resize_by_height)
    # binary = pre.binarization(org, grad_min=int(uied_params['min-grad']))

    # *** Step 2 *** element detection 第二步：元素检测
    # det.rm_line(binary, show=show, wait_key=wai_key)  # import element.detect_compo.lib_ip.ip_detection as det
    # 得到list[Component（自定义对象）]
    bbox_detect, sub_class_detect = det.split_compos_by_lines(org, bbox_detect, sub_class_detect, col_lines, row_lines)  # 根据划线 分割识别成一块的控件
    uicompos = det.component_detection(org, bbox_detect, sub_class_detect)
    # *** Step 3 *** results refinement 第三步：结果完善
    # uicompos = det.compo_filter(uicompos, min_area=int(uied_params['min-ele-area']), img_shape=(org.shape[0], org.shape[1]))  # 根据过滤条件过滤
    # uicompos = det.merge_intersected_compos(uicompos)  # 合并重叠组件
    # uicompos = det.split_compos_by_lines(uicompos, col_lines, row_lines)  # 根据划线 分割识别成一块的控件
    # det.compo_block_recognition(binary, uicompos)  # 标记blocks uicompos.category = 'Block'
    det.compo_block_recognition(uicompos)
    if uied_params['merge-contained-ele']:  # 默认为真
        uicompos = det.rm_contained_compos_not_in_block(uicompos)  # 删除不是被blocks包含的其他组件

    Compo.compos_update(uicompos, (org.shape[0], org.shape[1]))
    # Compo.compos_containment(uicompos)

    # *** Step 4 ** nesting inspection: check if big compos have nesting element 第四步：嵌套检查——检查大组合是否有嵌套元素
    # uicompos += nesting_inspection(org, grey, uicompos, ffl_block=uied_params['ffl-block'])
    # Compo.compos_update(uicompos, org.shape)

    # 存ip的图
    res_img = draw.draw_bounding_box(org, uicompos, show=show, name='merged compo', write_path=pjoin(ip_root, name + '.jpg'), clean_save=clean_save, wait_key=wai_key)

    # *** Step 7 *** save detection result 最后：保存结果
    Compo.compos_update(uicompos, org.shape)
    ip_js = file.save_corners_json(pjoin(ip_root, name + '.json'), uicompos, org.shape, clean_save=clean_save)  # 存IP里的json

    # if not clean_save:
    #     print("[Compo Detection Completed in %.3f s] Input: %s Output: %s" % (time.time() - start, input_img_path, pjoin(ip_root, name + '.json')))
    # else:
    #     print("[Compo Detection Completed in %.3f s] Input: %s Output: clean_save no output" % (time.time() - start, input_img_path))  # 打印信息
    return res_img, ip_js
