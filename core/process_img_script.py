import os
import GPUtil
import pandas as pd
from core.GUI import GUI
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import cv2
import time

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox, size):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

# 目标检测结果显示
def plot_results_detr(pil_img, prob, boxes, CLASSES, show_bu_show, clean_save):
    # 用PIL画图转换成opencv
    font_path = 'D:/UI_datasets/other_codes/simhei.ttf'  # 指定字体文件路径
    font = ImageFont.truetype(font_path, 35)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # 先转cv2扩大一个白边
    cv_img = cv2.copyMakeBorder(cv_img, 40, 0, 0, cv_img.shape[1] // 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))  # 再回去
    draw_img = pil_img.copy()
    draw = ImageDraw.Draw(draw_img)

    # 定义颜色映射
    jet_colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    # 绘制中文文本
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        # 计算颜色值
        cl = p.argmax()
        colour_value = int(cl) / 13
        colour_adjust = jet_colormap[int(colour_value * 255), 0]
        # 绘制矩形和文本
        draw.rectangle([xmin, ymin + 40, xmax, ymax + 40], outline=tuple(colour_adjust), width=4)  # 矩形框，可以根据需要设置颜色
        text = f'{CLASSES[int(cl)]}: {p.max():0.4f}'
        draw.text((xmin, ymin), text, font=font, fill=tuple(colour_adjust))  # 文本，可以根据需要设置颜色
    # 将 PIL 图像转换为 OpenCV 图像
    cv_img = cv2.cvtColor(np.array(draw_img), cv2.COLOR_RGB2BGR)
    if show_bu_show:
        plt.figure(figsize=(9, 12))
        plt.imshow(cv_img[:, :, [2, 1, 0]])
        plt.axis('off')  # 关闭坐标轴
        plt.show()
    return cv_img


def plot_results_yolo(pil_img, classes, prob, boxes, CLASSES, show_bu_show, clean_save):
    # 用PIL画图转换成opencv
    font_path = 'D:/UI_datasets/other_codes/simhei.ttf'  # 指定字体文件路径
    font = ImageFont.truetype(font_path, 35)
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # 先转cv2扩大一个白边
    cv_img = cv2.copyMakeBorder(cv_img, 40, 0, 0, cv_img.shape[1] // 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))  # 再回去
    draw_img = pil_img.copy()
    draw = ImageDraw.Draw(draw_img)
    # 定义颜色映射
    jet_colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    # 绘制中文文本
    for cls, p, (xmin, ymin, xmax, ymax) in zip(classes, prob, boxes.tolist()):
        # 计算颜色值
        colour_value = int(cls) / 13
        colour_adjust = jet_colormap[int(colour_value * 255), 0]
        # 绘制矩形和文本
        draw.rectangle([xmin, ymin + 40, xmax, ymax + 40], outline=tuple(colour_adjust), width=4)  # 矩形框，可以根据需要设置颜色
        text = f'{CLASSES[int(cls)]}: {p:0.4f}'
        draw.text((xmin, ymin), text, font=font, fill=tuple(colour_adjust))  # 文本，可以根据需要设置颜色
    # 将 PIL 图像转换为 OpenCV 图像
    cv_img = cv2.cvtColor(np.array(draw_img), cv2.COLOR_RGB2BGR)
    if show_bu_show:
        plt.figure(figsize=(9, 12))
        plt.imshow(cv_img[:, :, [2, 1, 0]])
        plt.axis('off')  # 关闭坐标轴
        plt.show()
    return cv_img

# 计算GIoU
def calculate_giou(pred_boxes, target_boxes):
    # 获取预测框和真实框的坐标
    pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
    target_boxes = box_cxcywh_to_xyxy(target_boxes)
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[0], pred_boxes[1], pred_boxes[2], pred_boxes[3]
    target_x1, target_y1, target_x2, target_y2 = target_boxes[0], target_boxes[1], target_boxes[2], target_boxes[3]

    # 计算预测框和真实框的面积
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    # 计算交集的坐标
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    # 计算交集的面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算并集的面积
    union_area = pred_area + target_area - inter_area

    # 计算GIoU
    iou = inter_area / union_area
    # 算最小闭包区域面积Ac
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

    giou = iou - (enclose_area - union_area) / enclose_area
    return giou

def ab_iou(fenmu_boxes, fenzi_boxes):
    # 获取预测框和真实框的坐标
    fenmu_boxes = box_cxcywh_to_xyxy(fenmu_boxes)
    fenzi_boxes = box_cxcywh_to_xyxy(fenzi_boxes)
    pred_x1, pred_y1, pred_x2, pred_y2 = fenmu_boxes[0], fenmu_boxes[1], fenmu_boxes[2], fenmu_boxes[3]
    target_x1, target_y1, target_x2, target_y2 = fenzi_boxes[0], fenzi_boxes[1], fenzi_boxes[2], fenzi_boxes[3]

    # 计算预测框和真实框的面积
    fenmu_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    # target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    # 计算交集的坐标
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    # 计算交集的面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算IoU
    iou = inter_area / fenmu_area

    return iou

def bbox_contain(pred_boxes, target_boxes):
    # 获取预测框和真实框的坐标
    pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
    target_boxes = box_cxcywh_to_xyxy(target_boxes)
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[0], pred_boxes[1], pred_boxes[2], pred_boxes[3]
    target_x1, target_y1, target_x2, target_y2 = target_boxes[0], target_boxes[1], target_boxes[2], target_boxes[3]
    if (pred_x1 <= target_x1 and pred_x2 >= target_x2 and pred_y1 <= target_y1 and pred_y2 >= target_y2) or \
       (pred_x1 > target_x1 and pred_x2 < target_x2 and pred_y1 > target_y1 and pred_y2 < target_y2):
        return True
    else:
        return False

def a_contain_b(a_boxes, b_boxes):
    # 获取预测框和真实框的坐标
    a_boxes = box_cxcywh_to_xyxy(a_boxes)
    b_boxes = box_cxcywh_to_xyxy(b_boxes)
    a_x1, a_y1, a_x2, a_y2 = a_boxes[0], a_boxes[1], a_boxes[2], a_boxes[3]
    b_x1, b_y1, b_x2, b_y2 = b_boxes[0], b_boxes[1], b_boxes[2], b_boxes[3]
    if (a_x1 <= b_x1 and a_x2 >= b_x2 and a_y1 <= b_y1 and a_y2 >= b_y2):
        return True
    else:
        return False

# NMS过滤

def NMS_yolo(bboxes, proba, cls):
    pred_idx = [idx for idx in range(len(proba))]  # 按照预测框个数编号
    pred_idx_set = set(pred_idx)  # 变成集合 方便扔
    # pb = 0.1148
    # print(np.where((proba.cpu().numpy() >= pb) & (proba.cpu().numpy() <= pb+0.0001))[0])
    for curr_idx in pred_idx:  # 从头开始 抓取一个框
        if curr_idx not in pred_idx_set:  # 如果这个框之前被干掉了 那就不用再检查了
            continue
        for other_idx in pred_idx:  # 抓取另一个框
            if other_idx not in pred_idx_set or other_idx == curr_idx:  # 如果另一个框也被干掉了/这俩是一个框 就放过
                continue
            else:
                giou = calculate_giou(bboxes[curr_idx], bboxes[other_idx])
                # 判断：1. giou大于0.25；2.对于同一类giou大于0.05 3.（选）存在互相包含且是同一类
                # print(cls[curr_idx], cls[other_idx])
                if giou > 0.25 or \
                        (giou > 0.05 and cls[curr_idx] == cls[other_idx]) or \
                        (bbox_contain(bboxes[curr_idx], bboxes[other_idx]) and ((cls[curr_idx] in [2,8,9] and cls[other_idx] in [2,8,9]) or cls[curr_idx] == cls[other_idx])):  # cls[curr_idx] == cls[other_idx]
                    pred_idx_set -= {other_idx if proba[curr_idx] > proba[other_idx] else curr_idx}  # 扔掉置信度小的一个
                elif a_contain_b(bboxes[other_idx], bboxes[curr_idx]) and ((cls[curr_idx] in [2,4] and cls[other_idx] in [2,4]) or cls[curr_idx] == cls[other_idx]):  # 在判断abiou之前 先判断包含 扔掉大的
                    if cls[other_idx] != 4:
                        pred_idx_set -= {other_idx}  # 扔掉外面的idx
                    elif cls[curr_idx] == 2:
                        pred_idx_set -= {curr_idx}  # 扔图片包含的按钮
                elif a_contain_b(bboxes[curr_idx], bboxes[other_idx]) and ((cls[curr_idx] in [2,4] and cls[other_idx] in [2,4]) or cls[curr_idx] == cls[other_idx]):
                    # pred_idx_set -= {curr_idx}  # 扔掉外面的idx
                    if cls[curr_idx] != 4:
                        pred_idx_set -= {curr_idx}  # 扔掉外面的idx
                    elif cls[other_idx] == 2:
                        pred_idx_set -= {other_idx}  # 扔图片包含的按钮
                elif ab_iou(bboxes[other_idx], bboxes[curr_idx]) > 0.5 and ((cls[curr_idx] in [2,4] and cls[other_idx] in [2,4]) or cls[curr_idx] == cls[other_idx]):  # 如果没被扔 再看一眼abiou
                    pred_idx_set -= {other_idx}  # 扔掉被占率高的other_idx
    return list(pred_idx_set)

def NMS_vins(bboxes, proba, cls, CLASSES):
    pred_idx = [idx for idx in range(len(proba))]  # 按照预测框个数编号
    pred_idx_set = set(pred_idx)  # 变成集合 方便扔
    for curr_idx in pred_idx:  # 从头开始 抓取一个框
        if curr_idx not in pred_idx_set:  # 如果这个框之前被干掉了 那就不用再检查了
            continue
        for other_idx in pred_idx:  # 抓取另一个框
            if other_idx not in pred_idx_set or other_idx == curr_idx:  # 如果另一个框也被干掉了/这俩是一个框 就放过
                continue
            else:
                giou = calculate_giou(bboxes[curr_idx], bboxes[other_idx])
                # 判断：1. giou大于0.25；2.对于同一类giou大于0.15 3.（选）存在互相包含且是同一类
                if giou > 0.25 or \
                        (giou > 0.15 and cls[curr_idx] == cls[other_idx]) \
                        or (bbox_contain(bboxes[curr_idx], bboxes[other_idx]) and (cls[curr_idx] == cls[other_idx] or
                           (CLASSES[int(cls[curr_idx])] in ['Image', 'Icon'] and CLASSES[int(cls[other_idx])] in ['Image', 'Icon']))):
                    pred_idx_set -= {other_idx if proba[curr_idx] > proba[other_idx] else curr_idx}  # 扔掉置信度小的一个
                if (ab_iou(bboxes[other_idx], bboxes[curr_idx]) > 0.1 and CLASSES[
                    int(cls[curr_idx])] == 'UpperTaskBar'):
                    pred_idx_set -= {other_idx}  # 扔掉包含的组件
        if CLASSES[int(cls[curr_idx])] == 'UpperTaskBar':
            pred_idx_set -= {curr_idx}  # 扔完包含的 最后扔upper栏
    return list(pred_idx_set)

# 按面积与置信度过滤（由于像素的原因~）
def size_conf_filter_detr(bbox_pred_, prob_, size_per=0.5, conf=0.3):
    sc_keep = []
    for n, bbox in enumerate(bbox_pred_):
        single_size = bbox[2] * bbox[3]
        if single_size < 0.45 and not (single_size >= size_per and prob_[n] <= conf):
            sc_keep.append(n)
        # else:
        #     print(single_size, 'is large with conf(', prob_[n], 'has been filtered')
    return sc_keep

def size_conf_filter_yolo(bbox_pred_, prob_, size_per=0.5, conf=0.3):
    sc_keep = []
    for n, bbox in enumerate(bbox_pred_):
        single_size = bbox[2] * bbox[3]
        if not (single_size >= size_per and prob_[n] <= conf) and (single_size < 0.85) and not (single_size >= size_per * 0.15 and prob_[n] <= 0.5):
            sc_keep.append(n)
        # else:
        #     print(single_size, 'is large with low conf(', prob_[n], 'has been filtered')
    return sc_keep


# 目标检测调用
def yolo_det(test_path, model, im, CLASSES, bar_id, conf_yolo=0.05, show_bu_show=True, clean_save=False):
    results = model([test_path], conf=conf_yolo, verbose=False)  # list格式
    # 输出格式整理 丢掉导航栏状态栏
    not_throw_id = (results[0].boxes.data[:, -1] != bar_id[0]) & (results[0].boxes.data[:, -1] != bar_id[1])
    result = results[0].boxes.data[not_throw_id]
    # result = results[0].boxes.data
    bbox_pred = result[:, :-2]
    bbox_pred = torch.stack(
        (((bbox_pred[:, 2] + bbox_pred[:, 0]) / 2) / im.size[0], ((bbox_pred[:, 3] + bbox_pred[:, 1]) / 2) / im.size[1],
         (bbox_pred[:, 2] - bbox_pred[:, 0]) / im.size[0], (bbox_pred[:, 3] - bbox_pred[:, 1]) / im.size[1]), dim=1)
    prob = result[:, -2]
    class_pred = result[:, -1]
    # 各种过滤
    sc_keep_idx = size_conf_filter_yolo(bbox_pred, prob, size_per=0.35, conf=0.5)  # 尺寸置信度过滤
    bbox_pred, prob, class_pred = bbox_pred[sc_keep_idx], prob[sc_keep_idx], class_pred[sc_keep_idx]
    pred_idx_set = NMS_yolo(bbox_pred, prob, class_pred)  # NMS
    bboxes_scaled = rescale_bboxes(bbox_pred, im.size)  # NMS前
    plot_results_yolo(im, class_pred, prob, bboxes_scaled, CLASSES, show_bu_show, clean_save)  # NMS前
    bboxes_scaled = rescale_bboxes(bbox_pred[pred_idx_set], im.size)
    det_img = plot_results_yolo(im, class_pred[pred_idx_set], prob[pred_idx_set], bboxes_scaled, CLASSES, show_bu_show, clean_save)  # NMS后
    # 目标检测结束 整理格式接UIED
    class_pred_name = [CLASSES[int(c)] for c in class_pred[pred_idx_set].tolist()]
    bbox = bbox_pred[pred_idx_set].to('cpu').tolist()
    return bbox, class_pred_name, det_img

def yolo_vins_det(test_path, model, im, CLASSES, text_id, bar_id, conf_vins=0.01, show_bu_show=True, clean_save=False):
    results = model([test_path], conf=conf_vins, verbose=False)  # list格式
    not_throw_id = results[0].boxes.data[:, -1] != bar_id[0]
    for tid in text_id + bar_id[1:]:
        not_throw_id &= results[0].boxes.data[:, -1] != tid
    result = results[0].boxes.data[not_throw_id]
    bbox_pred = result[:, :-2]
    bbox_pred = torch.stack(
        (((bbox_pred[:, 2] + bbox_pred[:, 0]) / 2) / im.size[0], ((bbox_pred[:, 3] + bbox_pred[:, 1]) / 2) / im.size[1],
         (bbox_pred[:, 2] - bbox_pred[:, 0]) / im.size[0], (bbox_pred[:, 3] - bbox_pred[:, 1]) / im.size[1]), dim=1)
    prob = result[:, -2]
    class_pred = result[:, -1]
    sc_keep_idx = size_conf_filter_yolo(bbox_pred, prob, size_per=0.35, conf=0.5)  # 尺寸置信度过滤
    bbox_pred, prob, class_pred = bbox_pred[sc_keep_idx], prob[sc_keep_idx], class_pred[sc_keep_idx]
    bboxes_scaled = rescale_bboxes(bbox_pred, im.size)  # NMS前
    plot_results_yolo(im, class_pred, prob, bboxes_scaled, CLASSES, show_bu_show, clean_save)  # NMS前
    pred_idx_set = NMS_vins(bbox_pred, prob, class_pred, CLASSES)  # NMS
    bboxes_scaled = rescale_bboxes(bbox_pred[pred_idx_set], im.size)
    det_img = plot_results_yolo(im, class_pred[pred_idx_set], prob[pred_idx_set], bboxes_scaled, CLASSES, show_bu_show, clean_save)  # NMS后
    # 目标检测结束 整理格式接UIED
    class_pred_name = [CLASSES[int(c)] for c in class_pred[pred_idx_set].tolist()]
    bbox = bbox_pred[pred_idx_set].to('cpu').tolist()
    return bbox, class_pred_name, det_img

# 边缘线检测
def line_det(img_path, output_root, show_bu_show=True, clean_save=False, workflow_only=True):
    if workflow_only:
        clean_save=True
    src = cv2.imread(img_path)
    im_name = img_path.split('/')[-1].split('.')[0]
    img_bgr = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    unique_elements, counts = np.unique(img_bgr, return_counts=True)  # 计算灰度图像中不同像素值的频率 按颜色数值升序排列
    per_counts = np.round(counts / np.sum(counts), 2)  # 计算每个像素的相对频率
    color_group = unique_elements[per_counts >= 0.09]  # 过滤 找出频率大于0.09的颜色 将像素值分为几个颜色级别
    if len(color_group) == 1:
        draw_group = [0]  # 只有一个颜色值 则不会改变
    else:  # 多个颜色分组
        color_group = sorted(color_group)[-3:]  # 只取三个色
        draw_group = [255 - 255//(len(color_group))*(i+1) for i in range(len(color_group))]  # 均匀分布

    img_clu = 255*np.ones(img_bgr.shape, np.uint8)  # 新建画布
    for n, i in enumerate(color_group):
        mask = (img_bgr >= i - 9) & (img_bgr <= i + 5)
        img_clu[mask] = draw_group[n]  # 颜色一样才量化
    fld = cv2.ximgproc.createFastLineDetector(length_threshold=200)
    dlines = fld.detect(img_clu)

    col_lines, row_lines = np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    if dlines is not None:
        for dline in dlines:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            cv2.line(src, (x0, y0), (x1, y1), (0, 0, 255), 3, cv2.LINE_AA)

        if not clean_save:
            cv2.imwrite(output_root + 'layout/' + im_name + '_lines.jpg', src)  # 存图
        if show_bu_show:
            plt.figure(figsize=(9, 16))
            plt.imshow(src[:, :, [2, 1, 0]])
            plt.axis('off')  # 关闭坐标轴
            plt.show()

        dlines = np.squeeze(dlines)  # 后处理

        if len(dlines.shape) == 1:
            dlines = np.expand_dims(dlines, axis=0)

        condition = np.abs(dlines[:, 0] - dlines[:, 2]) <= 3  # 提取垂直和水平的线条 第一列是垂直/水平坐标 第二三列是起止点坐标
        selected_rows = dlines[condition]
        col_lines = np.stack(((selected_rows[:, 0] + selected_rows[:, 2]) / 2, selected_rows[:, 1], selected_rows[:, 3]),
                             axis=1)
        condition = np.abs(dlines[:, 1] - dlines[:, 3]) <= 3
        selected_rows = dlines[condition]
        row_lines = np.stack(((selected_rows[:, 1] + selected_rows[:, 3]) / 2, selected_rows[:, 0], selected_rows[:, 2]),
                             axis=1)
        indices = np.where(col_lines[:, 1] > col_lines[:, 2])[0]  # 保证第二列小于第三列
        col_lines[indices, 1], col_lines[indices, 2] = col_lines[indices, 2], col_lines[indices, 1]
        indices = np.where(row_lines[:, 1] > row_lines[:, 2])[0]
        row_lines[indices, 1], row_lines[indices, 2] = row_lines[indices, 2], row_lines[indices, 1]
    return col_lines, row_lines, src  # 垂直线和水平线的array（都是3列）

# 主检测函数
def process_img(label_path_dir, img_path, output_root, layout_json_dir,
                high_conf_flag, alg, clean_save, plot_show, ocr_save_flag,
                model_ver, model_det, model_cls, preprocess, pd_free_ocr=None,
                ocr_only=True, workflow_only=True, accurate_ocr=True, lang='zh'):

    ocr = pd_free_ocr
    st_time = time.time()

    torch.set_grad_enabled(False)

    CLASSES = []  # 分类不同
    text_id = []  # 文字不要
    bar_id = [0, 1]  # 状态栏导航栏也不要
    # vins数据集的类别数可以选 classes
    if model_ver == '12' or model_ver == '12e':  # ns
        CLASSES = ['状态栏', '导航栏', 'buttonicon', 'EditText', '带复选框的文本',
                   'Image', '页面指示器', 'Switch', '背景图', '地图', '下拉菜单', '多选框'] if lang == 'zh' else \
                    ['status bar', 'navigation bar', 'buttonicon', 'EditText', 'check box with text',
                     'Image', 'page indicator', 'Switch', 'background image', 'map', 'spinner', 'check box']
    elif model_ver == '14':  # s
        CLASSES = ['状态栏', '导航栏', 'buttonicon', 'Text', 'EditText', 'TextButton', '带复选框的文本',
                   'Image', '页面指示器', 'Switch', '背景图', '地图', '下拉菜单', '多选框'] if lang == 'zh' else \
                    ['status bar', 'navigation bar', 'buttonicon', 'Text', 'EditText', 'TextButton', 'check box with text',
                     'Image', 'page indicator', 'Switch', 'background image', 'map', 'spinner', 'check box']
        text_id = [3, 5]  # 文字不要
    elif model_ver == '18':  # n
        CLASSES = ['buttonicon', 'EditText', '带复选框的文本', '状态栏',
                   'Image', '弹窗', 'APP工具栏', '侧拉栏', '导航栏', '页面指示器',
                   'Switch', '背景图', '标签导航', '卡片', '地图', '下拉菜单',
                   'Remember（缓存）', '多选框'] if lang == 'zh' else \
                    ['buttonicon', 'EditText', 'check box with text', 'status bar',
                     'Image', 'Pop-up Window', 'Toolbar', 'Slider', 'navigation bar', 'page indicator',
                     'Switch', 'background image', 'tag bar', 'card', 'map', 'spinner',
                     'Remember', 'check box']
        bar_id = [3, 8]
    elif model_ver == '20':  # all
        CLASSES = ['buttonicon', 'Text', 'EditText', 'TextButton', '带复选框的文本', '状态栏',
                   'Image', '弹窗', 'APP工具栏', '侧拉栏', '导航栏', '页面指示器',
                   'Switch', '背景图', '标签导航', '卡片', '地图', '下拉菜单',
                   'Remember（缓存）', '多选框'] if lang == 'zh' else \
                    ['buttonicon', 'Text', 'EditText', 'TextButton', 'check box with text', 'status bar',
                     'Image', 'Pop-up Window', 'toolbar', 'slider', 'navigation bar', 'page indicator',
                     'switch', 'background image', 'tag bar', 'card', 'map', 'spinner',
                     'remember', 'check box']
        bar_id = [5, 10]
        text_id = [1, 3]
    CLASSES = [cls.lower() for cls in CLASSES]  # 全部小写

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # —————————— 开始 ——————————
    label_path = label_path_dir + ('icon_labels_chn.json' if lang=='zh' else 'icon_labels_en.json')
    # 获取label文件
    with open(label_path_dir + 'icon_labels_final.json', "r") as json_file:
        real_labels_dict = json.load(json_file)
    # 获取人话版label
    with open(label_path, "r", encoding='utf-8') as json_file:
        labels_for_read = json.load(json_file)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if '.' in img_path:
        input_path = img_path
        # 目标检测
        im = Image.open(input_path)
        im = im.convert("RGB")
        col_lines, row_lines, line_img = line_det(img_path, output_root, show_bu_show=plot_show, clean_save=clean_save, workflow_only=workflow_only)  # 线框检测
        col_lines, row_lines = col_lines/(max(im.size)/800), row_lines/(max(im.size)/800)  # 按照长边800等比例缩放
        img = transform(im).unsqueeze(0).to(device)
        if alg == 'yolo':
            low_conf, high_conf = 0.02, 0.1
            CLASSES = ['状态栏', '导航栏', 'buttonicon', 'edittext',
                       'image', '页面指示器', '加载进度条', '评分条',
                       '多选框', '选项', '下拉菜单', 'autocompletetextview', 'switch'] if lang=='zh' else \
                        ['status bar', 'navigation bar', 'buttonicon', 'edittext',
                         'image', 'page indicator', 'progress bar', 'rating bar',
                         'check box', 'options', 'spinner', 'autocompletetextview', 'switch']
            bar_id = [0, 1]
            conf_th = low_conf if not high_conf_flag else high_conf
            bbox, class_pred_name, det_img = yolo_det(input_path, model_det, im, CLASSES, bar_id, conf_yolo=conf_th, show_bu_show=plot_show, clean_save=clean_save)

            gui = GUI(input_path, det_img, line_img, output_dir=output_root)  # 实例化各种存储空间
            gui.detect_element(col_lines, row_lines, True, True, True, bbox, class_pred_name,  img_resize_longest_side=800,
                               ocr_save=ocr_save_flag, clean_save=clean_save, ocr_only=ocr_only, workflow_only=workflow_only, ocr=ocr, accurate_ocr=accurate_ocr)  # is_ocr=True, is_non_text=True, is_merge=True
            if plot_show:
                gui.visualize_element_detection()
            result_js = gui.recognize_layout(model_cls, preprocess, device, real_labels_dict, labels_for_read, col_lines, row_lines,
                                             clean_save=clean_save, layout_json_dir=layout_json_dir, workflow_only=workflow_only, lang=lang)
            if plot_show:
                gui.visualize_final_imgs()
            # print('All Completed in ', round(time.time() - st_time, 4), 's')
            return result_js
        elif alg == 'vins':
            low_conf, high_conf = 0.2, 0.5
            conf_th = low_conf if not high_conf_flag else high_conf
            bbox, class_pred_name, det_img = yolo_vins_det(input_path, model_det, im, CLASSES, text_id, bar_id, conf_vins=conf_th, show_bu_show=plot_show, clean_save=clean_save)

            gui = GUI(input_path, det_img, line_img, output_dir=output_root)  # 实例化各种存储空间
            gui.detect_element(col_lines, row_lines, True, True, True, bbox, class_pred_name,  img_resize_longest_side=800,
                               ocr_save=ocr_save_flag, clean_save=clean_save, ocr_only=ocr_only, workflow_only=workflow_only, ocr=ocr, accurate_ocr=accurate_ocr)  # is_ocr=True, is_non_text=True, is_merge=True
            if plot_show:
                gui.visualize_element_detection()
            result_js = gui.recognize_layout(model_cls, preprocess, device, real_labels_dict, labels_for_read, col_lines, row_lines,
                                             clean_save=clean_save, layout_json_dir=layout_json_dir, workflow_only=workflow_only, lang=lang)
            if plot_show:
                gui.visualize_final_imgs()
            # print('All Completed in ', round(time.time() - st_time, 4), 's')
            return result_js
        else:
            print('你个老六alg写错了你个老六alg写错了你个老六alg写错了你个老六alg写错了你个老六alg写错了你个老六alg写错了')
            print('在这里复制一个吧： yolo / detr / vins')
            return ['你个老六alg写错了你个老六alg写错了你个老六alg写错了你个老六alg写错了你个老六alg写错了你个老六alg写错了']


if __name__ == "__main__":
    alg = 'yolo'  # yolo / vins 目标检测选一
    accurate_ocr = False  # 是否使用高精度版OCR
    ocr = None
    # 导入模型 开机仅一次即可
    import core.import_models as import_models

    gpus = GPUtil.getGPUs()
    print('终于开始工作了吗？你选择的模型是:', alg)
    now_gpu = gpus[0]
    if now_gpu.memoryFree/now_gpu.memoryTotal > 0.9:
        print('你的装备是：', now_gpu.name, '现在使用的内存：', now_gpu.memoryFree, '/', now_gpu.memoryTotal, '哈哈没人和你抢')
    else:
        print('你的装备是', now_gpu.name, '内存使用情况：', now_gpu.memoryFree, '/', now_gpu.memoryTotal, '好像有别的程序在用 显卡发出尖锐爆鸣')

    if not accurate_ocr:
        model_ver, model_det, model_cls, preprocess, ocr = import_models.import_all_models \
            (alg, accurate_ocr=accurate_ocr,
             # model_path_yolo='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/yolo_s_best.pt',
             model_path_yolo='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/yolo_mdl.pt',
             model_path_vins_dir='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/yolo_vins_',
             model_ver='14',
             model_path_vins_file='_mdl.pt',
             model_path_cls='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/clip_mdl.pth'
             )
    else:
        model_ver, model_det, model_cls, preprocess = import_models.import_all_models \
            (alg,
             # model_path_yolo='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/yolo_s_best.pt',
             model_path_yolo='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/yolo_mdl.pt',
             model_path_vins_dir='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/yolo_vins_',
             model_ver='14',
             model_path_vins_file='_mdl.pt',
             model_path_cls='D:/UI_datasets/other_codes/GUI-Detection-Grouping/pt_models/clip_mdl.pth'
             )

    label_path_dir = 'D:/UI_datasets/other_codes/GUI-Detection-Grouping/clip_labels/'  # 176类分类注释文件存放地址
    output_root = 'D:/UI_datasets/other_codes/GUI-Detection-Grouping/outputs/'  # 如果不是clean输出模式 完整版的输出文件存放的文件夹
    layout_json_dir = 'D:/UI_datasets/other_codes/GUI-Detection-Grouping/outputs/clean_json'  # clean模式下 最终识别结果json输出的文件夹


    # 单张
    img_path = 'D:/UI_datasets/other_codes/GUI-Detection-Grouping/test_pics/overviewnms.png'  # 检测图片路径
    '''
    0.png 1.png
    12123.jpg 12306.jpg
    airport.jpg alipay.jpg
    hema_new.png hema.jpg
    OneNote.png setting.jpg
    wechat_me_en.jpg wechat_me.jpg wechat_me_service.jpg
    
    cainiao_1.png cainiao_2.png
    music_1.png music_2.png music_3.png
    qq_1.png qq_2.png qq_3.png
    zfb_1.png zfb_12.png
    zfb_1.png zfb_2.png zfb_3.png
    '''
    # 选项
    high_conf_flag = False  # 是否使用高支持度阈值 对于系统应用可以提升加快速度
    clean_save = False  # 是否只按照路径要求输出layout的json
    plot_show = True  # 是否显示图片
    lang = 'zh'  # en / zh  # 输出语言选择
    if clean_save:
        plot_show = False
    ocr_save_flag = 'save'   # ocr省钱模式 用于反复调整时 直接使用已保存的ocr结果 部分文件支持
    ocr_output_only = False  # 新增：只输出ocr结果 不要所有ip
    workflow_only = False     # 只输出json和整体流程图

    # 主函数
    if not accurate_ocr:
        result_js = process_img(label_path_dir, img_path, output_root, layout_json_dir, high_conf_flag, alg,
                                clean_save, plot_show, ocr_save_flag, model_ver, model_det, model_cls, preprocess, pd_free_ocr=ocr,
                                ocr_only=ocr_output_only, workflow_only=workflow_only, accurate_ocr=accurate_ocr, lang=lang)
    else:
        result_js = process_img(label_path_dir, img_path, output_root, layout_json_dir, high_conf_flag, alg,
                                clean_save, plot_show, ocr_save_flag, model_ver, model_det, model_cls, preprocess,
                                ocr_only=ocr_output_only, workflow_only=workflow_only, accurate_ocr=accurate_ocr, lang=lang)
    for value in result_js:
        print(value)

    #
    # # 循环
    # # 选项
    # high_conf_flag = False  # 是否使用高支持度阈值 对于系统应用可以提升加快速度
    # alg = 'yolo'  # yolo / detr / vins 三种算法
    # clean_save = False  # 是否只按照路径要求输出layout的json
    # plot_show = False  # 是否显示图片
    # ocr_save_flag = ''  # ocr省钱模式 用于反复调整时 直接使用已保存的ocr结果 部分文件支持
    # ocr_output_only = False  # 新增：只输出ocr结果 不要所有ip
    # workflow_only = False  # 只输出json和整体流程图
    # accurate_ocr = True      # 是否使用高精度版OCR
    #
    # test_pics_path = 'D:/UI_datasets/other_codes/GUI-Detection-Grouping/test_pics/'
    # for img_name in tqdm(os.listdir(test_pics_path)[6:]):
    #     if '.' in img_name:
    #         img_path = test_pics_path + img_name  # 检测图片路径
    #         # 主函数
    #         result_js = process_img(label_path_dir, img_path, output_root, layout_json_dir, high_conf_flag,
    #         alg, clean_save, plot_show, ocr_save_flag, model_ver, model_det, model_cls, preprocess,
    #         ocr_only=ocr_output_only, workflow_only=workflow_only, accurate_ocr=accurate_ocr)
