# -*- coding: utf-8 -*-
import os
# gpu_id = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
# from detr_main import build_model_main
# import argparse
from ultralytics import YOLO
import torch
import clip
import time

torch.set_grad_enabled(False)

def import_all_models(alg, accurate_ocr=True,  # yolo / vins 目标检测选一
                      model_path_yolo='pt_model/yolo_mdl.pt',
                      model_path_vins_dir='pt_model/yolo_vins_',
                      model_ver='14',
                      model_path_vins_file='_mdl.pt',
                      model_path_cls='pt_model/clip_mdl.pth',
                      gpt4v_mode=False):

    model_path_vins = model_path_vins_dir + model_ver + model_path_vins_file

    print('🥱I\'m importing the model....')
    if accurate_ocr:
        # print('高精度版本OCR不要导入🤣')
        ocr = None
    else:
        # print('📋🤔Basic OCR being imported...')
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True)  #, lang='en'
    # print('The OCR has been imported and the detection model you selected is', alg)
    # 导入模型
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_det = None
    if alg == 'yolo':
        # print('🚀importing RICO-YOLOv8, path：', model_path_yolo)
        model_det = YOLO(model_path_yolo, task='detect')  # Yolov8n
    elif alg == 'vins':
        # print('🚀importing VINS-YOLOv8, path：', model_path_vins)
        model_det = YOLO(model_path_vins, task='detect')  # Yolov8n + vins数据集
    if not gpt4v_mode:
        # print('Target detection model import is complete😉 ✓ ...... I\'m importing the CLIP. It may take a while.😅 path：', model_path_cls)
        model_cls, preprocess = clip.load("ViT-L/14", device=device, jit=False)  # 分类数据集
        # 加载权重
        model_cls.load_state_dict(torch.load(model_path_cls, map_location=device)['network'])
        model_cls.eval()  # 推理模式
        # print('CLIP ✓ 🥰')
    else:
        model_cls, preprocess = None, None
        # print('GPT4V mode 不导入CLIP 你耗子尾汁')

    print('💯Successfully imported the model! Good luck!🍀')
    if accurate_ocr:
        return model_ver, model_det, model_cls, preprocess
    else:
        return model_ver, model_det, model_cls, preprocess, ocr
