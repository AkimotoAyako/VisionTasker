import numpy as np
import cv2


class Element:
    def __init__(self, id, corner, category, sub_category, text_content=None):
        self.id = id
        self.category = category
        self.sub_category = sub_category
        self.col_min, self.row_min, self.col_max, self.row_max = corner
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

        self.text_content = text_content
        self.parent_id = None
        self.children = []  # list of elements

    def init_bound(self):
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

    def put_bbox(self):
        return self.col_min, self.row_min, self.col_max, self.row_max

    def wrap_info(self):
        info = {'id':self.id, 'class': self.category, 'sub_class': self.sub_category, 'height': self.height, 'width': self.width,
                'position': {'column_min': self.col_min, 'row_min': self.row_min, 'column_max': self.col_max,
                             'row_max': self.row_max}}
        if self.text_content is not None:
            info['text_content'] = self.text_content
        if len(self.children) > 0:
            info['children'] = []
            for child in self.children:
                info['children'].append(child.id)
        if self.parent_id is not None:
            info['parent'] = self.parent_id
        return info

    def resize(self, resize_ratio):
        self.col_min = int(self.col_min * resize_ratio)
        self.row_min = int(self.row_min * resize_ratio)
        self.col_max = int(self.col_max * resize_ratio)
        self.row_max = int(self.row_max * resize_ratio)
        self.init_bound()

    def element_merge(self, element_b, new_element=False, new_category=None, new_id=None):
        col_min_a, row_min_a, col_max_a, row_max_a = self.put_bbox()
        col_min_b, row_min_b, col_max_b, row_max_b = element_b.put_bbox()
        new_corner = (min(col_min_a, col_min_b), min(row_min_a, row_min_b), max(col_max_a, col_max_b), max(row_max_a, row_max_b))
        if element_b.text_content is not None:
            self.text_content = element_b.text_content if self.text_content is None else self.text_content + '\n' + element_b.text_content
        if new_element:
            return Element(new_id, new_corner, new_category)
        else:
            self.col_min, self.row_min, self.col_max, self.row_max = new_corner
            self.init_bound()

    def calc_intersection_area(self, element_b, bias=(0, 0)):
        a = self.put_bbox()  #
        b = element_b.put_bbox()
        col_min_s = max(a[0], b[0]) - bias[0]
        row_min_s = max(a[1], b[1]) - bias[1]
        col_max_s = min(a[2], b[2])
        row_max_s = min(a[3], b[3])
        w = np.maximum(0, col_max_s - col_min_s)
        h = np.maximum(0, row_max_s - row_min_s)
        inter = w * h

        iou = inter / (self.area + element_b.area - inter)
        ioa = inter / self.area
        iob = inter / element_b.area

        return inter, iou, ioa, iob

    def element_relation(self, element_b, bias=(0, 0)):
        """
        @bias: (horizontal bias, vertical bias)
        :return: -1 : a in b
                 0  : a, b are not intersected
                 1  : b in a
                 2  : a, b are identical or intersected
        """
        inter, iou, ioa, iob = self.calc_intersection_area(element_b, bias)

        # area of intersection is 0
        if ioa == 0:
            return 0
        # a in b
        if ioa >= 1:
            return -1
        # b in a
        if iob >= 1:
            return 1
        return 2

    def visualize_element(self, img, color=(49, 125, 237), line=1, draw_rec=True, show=False, gpt4_text_size=0.55, gpt4_text_thickness=2):  # 真正的可视化！
        loc = self.put_bbox()  # x1, y1, x2, y2
        ele_id = self.id
        if draw_rec:  # 非gpt4v版本 正常画框
            cv2.putText(img, str(ele_id), (int(loc[0]), int(loc[1] - 2)), cv2.FONT_HERSHEY_PLAIN, 0.9, color)
            cv2.rectangle(img, loc[:2], loc[2:], color, line)
        else:  # 给gpt4v的版本
            # 注意 这里不是画坐标框 是在写id的地方下面画了一个白色底色
            # cv2.rectangle(img, (int((loc[0] + loc[2]) / 2 - 5), int(loc[1] - 12)), (int((loc[0] + loc[2]) / 2 + 7 + (len(str(ele_id))-1) * 10), int(loc[1] + 3)), (255, 255, 255), thickness=-1)
            # cv2.putText(img, str(ele_id), (int((loc[0]+loc[2])/2-5), int(loc[1] + 2)), cv2.FONT_HERSHEY_SIMPLEX, gpt4_text_size, (49, 125, 237), gpt4_text_thickness)
            # 计算文本的大小
            text_size, _ = cv2.getTextSize(str(ele_id), cv2.FONT_HERSHEY_SIMPLEX, gpt4_text_size, gpt4_text_thickness)

            # 调整框的大小以适应文本
            text_x_offset = 1
            text_y_offset = 1
            box_width = text_size[0] + text_x_offset * 2
            box_height = text_size[1] + text_y_offset * 2  


            # 绘制矩形
            cv2.rectangle(img, (int((loc[0] + loc[2]) / 2 - 5), int(loc[1] - 12)), 
                        (int((loc[0] + loc[2]) / 2 - 5 + box_width), int(loc[1] - 12 + box_height)), 
                        (255, 255, 255), thickness=-1)

            # 绘制文本
            cv2.putText(img, str(ele_id), (int((loc[0]+loc[2])/2 - 5 + text_x_offset), int(loc[1] - 12 + box_height - text_y_offset)), 
                        cv2.FONT_HERSHEY_SIMPLEX, gpt4_text_size, (49, 125, 237), gpt4_text_thickness)

        # for child in self.children:
        #     child.visualize_element(img, color=(255, 0, 255), line=line)
        if show:
            cv2.imshow('element', img)
            cv2.waitKey(0)
            cv2.destroyWindow('element')
