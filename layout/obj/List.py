from layout.obj.Compo import Compo
from layout.lib.draw import *

import cv2
import numpy as np
from random import randint as rint


class List(Compo):
    def __init__(self, compo_id, list_class, compo_df, list_alignment=None):
        super().__init__(compo_id=compo_id, compo_class='List', compo_sub_class='List', compo_df=compo_df)
        self.list_class = list_class    # single/multi
        self.list_alignment = list_alignment
        self.list_items = []  # list of list_items - list_item: a row in a vertical list / a column in a horizontal list

        self.partition_list_items()
        self.sort_items()

    def get_inner_compos(self):
        compos = []
        for list_item in self.list_items:
            for compo in list_item:
                compos.append(compo)
        return compos

    def wrap_info(self):
        # get basic info
        info = super().wrap_info()
        # add list items
        info['list_class'] = self.list_class
        info['list_alignment'] = self.list_alignment
        info['list_items'] = []
        for list_item in self.list_items:
            compos = []
            for compo in list_item:
                compos.append(compo.wrap_info())
            info['list_items'].append(compos)
        return info

    def wrap_list_items(self):
        items = []
        for list_item in self.list_items:
            compos = []
            for compo in list_item:
                compos.append(compo.compo_class)
            items.append(compos)
        return items

    def sort_items(self):
        if self.list_alignment == 'v':
            self.list_items = sorted(self.list_items, key=lambda x: x[0].top)
        else:
            self.list_items = sorted(self.list_items, key=lambda x: x[0].left)
        self.compo_df = self.compo_df.sort_values(by=['list_item'], ascending=True)

    def partition_list_items(self):
        # each row/column contains multiple compos
        if self.list_class == 'multi':
            groups = self.compo_df.groupby('list_item').groups
            for i in groups:
                group = list(groups[i])
                item_compos_df = self.compo_df.loc[group]
                list_item = []
                for j in range(len(item_compos_df)):
                    item = item_compos_df.iloc[j]
                    pair_to_id = str(item['pair_to']) if item['pair_to']!=-1 else ''  # 标注pair_to信息
                    list_item.append(Compo(compo_id='c-' + str(item['id']), compo_class=item['class'], compo_sub_class=item['sub_class'], compo_df=item, in_list=self.compo_id, compo_pair_to=pair_to_id))
                self.list_items.append(list_item)
        elif self.list_class == 'single':
            for i in range(len(self.compo_df)):
                item = self.compo_df.iloc[i]
                pair_to_id = str(item['pair_to']) if item['pair_to'] != -1 else ''  # 标注pair_to信息
                self.list_items.append([Compo(compo_id='c-' + str(item['id']), compo_class=item['class'], compo_sub_class=item['sub_class'], compo_df=item, in_list=self.compo_id, compo_pair_to=pair_to_id)])

    def visualize_list(self, img=None, flag='line', show=False):
        board = img.copy()
        for list_item in self.list_items:
            color = random_color()
            for compo in list_item:
                if compo.compo_class == 'Block':
                    board = compo.visualize(board, 'line', color=color)
                else:
                    board = compo.visualize(board, flag, color=color)
        draw_label(board, [self.left, self.top, self.right, self.bottom], (166, 100, 255), text='List')
        if show:
            cv2.imshow('list', board)
            cv2.waitKey()
            cv2.destroyWindow('list')
        return board
