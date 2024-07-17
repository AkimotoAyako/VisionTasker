import pandas as pd
import json
import cv2
import os


def export_compos_as_tree(compos, export_dir='data/output/tree'):
    def build_branch(compo):
        branch = compo.put_info()
        if len(compo.children) > 0:
            branch['children'] = []
            for c in compo.children:
                branch['children'].append(build_branch(c))
        return branch

    os.makedirs(export_dir, exist_ok=True)
    tree = []
    for cp in compos:
        tree.append(build_branch(cp))
    json.dump(tree, open(export_dir + '/tree.json', 'w'), indent=4)
    return tree


def visualize_Compos(compos_html, img):
    board = img.copy()
    for compo in compos_html:
        board = compo.visualize(board)
    cv2.imshow('compos', board)
    cv2.waitKey()
    cv2.destroyWindow('compos')


def cvt_list_and_compos_by_pair_and_group(df):
    '''
    :param df: type of dataframe
    :return: lists: [Compo obj]
             non_list_compos: [Compo obj]
    '''
    lists = []
    non_list_compos = []
    # list type of multiple (multiple compos in each list item) for paired groups
    groups = df.groupby('group_pair').groups
    compo_id = 0
    for i in groups:
        if i == -1 or len(groups[i]) == 1:
            continue
        sub_class_str = ''
        for sub_class in df.loc[groups[i][0]]['sub_class'].unique():
            sub_class_str += sub_class + '_'
        lists.append(Compo(compo_id=compo_id, compo_class='List-multi', compo_sub_class=sub_class_str[:-1],
                           compo_df=df.loc[groups[i]], list_alignment=df.loc[groups[i][0]]['alignment_in_group']))
        compo_id += 1
        # remove selected compos
        df = df.drop(list(groups[i]))

    # list type of single (single compo in each list item) for non-paired groups
    groups = df.groupby('group').groups
    for i in groups:
        if i == -1 or len(groups[i]) == 1:
            continue
        lists.append(Compo(compo_id=compo_id, compo_class='List-single', compo_sub_class=df.loc[groups[i][0]]['sub_class'],
                           compo_df=df.loc[groups[i]], list_alignment=df.loc[groups[i][0]]['alignment_in_group']))
        compo_id += 1
        # remove selected compos
        df = df.drop(list(groups[i]))

    # not count as list for non-grouped compos
    for i in range(len(df)):
        compo_df = df.iloc[i]
        # fake compo presented by colored div
        sub_class_str = ''
        for sub_class in df.loc[groups[i][0]]['sub_class'].unique():
            sub_class_str += sub_class + '_'
        compo = Compo(compo_id=compo_id, compo_class=compo_df['class'], compo_sub_class=sub_class_str[:-1], compo_df=compo_df)
        compo_id += 1
        non_list_compos.append(compo)
    return lists, non_list_compos


class Compo:
    def __init__(self, compo_id, compo_class, compo_sub_class, compo_df, in_list=None, compo_pair_to='', list_alignment=''):
        self.compo_df = compo_df   # df can contain one or more elements (list items)
        self.compo_id = compo_id
        self.compo_class = compo_class  # List, Compo, Text
        self.compo_sub_class = compo_sub_class
        self.compo_pair_to = compo_pair_to


        # get the clip for single element
        self.children = None
        self.parent = None
        self.in_list = in_list  # if the compo is in a list, store list id
        self.list_alignment = list_alignment

        # compo boundary
        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.center_row = None
        self.center_column = None
        self.width = None
        self.height = None

        self.text_content = None  # for Text

        self.init_info()

    def init_info(self):
        compo = self.compo_df
        if self.compo_class == 'List':
            self.top = int(compo['row_min'].min())
            self.left = int(compo['column_min'].min())
            self.bottom = int(compo['row_max'].max())
            self.right = int(compo['column_max'].max())
            self.center_row = (self.top + self.bottom) / 2
            self.center_column = (self.left + self.right) / 2
            self.width = int(self.right - self.left)
            self.height = int(self.bottom - self.top)
        else:
            self.top, self.left, self.bottom, self.right = compo['row_min'], compo['column_min'], compo['row_max'], compo['column_max']
            self.center_row, self.center_column = compo['center_row'], compo['center_column']
            self.width, self.height = compo['width'], compo['height']
            self.text_content = compo['text_content'] if compo['text_content'] != -1 else None
            self.children = compo['children'] if compo['children'] != -1 else None
            self.parent = compo['parent'] if compo['parent'] != -1 else None

    def wrap_info(self):
        info = {'id': self.compo_id, 'class': self.compo_class, 'sub_class': self.compo_sub_class, 'text_content': self.text_content, 'pair_to': self.compo_pair_to,
                'location': {'left': int(self.left), 'right': int(self.right), 'top': int(self.top), 'bottom': int(self.bottom)}}
        return info

    def visualize(self, img=None, flag='line', show=False, color=None):
        fill_type = {'line':2, 'block':-1}
        color_map = {'List':(166, 166, 166), 'Text': (196, 114, 68), 'Compo': (49, 125, 237), 'Text Content': (196, 114, 68)}
        if color is None:
            color = color_map[self.compo_class]
        board = img.copy()
        board = cv2.rectangle(board, (self.left, self.top), (self.right, self.bottom), color, fill_type[flag])
        if show:
            cv2.imshow('compo', board)
            cv2.waitKey()
            cv2.destroyWindow('compo')
        return board
