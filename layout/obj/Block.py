import pandas as pd
import cv2
from random import randint as rint

from layout.lib.draw import *

block_id = 0


def slice_blocks(compos, block_id, direction='v', change_alignment=True):
    '''
    Slice the GUI into hierarchical blocks based on the recognized Compos
    Vertically or horizontally scan compos
    :param direction: slice vertically or horizontally
    :param compos: Compo objects, including elements and lists
    :return blocks: list of [Block objs]
    :return compos: list of compos not blocked: list of [CompoHTML objects]

    将GUI界面分割成层次化的"Block"块，块内包含了识别出的组件。
    它的主要功能是按照指定的方向（垂直或水平）扫描组件，将它们划分为块。
    direction参数：指定划分块的方向，可以是垂直（'v'）或水平（'h'）方向。
    参数：
    compos：是一组Compo对象，包括各种元素和列表。
    返回：
    blocks：是一个包含Block对象的列表，每个Block对象代表一个块。
    non_blocked_compos：是一个包含未被块包含的组件的列表。
    '''
    blocks = []  # 块
    block_compos = []  # 块内组件
    non_blocked_compos = compos  # 没法被划分成块的
    is_divided = False
    divider = -1  # 用于跟踪扫描时的分割点，以便确定何时创建新的块。
    # slice from top to bottom 从上到下
    if direction == 'v':
        # reverse the direction of next slicing
        next_direction = 'h'  # 从上到下的大顺序里 后面再细分的顺序就是从左到右
        if not change_alignment:
            next_direction = 'v'  # 对于块内分块 不再更改方向
        compos.sort(key=lambda x: x.top)  # 按照顶边排序
        for compo in compos:  # 对每一个组件(合并成list的就是list)
            # new block
            # if divider is above than this compo's top, then gather the previous block_compos as a block
            # 根据扫描过程中的分割点（divider）和组件的位置，将组件分配给不同的块
            if divider <= compo.top:  # 1.如果当前组件的位置超过了当前的 divider（现在是顶部位置低于之前的），则说明需要创建一个新的块。
                # prev_divider = divider
                divider = compo.bottom  # 更新divider 是目前组件的下边缘
                if len(block_compos) == 1:  # block_compos这里存放的是上一个块缓存的几个组件 如果只有一个组件 那么不算块 所以没存到最后的blocks里
                    is_divided = True  # 所以啥都不干 但是因为当前组件的位置超过了当前的 divider 所以还是置为True（但是后面也没用
                if len(block_compos) > 1:
                    is_divided = True
                    block_id += 1  # 给它一个新编号
                    blocks.append(Block(id='b-'+str(block_id), compos=block_compos, slice_sub_block_direction=next_direction))  # 把之前的缓存放到blocks里 并在这个里面从左往右分小block
                    # remove blocked compos
                    non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))  # 从还没分的block里面带走
                block_compos = []  # 清空缓存区
            # extend block
            elif compo.top < divider < compo.bottom:  # 2. 如果新的组件头在block里面 脚出来了
                divider = compo.bottom  # 更新块的当前最低位置 盖住它
            block_compos.append(compo)  # 无论是新的还是上一个 先放入block_compos

        # if there are some sub-blocks, gather the left compos as a block 如果检测到不止一个block（有时会因为排版原因 没有办法分出两个block）把剩下的元素合并成一个block
        if is_divided and len(block_compos) > 1:
            block_id += 1
            blocks.append(Block(id='b-'+str(block_id), compos=block_compos, slice_sub_block_direction=next_direction))
            # remove blocked compos
            non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))

    # slice from left to right 从左到右
    elif direction == 'h':
        # reverse the direction of next slicing
        next_direction = 'v'
        if not change_alignment:
            next_direction = 'h'  # 对于块内分块 不再更改方向
        compos.sort(key=lambda x: x.left)
        for compo in compos:
            # new block
            # if divider is lefter than this compo's right, then gather the previous block_compos as a block
            if divider < compo.left:
                prev_divider = divider
                divider = compo.right

                gap = int(compo.left - prev_divider)
                # gather previous compos in a block
                # a single compo is not to be counted as a block
                if len(block_compos) == 1:
                    is_divided = True
                if len(block_compos) > 1:
                    is_divided = True
                    block_id += 1
                    blocks.append(Block(id='b-'+str(block_id), compos=block_compos, slice_sub_block_direction=next_direction))
                    # remove blocked compos
                    non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))
                block_compos = []
            # extend block
            elif compo.left < divider < compo.right:
                divider = compo.right
            block_compos.append(compo)

        # if there are some sub-blocks, gather the left compos as a block 如果有一些子块，则将左侧的合成块收集为一个块
        if is_divided and len(block_compos) > 1:
            block_id += 1
            blocks.append(Block(id='b-'+str(block_id), compos=block_compos, slice_sub_block_direction=next_direction))
            # remove blocked compos
            non_blocked_compos = list(set(non_blocked_compos) - set(block_compos))

    return blocks, non_blocked_compos, direction, block_id + 1


class Block:
    def __init__(self, id, compos, slice_sub_block_direction='v', merge=False):
        self.block_id = id
        self.compos = compos                # list of Compo/List objs
        self.sub_blocks = []                # list of Block objs
        self.children = []                  # compos + sub_blocks
        self.compo_class = 'Block'

        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.width = None
        self.height = None

        self.next_block_id = int(id.split('-')[-1]) + 1

        # slice sub-block comprising multiple compos
        self.sub_blk_alignment = slice_sub_block_direction
        if not merge and slice_sub_block_direction != 'h':  # 如果只是合并那么就不再划分和排序了
            self.slice_sub_blocks()
        self.sort_compos_and_sub_blks()

        # print(self.html_id, slice_sub_block_direction)
        self.init_boundary()

    def init_boundary(self):
        self.top = int(min(self.compos + self.sub_blocks, key=lambda x: x.top).top)
        self.bottom = int(max(self.compos + self.sub_blocks, key=lambda x: x.bottom).bottom)
        self.left = int(min(self.compos + self.sub_blocks, key=lambda x: x.left).left)
        self.right = int(max(self.compos + self.sub_blocks, key=lambda x: x.right).right)
        self.height = int(self.bottom - self.top)
        self.width = int(self.right - self.left)

    def get_inner_compos(self):
        compos = []
        for child in self.children:
            if child.compo_class == 'List':
                compos += child.get_inner_compos()
            else:
                compos.append(child)
        return compos

    def wrap_info(self):
        info = {'id': self.block_id, 'class': 'Block', 'alignment': self.sub_blk_alignment, 'children': [],
                'location': {'left': int(self.left), 'right': int(self.right), 'top': int(self.top), 'bottom': int(self.bottom)}}
        for child in self.children:
            info['children'].append(child.wrap_info())
        return info

    '''
    ******************************
    ********** Children **********
    ******************************
    '''

    def slice_sub_blocks(self):
        '''
        slice the block into sub-blocks
        '''
        self.sub_blocks, self.compos, alignment_msg, self.next_block_id = slice_blocks(self.compos, int(self.block_id.split('-')[-1]), direction=self.sub_blk_alignment, change_alignment=True)

    def sort_compos_and_sub_blks(self):
        '''
        combine comps and sub_blocks w.r.t the slicing direction
        :param direction: slicing direction: 'v': from top to bottom; 'h': from left to right
        :return: children: sorted sub-blocks and compos
        '''
        if self.sub_blk_alignment == 'v':
            self.children = sorted(self.compos + self.sub_blocks, key=lambda x: x.top + x.height/2)
        elif self.sub_blk_alignment == 'h':
            self.children = sorted(self.compos + self.sub_blocks, key=lambda x: x.left + x.width/2)

    '''
    ******************************
    ******** Visualization *******
    ******************************
    '''
    def visualize_block(self, img, flag='line', show=False, color=(166,255,100)):
        fill_type = {'line': 2, 'block': -1}
        board = img.copy()
        draw_label(board, [self.left, self.top, self.right, self.bottom], color, text='Block', put_text=True)
        if show:
            cv2.imshow('compo', board)
            cv2.waitKey()
            cv2.destroyWindow('compo')
        return board

    def visualize_compos(self, img, flag='line', show=False, color=(0, 255, 0)):
        board = img.copy()
        for compo in self.compos:
            board = compo.visualize(board, flag, color=color)
        if show:
            cv2.imshow('blk_compos', board)
            cv2.waitKey()
            cv2.destroyWindow('blk_compos')
        return board

    def visualize_sub_blocks(self, img, flag='line', show=False, color=(0, 255, 0)):
        board = img.copy()
        for sub_block in self.sub_blocks:
            board = sub_block.visualize_block(board, flag, color=color)
        if show:
            cv2.imshow('blk_compos', board)
            cv2.waitKey()
            cv2.destroyWindow('blk_compos')
        return board

    def visualize_sub_blocks_and_compos(self, img, recursive=False, show=True):
        board = img.copy()
        board = self.visualize_block(board)
        board = self.visualize_compos(board, color=(0,0,200))
        for sub_block in self.sub_blocks:
            board = sub_block.visualize_block(board, color=(200,200,0))
        if show:
            print('Num of sub_block:%i; Num of element: %i' % (len(self.sub_blocks), len(self.compos)))
            cv2.imshow('sub_blocks', board)
            cv2.waitKey()
            cv2.destroyWindow('sub_blocks')

        if recursive:
            board = img.copy()
            for sub_block in self.sub_blocks:
                board = sub_block.visualize_sub_blocks_and_compos(board, recursive)
        return board
