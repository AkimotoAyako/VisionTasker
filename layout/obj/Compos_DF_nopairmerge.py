import json
import pandas as pd
import numpy as np
import copy
import cv2
from random import randint as rint
from sklearn.cluster import DBSCAN, KMeans
from collections import Counter
import layout.lib.repetition_recognition as rep
import layout.lib.draw as draw
import layout.lib.pairing as pairing
from PIL import Image
import clip
import torch
import re

class ComposDF:
    def __init__(self, json_file=None, json_data=None, gui_img=None):
        self.json_file = json_file
        self.json_data = json_data if json_data is not None else json.load(open(self.json_file))
        self.compos_json = self.json_data['compos']
        self.compos_dataframe = self.cvt_json_to_df()
        self.img = gui_img

        self.item_id = 0    # id of list item

    '''
    ***********************
    *** Basic Operation ***
    ***********************
    '''
    def copy(self):
        return copy.deepcopy(self)

    def reload_compos(self, json_file=None):
        if json_file is None:
            json_file = self.json_file
        self.json_data = json.load(open(json_file))
        self.compos_json = self.json_data['compos']
        self.compos_dataframe = self.cvt_json_to_df()

    def cvt_json_to_df(self):
        df = pd.DataFrame(columns=['id', 'class', 'column_min', 'column_max', 'row_min', 'row_max',
                                   'height', 'width', 'area', 'center', 'center_column', 'center_row', 'text_content',
                                   'children', 'parent', 'sub_class'])
        for i, compo in enumerate(self.compos_json):
            if 'clip_path' in compo:
                compo.pop('clip_path')
            if 'text_content' not in compo:
                compo['text_content'] = None
            if 'position' in compo:
                pos = compo['position']
                compo['column_min'], compo['column_max'] = int(pos['column_min']), int(pos['column_max'])
                compo['row_min'], compo['row_max'] = int(pos['row_min']), int(pos['row_max'])
                compo.pop('position')
            else:
                compo['column_min'], compo['column_max'] = int(compo['column_min']), int(compo['column_max'])
                compo['row_min'], compo['row_max'] = int(compo['row_min']), int(compo['row_max'])
            if 'children' not in compo or compo['children'] is None:
                compo['children'] = None
            else:
                compo['children'] = tuple(compo['children'])
            if 'parent' not in compo:
                compo['parent'] = None
            compo['height'], compo['width'] = int(compo['height']), int(compo['width'])
            compo['area'] = compo['height'] * compo['width']
            compo['center'] = ((compo['column_min'] + compo['column_max']) / 2, (compo['row_min'] + compo['row_max']) / 2)
            compo['center_column'] = compo['center'][0]
            compo['center_row'] = compo['center'][1]


            df.loc[i] = compo
        return df

    def to_csv(self, file):
        self.compos_dataframe.to_csv(file)

    def select_by_class(self, categories, no_parent=False, replace=False):
        df = self.compos_dataframe
        df = df[df['class'].isin(categories)]
        if no_parent:
            df = df[pd.isna(df['parent'])]
        if replace:
            self.compos_dataframe = df
        else:
            return df

    def calc_gap_in_group(self, compos=None):  # 计算每个组内组件的间隙
        '''
        Calculate the gaps between elements in each group
        '''
        if compos is None:
            compos = self.compos_dataframe
        compos['gap'] = -1
        groups = compos.groupby('group').groups
        for i in groups:  # 对每个组 根据vh的分类依据对应排序 计算组内间隙差（下一个组件的头-上一个组件的屁股）
            group = groups[i]
            if i != -1 and len(group) > 1:  # 有分组不单项
                group_compos = compos.loc[list(groups[i])]
                # check element's alignment (h or v) in each group
                if 'alignment_in_group' in group_compos:  # 有'v' 'h'的那一列的话
                    alignment_in_group = group_compos.iloc[0]['alignment_in_group']
                else:
                    alignment_in_group = group_compos.iloc[0]['alignment']
                # calculate element gaps in each group
                if alignment_in_group == 'v':
                    # sort elements vertically
                    group_compos = group_compos.sort_values('center_row')
                    for j in range(len(group_compos) - 1):
                        id = group_compos.iloc[j]['id']
                        compos.loc[id, 'gap'] = group_compos.iloc[j + 1]['row_min'] - group_compos.iloc[j]['row_max']
                else:
                    # sort elements horizontally
                    group_compos = group_compos.sort_values('center_column')
                    for j in range(len(group_compos) - 1):
                        id = group_compos.iloc[j]['id']
                        compos.loc[id, 'gap'] = group_compos.iloc[j + 1]['column_min'] - group_compos.iloc[j]['column_max']

    '''
    ******************
    *** Clustering ***
    ******************
    '''
    def cluster_dbscan_by_attr(self, attr, eps, min_samples=1, show=True, show_method='block'):
        '''
        Cluster elements by attributes using DBSCAN
        eg:compos_cp.cluster_dbscan_by_attr('center_column', eps=15, show=show)
        attr：要基于其进行聚类的属性名称。
        eps：DBSCAN 中的 "epsilon" 参数，用于定义邻域的大小。
        min_samples：DBSCAN 中的 "min_samples" 参数，表示在一个核心点的邻域内所需的最小样本数。
        show：一个布尔值，表示是否显示中间结果。
        show_method：用于指定显示聚类结果的方法，可以是 'line'（线形显示）或 'block'（块状显示）。
        '''
        # 选取中心位置那一列xc 如(x_1: 14+ x2: 55)/2=34.5
        x = np.reshape(list(self.compos_dataframe[attr]), (-1, 1))
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x)  # 根据这一列数值进行的聚类 得到labels比如[0 0 1 0 0 0...]
        tag = 'cluster_' + attr  # 写一个标签名 根据调用函数时给出的聚类要求
        self.compos_dataframe[tag] = clustering.labels_  # compos的这一列把标签写进去
        self.compos_dataframe[tag].astype(int)  # 格式统一是int
        if show:
            if show_method == 'line':
                self.visualize(gather_attr=tag, name=tag)
            elif show_method == 'block':
                self.visualize_fill(gather_attr=tag, name=tag)

    def cluster_dbscan_by_area(self, attr, img_size, eps, min_samples=1, show=True, show_method='block'):
        '''
        面积用 归一化的dbscan
        attr：要基于其进行聚类的属性名称。
        eps：DBSCAN 中的 "epsilon" 参数，用于定义邻域的大小。
        min_samples：DBSCAN 中的 "min_samples" 参数，表示在一个核心点的邻域内所需的最小样本数。
        show：一个布尔值，表示是否显示中间结果。
        show_method：用于指定显示聚类结果的方法，可以是 'line'（线形显示）或 'block'（块状显示）。
        '''
        # 选取中心位置那一列xc 如(x_1: 14+ x2: 55)/2=34.5
        x = np.reshape(list(self.compos_dataframe[attr]), (-1, 1))
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x/img_size)  # 根据这一列数值进行的聚类 得到labels比如[0 0 1 0 0 0...]
        tag = 'cluster_' + attr  # 写一个标签名 根据调用函数时给出的聚类要求
        self.compos_dataframe[tag] = clustering.labels_  # compos的这一列把标签写进去
        self.compos_dataframe[tag].astype(int)  # 格式统一是int
        if show:
            if show_method == 'line':
                self.visualize(gather_attr=tag, name=tag)
            elif show_method == 'block':
                self.visualize_fill(gather_attr=tag, name=tag)

    def cluster_area_by_relational_size(self, show=True, show_method='block'):
        self.compos_dataframe['cluster_area'] = -1  # 新建一列用于存放结果
        cluster_id = 0
        for i in range(len(self.compos_dataframe) - 1):  # 遍历每一对（i，j）元素
            compo_i = self.compos_dataframe.iloc[i]
            for j in range(i + 1, len(self.compos_dataframe)):
                compo_j = self.compos_dataframe.iloc[j]
                # 对于每一对元素，计算它们的最大面积和最小面积 如果最大面积小于最小面积的 1.8 倍（即，一个元素的面积接近于另一个元素的面积），则认为它们应该属于同一群组。
                if max(compo_i['area'], compo_j['area']) < min(compo_i['area'], compo_j['area']) * 1.8:
                    if compo_i['cluster_area'] != -1:  # 如果其中一个元素已经分配到了某个群组 则将另一个元素分配到相同的群组。
                        self.compos_dataframe.loc[compo_j['id'], 'cluster_area'] = compo_i['cluster_area']
                    elif compo_j['cluster_area'] != -1:  # 如果其中一个元素已经分配到了某个群组 则将另一个元素分配到相同的群组。
                        self.compos_dataframe.loc[compo_i['id'], 'cluster_area'] = compo_j['cluster_area']
                        compo_i = self.compos_dataframe.iloc[i]
                    else:  # 如果两个元素都没有分配到群组，将它们分配到新的群组，并将群组编号存储在它们的 'cluster_area' 列中。
                        self.compos_dataframe.loc[compo_i['id'], 'cluster_area'] = cluster_id
                        self.compos_dataframe.loc[compo_j['id'], 'cluster_area'] = cluster_id
                        compo_i = self.compos_dataframe.iloc[i]
                        cluster_id += 1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr='cluster_area', name='cluster_area')
            elif show_method == 'block':
                self.visualize_fill(gather_attr='cluster_area', name='cluster_area')

    def check_group_by_attr(self, target_attr='cluster_column_min', check_by='height', eps=20, show=True, show_method='block'):
        '''
        Double check the group by additional attribute, using DBSCAN upon the additional attribute to filter out the abnormal element
        @target_attr: gather element groups by this attribute
        @check_by: double check the gathered groups by this attribute
        @eps: EPS for DBSCAN
        target_attr：用于指定要根据其分组的主要属性。
        check_by：用于指定双重检查的属性，通常与尺寸或位置相关。
        eps：DBSCAN（密度聚类算法）的参数，用于确定元素是否属于同一群组。
        show：一个布尔值，表示是否显示中间结果。
        show_method：用于指定显示聚类结果的方法，可以是 'line'（线形显示）或 'block'（块状显示）。
        '''
        compos = self.compos_dataframe
        # 根据 target_attr 分组元素，将具有相同主要属性值的元素分成不同的群组。这些群组存储在 groups 中，其中每个群组由元素的ID列表组成。
        groups = compos.groupby(target_attr).groups  # {group name: list of compo ids}
        for i in groups:  # 对于每一个成型的分组
            if i != -1 and len(groups[i]) > 2:  # 除了未分组的群组 -1 和单个元素的群组
                group = groups[i]  # list of component ids in the group
                checking_attr = list(compos.loc[group][check_by])  # 使用check_by（一般是高度 height）属性双重检查
                # cluster compos height
                clustering = DBSCAN(eps=eps, min_samples=1).fit(np.reshape(checking_attr, (-1, 1)))  # 对check列聚类
                checking_attr_labels = list(clustering.labels_)
                checking_attr_label_count = dict((i, checking_attr_labels.count(i)) for i in checking_attr_labels)  # {label: frequency of label}

                # print(i, checking_attr, checking_attr_labels, checking_attr_label_count)
                for label in checking_attr_label_count:
                    # invalid compo if the compo's height is different from others
                    if checking_attr_label_count[label] < 2:  # 如果群组中的元素数小于 2（表示该群组中的元素差异较大），则将该群组中的所有元素的 target_attr 属性值设置为 -1，以将其标记为无效群组。
                        for j, lab in enumerate(checking_attr_labels):
                            if lab == label:
                                compos.loc[group[j], target_attr] = -1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr=target_attr, name=target_attr)
            elif show_method == 'block':
                self.visualize_fill(gather_attr=target_attr, name=target_attr)

    '''
    ****************
    *** Grouping ***
    ****************
    '''
    def group_by_clusters(self, cluster, alignment, show=True, show_method='block'):
        '''
        Group elements by cluster name
        Record element group in 'group' attribute
        eg: compos_cp.group_by_clusters(cluster=['cluster_area', 'cluster_center_column'], alignment='v', show=show)
        '''
        compos = self.compos_dataframe
        # 判断group列是否存在 存在的话接着之前的编号
        if 'group' not in compos.columns:
            self.compos_dataframe['group'] = -1
            group_id = 0
        else:
            group_id = compos['group'].max() + 1

        groups = self.compos_dataframe.groupby(cluster).groups  # 根据cluster给进来的列进行分组（nontext默认面积） 按照两列的组合分组 比如：（数值对）：对应索引 {(0, 0): [0, 1, 3, 4, 5, 7, 8], (0, 1): [9], (1, 1): [2, 6]}
        for i in groups:
            if len(groups[i]) > 1:
                self.compos_dataframe.loc[list(groups[i]), 'group'] = group_id
                self.compos_dataframe.loc[list(groups[i]), 'alignment'] = alignment  # 布局赋值 按照传入的参数定 标记group的原因
                group_id += 1
        self.compos_dataframe['group'].astype(int)

        if show:
            name = cluster if type(cluster) != list else '+'.join(cluster)
            if show_method == 'line':
                self.visualize(gather_attr='group', name=name)
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name=name)

    def closer_group_by_mean_area(self, compo_index, group1, group2, alignment):
        # 计算小组平均面积 2以内
        compo = self.compos_dataframe.loc[compo_index]
        g1 = group1[group1['id'] != compo['id']]
        g2 = group2[group2['id'] != compo['id']]
        # if len(cl2) == 1: return 1
        # elif len(cl1) == 1: return 2

        mean_area1 = g1['area'].mean()  # g1新的
        mean_area2 = g2['area'].mean()  # g2旧的

        # 1. if g1 and g2's area is not too different while one of g1 and g2 is length of 1, choose another group
        # 按照个数的判决方式：两个组面积绝对值相差不大（小于500）或面积相差小于2倍 且压倒性人数优势
        if ((abs(mean_area1 - mean_area2) <= 400)
            or max(mean_area1, mean_area2) < min(mean_area1, mean_area2) * 1.2) and \
                (max(len(g1), len(g2)) > min(len(g1), len(g2)) * 5 or min(len(g1), len(g2)) <= 2):
            # if not (g1['class'].iloc[0]==g2['class'].iloc[0]=='Compo' and abs(max(g1['area'].max(), g2['area'].max()) - min(g1['area'].min(), g2['area'].min())) >= 400 and min(len(g1), len(g2)) > 2):  # 对非文本加强约束
            if len(g1) < len(g2):  # 按照组的成员数来过滤 选择人多的
                return 2
            elif len(g1) > len(g2):
                return 1

        # 2.1 面积方差明显更相似的 放到那个组（适用于nontext）
        if compo['class'] != 'Text':
            compo_area = compo['area']
            if len(g1) * 3 < len(g2):  # 数量悬殊大 直接比平均
                if abs(mean_area1 - compo_area) < abs(mean_area2 - compo_area):
                    return 1
                else:
                    return 2
            else:
                if len(g1) == len(g2) and 0.5 < np.append(g1['area'].values, compo_area).std()/np.append(g2['area'].values, compo_area).std() < 3:
                    return 1  # 极端条件 数量相等且方差差异不大 横向右先
                if np.append(g1['area'].values, compo_area).std() > np.append(g2['area'].values, compo_area).std():
                    return 2
                elif np.append(g1['area'].values, compo_area).std() <= np.append(g2['area'].values, compo_area).std():
                    return 1

        # 2.2 对于文本 使用gap和布局判断接近程度
        if compo['class'] == 'Text':
            if alignment == 'v':  # 当前新组g1是垂直
                gap1 = np.sort(np.append(g1['row_min'].values, compo['row_min']))[1:] - np.sort(np.append(g1['row_max'].values, compo['row_max']))[:-1]
                gap2 = np.sort(np.append(g2['column_min'].values, compo['column_min']))[1:] - np.sort(np.append(g2['column_max'].values, compo['column_max']))[:-1]
            else:
                gap1 = np.sort(np.append(g1['column_min'].values, compo['column_min']))[1:] - np.sort(np.append(g1['column_max'].values, compo['column_max']))[:-1]
                gap2 = np.sort(np.append(g2['row_min'].values, compo['row_min']))[1:] - np.sort(np.append(g2['row_max'].values, compo['row_max']))[:-1]
            if gap1.mean() > gap2.mean() and not (len(g2) == 1 and len(g1) > 1):
                return 2
            else:
                return 1

        # # 3. choose the group with similar area 选不了 只能看面积更接近谁
        # compo_area = compo['area']
        # if abs(compo_area - mean_area1) < abs(compo_area - mean_area2):  # 更接近1
        #     return 1
        # return 2

    def group_by_clusters_conflict(self, cluster, alignment, show=True, show_method='block'):
        '''
        If an element is clustered into multiple clusters, assign it to the cluster where the element's area is mostly like others
        Then form the cluster as a group
        eg: compos_cp.group_by_clusters_conflict(cluster=['cluster_area', 'cluster_center_row'], alignment='h', show=show)
        '''
        compos = self.compos_dataframe
        group_id = compos['group'].max() + 1

        compo_new_group = {}  # {'id':'new group id'}
        groups = self.compos_dataframe.groupby(cluster).groups  # 和之前类似 按面积和列坐标组分类
        for i in groups:
            if len(groups[i]) > 1:  # 如果不是单人组
                member_num = len(groups[i])  # 记一下分组的人数
                for j in list(groups[i]):  # 细看一下分组里的成员
                    # if the compo hasn't been grouped, then assign it to a new group and no conflict happens 如果没被分组 那就加一个
                    if compos.loc[j, 'group'] == -1:  # 如果没被分组 那就加一个
                        compos.loc[j, 'group'] = group_id
                        compos.loc[j, 'alignment'] = alignment
                    # conflict raises if a component can be grouped into multiple groups
                    # then double check it by the average area of the groups
                    else:
                        # keep in the previous group if it is the only member in a new group
                        if member_num <= 1:  # 如果是已经是单人组那就放过他 让他在以前的组呆着（这句话是给295行那个member_num -= 1兜底的
                            continue
                        # close to the current cluster
                        prev_group = compos[compos['group'] == compos.loc[j, 'group']]  # 目前控件提取原本的分组
                        # 根据它们所在群组的平均面积（通过 closer_group_by_mean_area 函数来计算），将元素分配到最符合条件的群组中。
                        if self.closer_group_by_mean_area(j, compos.loc[list(groups[i])], prev_group, alignment) == 1:
                            # compos.loc[j, 'group'] = group_id
                            # compos.loc[j, 'alignment'] = alignment
                            compo_new_group[j] = group_id  # 如果更符合新的组那么就加入
                        else:
                            member_num -= 1  # 这个组减去一个人 保留原始的组队
                group_id += 1

        for i in compo_new_group:
            compos.loc[i, 'group'] = compo_new_group[i]  # 根据上面写好的分组写入表格
            compos.loc[i, 'alignment'] = alignment
        self.compos_dataframe['group'].astype(int)

        if show:
            name = cluster if type(cluster) != list else '+'.join(cluster)
            if show_method == 'line':
                self.visualize(gather_attr='group', name=name)
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name=name)

    '''
    *********************************
    *** Similar Group Recognition ***
    *********************************
    '''
    #  第2.2步 识别block之间的相似性
    def recognize_similar_blocks(self):
        '''
        Recognize similar Blocks (Containers) that contains similar elements
        '''
        df = self.compos_dataframe
        blocks = df[(df['class']=='Block') & (~pd.isna(df['children'])) & (df['children']!=-1)]  # 筛选是block、有children的元素
        children_list = []
        connections_list = []
        for i in range(len(blocks)):  # 如果有孩子 它提取块内的子元素（对应children的id），并按照它们的 'center_row'（竖直中心位置）进行排序。
            children = df[df['id'].isin(blocks.iloc[i]['children'])].sort_values('center_row')
            children_list.append(children)
            # calculate children connections in each block
            connections_list.append(rep.calc_connections(children))  # 计算连接

        # match children connections
        start_pair_id = max(self.compos_dataframe['group_pair']) if 'group_pair' in self.compos_dataframe else 0  # 如果已经有group pair了那接着编号
        paired_blocks = rep.recog_repetition_block_by_children_connections(children_list, connections_list, start_pair_id)  # 调用函数 根据子元素的连接状态识别相似的block

        # merge the pairing result into the original dataframe 如果成功识别到相似块并进行了配对，它将配对结果合并到原始数据框中
        df_all = self.compos_dataframe
        if paired_blocks is not None:
            if 'group_pair' not in self.compos_dataframe:  # 如果识别到了就新建一列表示组的关系
                df_all = self.compos_dataframe.merge(paired_blocks, how='left')
            else:
                df_all.loc[df_all[df_all['id'].isin(paired_blocks['id'])]['id'], 'group_pair'] = paired_blocks['group_pair']

            group_id = 0
            # include the parent block in the pair
            children_group = paired_blocks.groupby('parent').groups  # {parent block id: [children id]} 按照parent属性（父块的标识）分组 字典形式
            for i in children_group:
                # set the block as a group  将块分配到一个新的组（group），标记为 'c-' + str(group_id)，其中 group_id 递增
                df_all.loc[i, 'group'] = 'c-' + str(group_id)  # 父块标记
                df_all.loc[children_group[i], 'group'] = 'c-' + str(group_id)  # 多个子块一起标记
                group_id += 1  # 组号+1

                df_all.loc[i, 'group_pair'] = df_all.loc[children_group[i][0], 'group_pair']  # 将父块与其子块关联起来，以表示它们是一对
                # pair to the parent block
                df_all.loc[i, 'pair_to'] = i
                df_all.loc[children_group[i], 'pair_to'] = i

            df_all = df_all.fillna(-1)  # 补全剩下的
            df_all['group_pair'] = df_all['group_pair'].astype(int)  # 格式统一int
        else:
            if 'group_pair' not in self.compos_dataframe:
                df_all['group_pair'] = -1
        self.compos_dataframe = df_all   # 更新大表

   #  第2.1步 最开始的使用dbscan做的分组
    def recognize_element_groups_by_clustering(self, show=False):
        '''
        Recognize repetitive layout of elements that are not contained in Block by clustering
        '''
        # 把控件和文本转换成df格式 df是合并的表格
        df_nontext = rep.recog_repetition_nontext(self, show, only_non_contained_compo=False)
        df_text = rep.recog_repetition_text(self, show, only_non_contained_compo=False)
        df = self.compos_dataframe

        df['alignment'] = np.nan  # 在数据框中添加了两空列：'alignment' 和 'gap'
        df['gap'] = np.nan  # 这些列将用于存储分组和排列信息
        if 'alignment' in df_nontext:  # 非文本的布局合并
            df.loc[df['alignment'].isna(), 'alignment'] = df_nontext['alignment']
        if 'gap' in df_nontext:  # 没有gap
            df.loc[df['gap'].isna(), 'gap'] = df_nontext['gap']
        df = df.merge(df_nontext, how='left')  # 整合初步分组后的信息 各种group进来了 因为nontext和text的分组方式不一样所以用的merge 按索引多列合并
        if 'alignment' in df_text:
            df.loc[df['alignment'].isna(), 'alignment'] = df_text['alignment']
        if 'gap' in df_text:
            df.loc[df['gap'].isna(), 'gap'] = df_text['gap']
        df = df.merge(df_text, how='left')
        df.rename({'alignment': 'alignment_in_group'}, axis=1, inplace=True)

        # clean and rename attributes
        df = df.fillna(-1)
        df['group'] = -1  # 现在是一个大分组 根据nontext和text的分组信息 生成整体的nt和t打头的分组列
        for i in range(len(df)):
            if df.iloc[i]['group_nontext'] != -1:
                df.loc[i, 'group'] = 'nt-' + str(int(df.iloc[i]['group_nontext']))
            elif df.iloc[i]['group_text'] != -1:
                df.loc[i, 'group'] = 't-' + str(int(df.iloc[i]['group_text']))
        groups = df.groupby('group').groups  # 根据刚刚生成的 把组号搞出来
        for i in groups:
            if len(groups[i]) == 1:
                df.loc[groups[i], 'group'] = -1  # 如果这个组只出现一次那么就视为无效分组

        df = df.drop(list(df.filter(like='cluster')), axis=1)  # 删除以 'cluster' 开头的列
        df = df.drop(columns=['group_nontext', 'group_text'])  # 和 'group_nontext'、'group_text' 列
        self.compos_dataframe = df  # 更新信息

        # slip a group into several ones according to compo gaps if necessary 必要时，根据组合间隙将一组分成几组
        self.regroup_compos_by_compos_gap()  # 我改了:二次划分
        # search and add compos into a group according to compo gaps in the group 根据组中的成分间隙，搜索并将成分添加到组中
        # self.add_missed_compo_to_group_by_gaps(search_outside=False)
        # check group validity by compos gaps in the group, the gaps among compos in a group should be similar 通过组内成分间的差距检查组的有效性，组内成分间的差距应相似
        # self.check_group_validity_by_compos_gap()  # 我改了

    '''
    ********************************
    *** Refine Repetition Result ***
    ********************************
    '''
    def regroup_left_compos_by_cluster(self, cluster, alignment, show=True, show_method='block'):
        # eg: compos_cp.regroup_left_compos_by_cluster('cluster_column_min', alignment='v', show=show)
        compos = self.compos_dataframe
        group_id = compos['group'].max() + 1

        # select left compos that in a group only containing itself
        groups = compos.groupby('group').groups
        left_compos_id = []
        for i in groups:  # 如果是单元素组 就把他拎出来
            if i != -1 and len(groups[i]) == 1:
                left_compos_id += list(groups[i])
        left_compos = compos.loc[left_compos_id]

        # regroup the left compos by given cluster
        groups = left_compos.groupby(cluster).groups  # 用给定的再次分组 比如左对齐（列最小）
        for i in groups:
            if len(groups[i]) > 1:
                self.compos_dataframe.loc[list(groups[i]), 'group'] = group_id
                self.compos_dataframe.loc[list(groups[i]), 'alignment'] = alignment
                group_id += 1
        self.compos_dataframe['group'].astype(int)

        if show:
            name = cluster if type(cluster) != list else '+'.join(cluster)
            if show_method == 'line':
                self.visualize(gather_attr='group', name=name)
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name=name)

    def regroup_compos_by_compos_gap(self):
        '''
        slip a group into several ones according to compo gaps if necessary
        根据间隙把小组再划分
        '''
        self.calc_gap_in_group()  # 首先通过调用 calc_gap_in_group 方法计算每个组内组件的间隙。
        compos = self.compos_dataframe
        groups = compos.groupby('group').groups  # {group name: list of compo ids}
        for i in groups:
            if i != -1 and len(groups[i]) > 2:  # 首先，它检查组内是否有至少3个组件（len(groups[i]) > 2），因为只有在组件数量大于等于3时才需要重新分组
                group = groups[i]  # list of component ids in the group
                group_compos = compos.loc[group]
                if group_compos.iloc[0]['alignment_in_group'] == 'v':  # 根据对齐方式 垂直对齐的根据组件y坐标排序 反之亦然
                    group_compos = group_compos.sort_values('center_row')
                else:
                    group_compos = group_compos.sort_values('center_column')
                gaps = list(group_compos['gap'])  # 提取刚刚算的gap
                # cluster compos gaps 聚类 先把list的gap去掉最后一个没有间隙的 竖过来成 array
                # clustering = DBSCAN(eps=int(self.img.shape[0]*0.025), min_samples=1).fit(np.reshape(gaps[:-1], (-1, 1)))  # 我改了 10变20int(self.img.shape[0]*0.025)
                # 既然是分段 只要确认gap是行还是列就行 所以可以用kmeans 只分两个类
                group_by_para_flag, gap_labels = False, []
                if len(set(gaps[:-1])) != 1:  # 如果gaps值都不一样 才判断是否要按照段间距分组
                    clustering = KMeans(n_clusters=2).fit(np.reshape(gaps[:-1], (-1, 1)))
                    gap_labels = list(clustering.labels_)
                    # 拿到gap_labels 先确认哪个是段间距 哪个是行间距 计算平均间距 array([1, 1, 0, 1, 1, 0], dtype=int32)
                    l0_avgap = sum([gaps[:-1][j] for j, gl in enumerate(gap_labels) if gl == 0]) / gap_labels.count(0)  # 标签为0的gap均值
                    l1_avgap = sum([gaps[:-1][j] for j, gl in enumerate(gap_labels) if gl == 1]) / gap_labels.count(1)  # 标签为1的gap均值
                    para_flag = 0 if l0_avgap > l1_avgap else 1  # 如果0的平均间距大于1 那么0是段落结束符
                    if para_flag == 0:
                        gap_labels = list(1-np.array(gap_labels))  # 如果0大就取反
                    # 情况总结 段间距分组、最大间距分组
                    group_by_para_flag = (  (((l0_avgap > 1.5 * l1_avgap or l1_avgap > 1.5 * l0_avgap) and group_compos.iloc[0]['alignment_in_group'] == 'v')
                                            or (((l0_avgap > 4 * l1_avgap or l1_avgap > 4 * l0_avgap) or abs(l0_avgap-l1_avgap) > 50) and group_compos.iloc[0]['alignment_in_group'] == 'h'))
                                              and ((min(abs(l0_avgap), abs(l1_avgap)) != 0) or (abs(l0_avgap - l1_avgap) > 10 and min(abs(l0_avgap), abs(l1_avgap)) == 0))  )  # 行间距和段间距必须有明显差异
                regroup_by_gap_labels = [1 if g > 70 and group_compos.iloc[0]['alignment_in_group'] == 'v' else 0 for g in gaps[:-1]]  # 计算大于阈值的最大间距分组标签
                group_by_gap_flag = sum(regroup_by_gap_labels) != 0  # 是否需要段间距分组
                if group_by_para_flag or group_by_gap_flag:
                    # 仅最大间距分组
                    if not group_by_para_flag and group_by_gap_flag:
                        gap_labels = regroup_by_gap_labels
                    elif group_by_para_flag and group_by_gap_flag:
                        gap_labels = [max(x, y) for x, y in zip(gap_labels, regroup_by_gap_labels)]  # 两个label或操作
                    label_cnt, gl_new = 0, [0]  # 第一个先成一段的开始
                    for gl in gap_labels:
                        if gl != 1:  # 如果没有隔开一段
                            gl_new.append(label_cnt)
                        else:
                            label_cnt += 1
                            gl_new.append(label_cnt)
                    gap_labels = gl_new
                    gap_label_count = dict((i, gap_labels.count(i)) for i in gap_labels)  # 统计每个标签的出现次数{label: frequency of label}
                    # 呼呼 我改完了
                    new_group_num = 0
                    for label in gap_label_count:  # 对于每个标签
                        if gap_label_count[label] >= 2:  # 如果该标签的出现次数大于等于2，表示有至少2个组件具有相似的间隙
                            # select compos with same gaps to form a new group
                            new_group = pd.DataFrame()  # 然后，它选择具有相同间隙的组件，形成一个新的组（new_group）。
                            for j, lab in enumerate(gap_labels):  # 对于现在讨论的标签
                                if lab == label:
                                    new_group = new_group.append(compos.loc[group_compos.iloc[j]['id']])  # 查id从原始表格里取数据 放入new_group
                                    self.calc_gap_in_group(new_group)  # 更新一下new_group的间隙
                            new_gap = list(new_group['gap'])
                            # check if the new group is valid by gaps (should have the same gaps between compos)
                            is_valid_group = True  # 检查有效性
                            for j in range(1, len(new_gap) - 1):
                                if abs(new_gap[j] - new_gap[j - 1]) > 50:  # 绝对值之类的要求
                                    is_valid_group = False
                                    break
                            if is_valid_group:  # 如果新的组有效 需要拆分原始的组
                                # do not change the first group
                                if new_group_num >= 1:
                                    compos.loc[new_group['id'], 'group'] += '-' + str(new_group_num)  # 加一个后缀
                            new_group_num += 1

                        elif gap_label_count[label] == 1:
                            if new_group_num >= 1:
                                compos.loc[group_compos.iloc[gap_labels.index(label)]['id'], 'group'] += '-' + str(new_group_num)  # 加一个后缀
                            new_group_num += 1

                        # check the last compo 不用了已经 呵呵
                        # 检查最后一个组件（last_compo）是否应该与新组（new_group）合并。 因为gap的存储方式是A的gap是B-A的空隙 因此聚类之后总会丢掉最后一个
                        # 合并的条件是，最后一个组件的位置与新组的位置之间的间隙 与新组的第一个组件的间隙 之间的差异小于10。
                        # last_compo = compos.loc[group[-1]]
                        # if new_group.iloc[-1]['alignment_in_group'] == 'v':
                        #     new_group.sort_values('center_row')
                        #     gap_with_the_last = last_compo['row_min'] - new_group.iloc[-1]['row_max']
                        # else:
                        #     new_group.sort_values('center_column')
                        #     gap_with_the_last = last_compo['column_min'] - new_group.iloc[-1]['column_max']
                        # if abs(gap_with_the_last - new_group.iloc[0]['gap']) < 10:
                        #     compos.loc[last_compo['id'], 'group'] += '-' + str(new_group_num)
                       # new_group_num += 1
            elif len(groups[i]) == 2:
                group = groups[i]  # list of component ids in the group
                group_compos = compos.loc[group]
                if group_compos.iloc[0]['alignment_in_group'] == 'v':  # 根据对齐方式 垂直对齐的根据组件y坐标排序 反之亦然
                    group_compos = group_compos.sort_values('center_row')
                else:
                    group_compos = group_compos.sort_values('center_column')
                gaps = list(group_compos['gap'])
                if gaps[0] > 70:
                    compos.loc[group_compos.iloc[1]['id'], 'group'] += '-' + str(1)  # 加一个后缀

    def search_possible_compo(self, anchor_compo, approximate_gap, direction='next'):
        compos = self.compos_dataframe
        if 'alignment_in_group' in anchor_compo:
            alignment_in_group = anchor_compo['alignment_in_group']
        else:
            alignment_in_group = anchor_compo['alignment']

        if alignment_in_group == 'v':
            # search below
            if direction == 'next':
                approx_row = anchor_compo['row_max'] + approximate_gap + 0.5 * anchor_compo['height']
            # search above
            else:
                approx_row = anchor_compo['row_min'] - (approximate_gap + 0.5 * anchor_compo['height'])
            if approx_row >= self.img.shape[0] or approx_row <= 0: return None

            for i in range(len(compos)):
                compo = compos.iloc[i]
                if max(compo['area'], anchor_compo['area']) < min(compo['area'], anchor_compo['area']) * 3 and\
                        compo['row_min'] < approx_row < compo['row_max'] and \
                        max(compo['column_min'], anchor_compo['column_min']) < min(compo['column_max'], anchor_compo['column_max']):
                    return compo
        else:
            # search right
            if direction == 'next':
                approx_column = anchor_compo['column_max'] + approximate_gap + 0.5 * anchor_compo['width']
            # search left
            else:
                approx_column = anchor_compo['column_min'] - (approximate_gap + 0.5 * anchor_compo['width'])
            if approx_column >= self.img.shape[1] or approx_column <= 0: return None

            for i in range(len(compos)):
                compo = compos.iloc[i]
                if max(compo['area'], anchor_compo['area']) < min(compo['area'], anchor_compo['area']) * 3 and \
                        compo['column_min'] < approx_column < compo['column_max'] and \
                        max(compo['row_min'], anchor_compo['row_min']) < min(compo['row_max'], anchor_compo['row_max']):
                    return compo
        return None

    def add_missed_compo_to_group_by_gaps(self, search_outside=True):
        '''
        search and add compos into a group according to compo gaps in the group
        '''
        self.calc_gap_in_group()
        compos = self.compos_dataframe
        groups = compos.groupby('group').groups  # {group name: list of compo ids}
        for i in groups:
            if i != -1 and len(groups[i]) >= 2:
                group = groups[i]  # list of component ids in the group
                group_compos = compos.loc[group]
                if group_compos.iloc[0]['alignment_in_group'] == 'v':
                    group_compos = group_compos.sort_values('center_row')
                else:
                    group_compos = group_compos.sort_values('center_column')
                gaps = list(group_compos['gap'])

                # cluster compos gaps
                clustering = DBSCAN(eps=10, min_samples=1).fit(np.reshape(gaps[:-1], (-1, 1)))
                gap_labels = list(clustering.labels_)
                gap_label_count = dict((i, gap_labels.count(i)) for i in gap_labels)  # {label: frequency of label}
                # get the most counted label
                max_label = max(gap_label_count.items(), key=lambda k: k[1])  # (label id, count number)
                # if there are more than half compos with that similar gap, find possibly missed compos by gap
                if max_label[1] > len(group) * 0.5:
                    anchor_label = max_label[0]
                    # calculate the mean gap with the
                    mean_gap = 0
                    for k, l in enumerate(gap_labels):
                        if gap_labels[k] == anchor_label:
                            mean_gap += gaps[k]
                    mean_gap = int(mean_gap / max_label[1])

                    for k, gap_label in enumerate(gap_labels):
                        # search possible compo from the compo with a different label
                        if gap_label != anchor_label:
                            possible_compo = self.search_possible_compo(anchor_compo=compos.loc[group_compos.iloc[k]['id']], approximate_gap=mean_gap)
                            if possible_compo is not None:
                                compos.loc[possible_compo['id'], 'group'] = i
                                compos.loc[possible_compo['id'], 'alignment_in_group'] = compos.loc[group_compos.iloc[k]['id'], 'alignment_in_group']

                # search possible compos outside the group
                if search_outside:
                    group_compos = compos.loc[group]
                    if group_compos.iloc[0]['alignment_in_group'] == 'v': group_compos = group_compos.sort_values('center_row')
                    else: group_compos = group_compos.sort_values('center_column')
                    # search previously (left or above)
                    possible_compo = self.search_possible_compo(group_compos.iloc[0], group_compos.iloc[0]['gap'], direction='prev')
                    if possible_compo is not None:
                        compos.loc[possible_compo['id'], 'group'] = i
                        compos.loc[possible_compo['id'], 'alignment_in_group'] = group_compos.iloc[0]['alignment_in_group']
                    # search next (right or below)
                    possible_compo = self.search_possible_compo(group_compos.iloc[-1], group_compos.iloc[0]['gap'], direction='next')
                    if possible_compo is not None:
                        compos.loc[possible_compo['id'], 'group'] = i
                        compos.loc[possible_compo['id'], 'alignment_in_group'] = group_compos.iloc[0]['alignment_in_group']

    '''
    ***********************
    *** Validate Groups ***
    ***********************
    '''
    def check_group_of_two_compos_validity_by_areas(self, show=False, show_method='block'):
        groups = self.compos_dataframe.groupby('group').groups
        for i in groups:
            # if the group only has two elements, check if it's valid by elements' areas
            if i != -1 and len(groups[i]) >= 2:  # 对于分了组的且不是单元素组的
                compos = self.compos_dataframe.loc[groups[i]]
                # if the two are too different in area, mark the group as invalid 计算这个组里面的面积差异 如果大于2.2则错
                if compos['area'].max() - compos['area'].min() > 500 and compos['area'].max() > compos['area'].min() * 3:
                    self.compos_dataframe.loc[groups[i], 'group'] = -1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr='group', name='valid-two-compos')
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name='valid-two-compos')

    def check_unpaired_group_of_two_compos_validity_by_min_area(self, show=False, show_method='block'):
        groups = self.compos_dataframe.groupby('group').groups
        for i in groups:
            # if the group is unpaired and only has two elements, check if it's valid by elements' areas
            if i != -1 and len(groups[i]) >= 2:
                compos = self.compos_dataframe.loc[groups[i]]
                if compos.iloc[0]['group_pair'] == -1:
                    # if the two elements are in vertical alignment, then remove the group
                    if compos.iloc[0]['alignment_in_group'] == 'v':
                        self.compos_dataframe.loc[groups[i], 'group'] = -1
                    # if the two are too different in area, mark the group as invalid
                    elif compos['area'].min() < 150 or compos['area'].max() / compos['area'].min() > 1.5:
                        self.compos_dataframe.loc[groups[i], 'group'] = -1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr='group', name='valid-two-compos')
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name='valid-two-compos')

    def remove_unpaired_group_of_two_compos(self, show=False, show_method='block'):
        # 检测并删除那些未配对且只包含两个元素的组
        groups = self.compos_dataframe.groupby('group').groups
        for i in groups:
            # if the group is unpaired and only has two elements, remove the group
            if i != -1 and len(groups[i]) == 2:
                compos = self.compos_dataframe.loc[groups[i]]
                if compos.iloc[0]['group_pair'] == -1 and not (compos.iloc[0]['gap'] < 25 and compos.iloc[0]['class'] == 'Text'):  # 没有配对的组
                    self.compos_dataframe.loc[groups[i], 'group'] = -1
                    self.compos_dataframe.loc[groups[i], 'sup_group'] = -1
        if show:
            if show_method == 'line':
                self.visualize(gather_attr='group', name='valid-two-compos')
            elif show_method == 'block':
                self.visualize_fill(gather_attr='group', name='valid-two-compos')

    # def check_group_validity_by_compos_gap(self, show=False, show_method='block'):
    #     '''
    #     check group validity by compos gaps in the group, the gaps among compos in a group should be similar
    #     '''
    #     changed = False
    #     self.calc_gap_in_group()
    #     compos = self.compos_dataframe
    #     groups = compos.groupby('group').groups  # {group name: list of compo ids}
    #     for i in groups:
    #         if i == -1: continue
    #         if len(groups[i]) == 1:  # 只有一个组件--移除
    #             compos.loc[groups[i][0], 'group'] = -1
    #         elif len(groups[i]) > 2:  # 多于两个
    #             group = groups[i]  # list of component ids in the group
    #             group_compos = compos.loc[group]
    #             if group_compos.iloc[0]['alignment_in_group'] == 'v':  # 又是按顺序排序
    #                 group_compos = group_compos.sort_values('center_row')
    #             else:
    #                 group_compos = group_compos.sort_values('center_column')
    #             gaps = list(group_compos['gap'])
    #
    #             # cluster compos gaps
    #             eps = 30 if group_compos.iloc[0]['class'] == 'Text' else 15  # 对于文本类来说邻域更宽松
    #             clustering = DBSCAN(eps=eps, min_samples=1).fit(np.reshape(gaps[:-1], (-1, 1)))  # 再次使用DBSCAN 计算组内组件之间的间隙
    #             gap_labels = list(clustering.labels_)
    #             gap_label_count = dict((i, gap_labels.count(i)) for i in gap_labels)  # {label: frequency of label}
    #
    #             for label in gap_label_count:
    #                 # invalid compo if the compo's gap with others is different from others
    #                 if gap_label_count[label] < 2:  # 如果某个间隙模式的出现频率小于 2，即组内的大多数组件具有不同的间隙，它将这些组件从组中移除，并将其所在组的标识设置为 -1，表示无效组
    #                     for j, lab in enumerate(gap_labels):
    #                         if lab == label:
    #                             compos.loc[group_compos.iloc[j]['id'], 'group'] = -1
    #                             changed = True
    #
    #     self.check_group_of_two_compos_validity_by_areas(show=show)  # 进一步检查包含两个组件的组是否有效，主要是通过比较这两个组件的面积差异来判断
    #
    #     # recursively run till no changes
    #     if changed:
    #         self.check_group_validity_by_compos_gap()  # 如果有无效的 就循环调用到都有效为止
    #
    #     if show:
    #         if show_method == 'line':
    #             self.visualize(gather_attr='group', name='valid')
    #         elif show_method == 'block':
    #             self.visualize_fill(gather_attr='group', name='valid')

    # 检查未配对的组的有效性，通过查看它们是否与其他 未分组 的元素交错（或者说交叉）
    def check_unpaired_group_validity_by_interleaving(self):
        compos = self.compos_dataframe
        groups = compos.groupby('group').groups  # {group name: list of compo ids}
        if -1 not in groups:
            return
        ungrouped_compos = groups[-1]  # list of ungrouped compo id 未分组的元素存储
        for i in groups:
            # only check unpaired groups
            if i == -1 or compos.loc[groups[i]].iloc[0]['group_pair'] != -1 or len(groups[i]) > 2: continue  # 只检查未配对的分组（group_pair为 -1，且分组中的元素没有配对，或分组中的元素数量大于 2 的情况除外）
            group_compos = compos.loc[groups[i]]
            group_bound = [group_compos['column_min'].min(), group_compos['row_min'].min(), group_compos['column_max'].max(), group_compos['row_max'].max()]  # 获取整个组的边界
            for j in ungrouped_compos:  # 只检查未配对的分组
                c = compos.loc[j]
                # intersection area 看交集面积大于0.7就删除分组
                left = max(group_bound[0], c['column_min'])
                top = max(group_bound[1], c['row_min'])
                right = min(group_bound[2], c['column_max'])
                bottom = min(group_bound[3], c['row_max'])
                width = max(0, right - left)
                height = max(0, bottom - top)
                # if intersected
                if width == 0 or height == 0:
                    continue
                else:
                    if width * height / c['area'] > 0.7:
                        compos.loc[groups[i], 'group'] = -1

    def remove_invalid_groups(self):  # 需要改：判断group之间的交叉
        # self.check_unpaired_group_of_two_compos_validity_by_min_area()
        self.remove_unpaired_group_of_two_compos()  # 删掉没有配对的 只有两个元素的集合（恐怕是特别定制版）
        self.check_unpaired_group_validity_by_interleaving()  # 检查未配对的组的有效性 通过判断是否和未分组的元素交叉

    def add_missed_compos_by_checking_group_item(self):
        df = self.compos_dataframe
        pairs = df.groupby('group_pair').groups
        for p in pairs:
            if p == -1: continue
            pair = pairs[p]
            pair_all_compos = df.loc[pair]
            paired_groups = pair_all_compos.groupby('group').groups

            max_group_compo_num = max([len(paired_groups[i]) for i in paired_groups])  # 找到当前组中包含最多控件的子组，记录其包含的控件数量。
            for i in paired_groups:  # 遍历已配对的分组
                # Identify abnormal groups that have fewer compos that others do
                if len(paired_groups[i]) < max_group_compo_num:
                    # calculate the related position of the group compos in their paired item 检查当前子组是否包含的控件数量少于最多控件数。如果是，说明这是一个异常组，可能存在缺失的控件。
                    group_compos = df.loc[paired_groups[i]]  # compos in the abnormal group 获取当前异常组内的所有控件。
                    compo_related_pos = pairing.calc_compo_related_position_in_its_paired_item(group_compos, pair_all_compos)  # 计算异常组中控件相对于其相关项的位置。 (column_min, row_min, column_max, row_max)

                    # identify the abnormal item and its position
                    abnormal_items = pair_all_compos[~pair_all_compos['list_item'].isin(list(group_compos['list_item']))]  # 找到未包含在异常组的相关项。
                    abnormal_items_grp = abnormal_items.groupby('list_item').groups  # 将这些未包含在异常组中的相关项按照list_item进行分组。
                    for j in abnormal_items_grp:
                        abnormal_item = abnormal_items.loc[abnormal_items_grp[j]]
                        abnormal_item_pos = abnormal_item['column_min'].min(), abnormal_item['row_min'].min()  # 获取当前相关项的位置

                        # 计算相关项的潜在位置（基于控件相对位置和相关项的绝对位置。） calculate the potential missed compo area based on the related compo position and the absolute item position
                        potential_missed_compo_area = (compo_related_pos[0] + abnormal_item_pos[0], compo_related_pos[1] + abnormal_item_pos[1],
                                                       compo_related_pos[2] + abnormal_item_pos[0], compo_related_pos[3] + abnormal_item_pos[1])

                        # find the potential missed compo through iou with the potential_missed_compo_area 通过与潜在缺失控件区域的交并比（IOU）来查找可能的缺失控件
                        missed_compo_id = pairing.find_missed_compo_by_iou_with_potential_area(potential_missed_compo_area, df)
                        if missed_compo_id and df.loc[missed_compo_id, 'class'] == group_compos.iloc[0]['class']:  #   类别不一定一样
                            # print(df.loc[missed_compo_id, 'class'], group_compos.iloc[0]['class'], i, j)
                            df.loc[missed_compo_id, 'group_pair'] = p
                            df.loc[missed_compo_id, 'group'] = i
                            df.loc[missed_compo_id, 'list_item'] = j

    '''
    ******************************
    ******** Pair Groups *********
    ******************************
    '''
    def pair_groups(self):
        # gather by same groups
        self.compos_dataframe['sup_group'] = [g_name.split('-')[0]+'-'+g_name.split('-')[1] if g_name != -1 else -1
                                              for g_name in self.compos_dataframe['group']]  # 我改了 这里pair-group先看大组
        all_groups = self.split_groups('sup_group')  # 调用 split_groups('group') 方法，将数据中的元素按照它们所属的组（group）进行分组，返回list[df,df,df,...]，其中每个组的元素都被收集在一起

        def sort_by_row_min(df):
            return df.iloc[0]['row_min']
        # 对DataFrame列表按照'row_min'进行排序
        all_groups = sorted(all_groups, key=sort_by_row_min)

        # pairing between groups
        if 'group_pair' in self.compos_dataframe:  # 如果有组的配对了 接着上一个编号 否则从0开始
            start_pair_id = max(self.compos_dataframe['group_pair'])
        else:
            start_pair_id = 0
        pairs = pairing.pair_matching_within_groups(all_groups, start_pair_id)  # 调用函数 在组之间查找相似性来尝试进行组的配对
        # merge the pairing result into the original dataframe
        df_all = self.compos_dataframe  # 取原版数据df
        if pairs is not None:  # 如果有配对结果
            if 'group_pair' not in self.compos_dataframe:  # 如果之前没有列就新建一列
                df_all = self.compos_dataframe.merge(pairs, how='left')
            else:
                df_all.loc[df_all[df_all['id'].isin(pairs['id'])]['id'], 'group_pair'] = pairs['group_pair']  # 更新组配对结果
                df_all.loc[df_all[df_all['id'].isin(pairs['id'])]['id'], 'pair_to'] = pairs['pair_to']  # 更新和谁匹配

            # add alignment between list items
            # df_all.rename({'alignment': 'alignment_list'}, axis=1, inplace=True)
            # df_all.loc[list(df_all[df_all['alignment_list'] == 'v']['id']), 'alignment_item'] = 'h'
            # df_all.loc[list(df_all[df_all['alignment_list'] == 'h']['id']), 'alignment_item'] = 'v'

            # fill nan and change type
            df_all = df_all.fillna(-1)  # 填充 并统一格式

            # df_all[list(df_all.filter(like='group'))] = df_all[list(df_all.filter(like='group'))].astype(int)
            df_all['group_pair'] = df_all['group_pair'].astype(int)
            df_all['pair_to'] = df_all['pair_to'].astype(int)
        else:
            df_all['group_pair'] = -1
            df_all['pair_to'] = -1
        df_all.loc[df_all['pair_to'] == -1, 'group_pair'] = -1
        self.compos_dataframe = df_all

    def split_groups(self, group_name):
        compos = self.compos_dataframe
        groups = []
        g = compos.groupby(group_name).groups
        for i in g:
            if i == -1 or len(g[i]) <= 1:
                continue
            groups.append(compos.loc[g[i]])
        return groups

    '''
    ******************************
    ******* Item Partition *******
    ******************************
    '''
    def list_item_partition(self):
        '''
        identify list item (paired elements) in each compound large group
        track paired compos' "pair_to" attr to assign "list_item" id
        识别列表 并分配 list-id
        '''
        if 'pair_to' not in self.compos_dataframe:  # 检查数据框中是否存在 "pair_to" 属性，如果不存在，则直接返回
            return
        compos = self.compos_dataframe

        # 查看有无可以再细分小组的list group_set:键是组的标识符，值是属于该组的行的索引
        # pair_set, group_set = [], compos.groupby(['group']).groups
        # for g in group_set:
        #     pair_set.append(sorted(group_set[g].tolist() + compos.loc[group_set[g], 'pair_to'].tolist()))  # 每个组和它pairto的合并一个list的嵌套list
        # list_counts = Counter(tuple(sublist) for sublist in pair_set)
        # duplicate_lists = [list(sublist) for sublist, count in list_counts.items() if count > 1]
        # start_idx = max(compos['group_pair'].tolist()) + 1
        # for dl in duplicate_lists:
        #     compos.loc[dl, 'group_pair'] = start_idx
        #     start_idx += 1

        # 检查同一个group_pair里有没有pair到 不符合的直接-1
        group_set = compos.groupby(['group_pair']).groups
        for g in group_set:  # 对每个小小组单独拿出来
            if g != -1:
                g_df = compos.loc[group_set[g]]
                for one_ele in g_df.index:  # 对组内每一个元素
                    if g_df.loc[one_ele, 'pair_to'] not in g_df.index and g_df.loc[one_ele, 'pair_to'] != -1:  # 如果它匹配的不在小组里 且pairto的不是-1
                        compos.loc[one_ele, 'pair_to'] = -1

        groups = compos.groupby(["group_pair"]).groups  # 函数获取数据框 compos_dataframe 中所有具有相同 "group_pair" 值的组（组间有配对关系），将它们分组
        listed_compos = pd.DataFrame()
        for i in groups:
            if i == -1 or len(groups[i]) < 3:  # 小于3的不算list
                continue
            group = groups[i]
            paired_compos = self.compos_dataframe.loc[list(group)]  # 拿到同一个当前组的df
            paired_compos = self.gather_list_items(paired_compos)  # 调用函数 将该组中的列表项识别并标记，并将它们存储在 listed_compos 中。
            listed_compos = listed_compos.append(paired_compos)

        if len(listed_compos) > 2:  # 如果在 listed_compos 中找到了列表项，将它们合并到原始数据框 compos_dataframe 中，并为 "list_item" 列赋予一个唯一的整数 ID。
            self.compos_dataframe = self.compos_dataframe.merge(listed_compos, how='left')
            self.compos_dataframe['list_item'] = self.compos_dataframe['list_item'].fillna(-1).astype(int)
        else:
            self.compos_dataframe['list_item'] = -1

    def gather_list_items(self, compos):  # 在这里 第一次出现list_items
        '''
            gather compos into a list item in the same row/column of a same pair(list) 将一组具有相同 "pair_to" 属性的元素聚合到同一列表项中
            the reason for this is that some list contain more than 2 items, while the 'pair_to' attr only contains relation of two
        '''
        def search_list_item_by_compoid(compo_id):
            """
                list_items: dictionary => {id of first compo: ListItem}
            """
            for i in item_ids:
                if compo_id in item_ids[i]:
                    return i

        compos = compos.sort_values(by=['center_row'])
        list_items = {}  # list_items 用于存储列表项
        item_ids = {}  # item_ids 用于存储与列表项相关的元素ID
        mark = []  # mark 用于标记已处理的元素ID
        for i in range(len(compos)):  # 开始遍历输入的 compos 的每个元素
            compo = compos.iloc[i]
            if compo['pair_to'] == -1:  # 对于每个元素，检查其 "pair_to" 属性：如果 "pair_to" 属性的值为 -1，表示该元素不与任何其他元素配对
                compos.loc[compo['id'], 'list_item'] = self.item_id  # 将其标记为一个新的列表项，为其分配一个唯一的 "list_item" ID
                self.item_id += 1
            # new item 如果 "pair_to" 属性的值不为 -1，表示该元素与另一个元素配对
            elif compo['id'] not in mark and compo['pair_to'] not in mark:  # 当前元素和其配对元素都没有被标记过
                compo_paired = compos.loc[compo['pair_to']]

                list_items[self.item_id] = [compo, compo_paired]  # 把这对元素的df取出来放一个list
                item_ids[self.item_id] = [compo['id'], compo['pair_to']]  # 这对id

                compos.loc[compo['id'], 'list_item'] = self.item_id  # 把当前和pair to都标记一样的list-id
                compos.loc[compo['pair_to'], 'list_item'] = self.item_id
                mark += [compo['id'], compo['pair_to']]  # 标记为已经处理过
                self.item_id += 1

            # 其中一个被标记过 就合并到被标记过的里面
            elif compo['id'] in mark and compo['pair_to'] not in mark:
                index = search_list_item_by_compoid(compo['id'])
                list_items[index].append(compos.loc[compo['pair_to']])
                item_ids[index].append(compo['pair_to'])

                compos.loc[compo['pair_to'], 'list_item'] = index
                mark.append(compo['pair_to'])

            elif compo['id'] not in mark and compo['pair_to'] in mark:
                index = search_list_item_by_compoid(compo['pair_to'])
                list_items[index].append(compos.loc[compo['id']])
                item_ids[index].append(compo['id'])

                compos.loc[compo['id'], 'list_item'] = index
                mark.append(compo['id'])

        compos['list_item'] = compos['list_item'].astype(int)
        compos = compos.sort_index()
        return compos  # list_items

    '''
    ******************************
    *****     Line Split      ****
    ******************************
    '''

    def line_split(self, col_lines, row_lines):
        col_lines, row_lines = col_lines[np.argsort(col_lines[:, 0])], row_lines[np.argsort(row_lines[:,0])]
        df = self.compos_dataframe
        df_sorted = df.sort_values('center_row')  # 默认按行从上到下排序
        df_sorted.insert(df_sorted.shape[1], 'line_split_r', '')  # 新增row列 记录行分割出的结果
        df_sorted.insert(df_sorted.shape[1], 'line_split_c', '')  # 新增col列 记录列分割的结果
        df_sorted.insert(df_sorted.shape[1], 'line_merge_c', -1)  # 新增col列 记录列合并的结果
        if len(row_lines) != 0:
            row_split = np.append(np.append([0], row_lines[:, 0]), [self.img.shape[0]])
            r_n = 0
            for r_i in range(len(row_split) - 1):
                row_s, row_e = row_split[r_i], row_split[r_i + 1]  # 更新本段起止点
                row_selected = (df_sorted['center_row'] >= row_s) & (df_sorted['center_row'] < row_e)
                if len(df_sorted.loc[row_selected]) > 0:
                    df_sorted.loc[row_selected, 'line_split_r'] = str(r_n)  # 给行编号
                    r_n += 1  # 两条线之间有东西 r编号 r_n+1
            df_sorted['line_split_c'] = df_sorted['line_split_r'] + '_0'  # 加上默认值
            if len(col_lines) != 0:  # 如果还有纵向 就有合并、分裂问题
                # 查看每一条竖线涉及的行
                for col_line in col_lines:
                    col_covered = (df_sorted['center_row'] >= col_line[1]) & (df_sorted['center_row'] <= col_line[2])  # 查看本条竖线涉及的行
                    # col_covered = df_sorted['line_split_r'].isin(df_sorted.loc[col_covered, 'line_split_r'].unique())  # 取全一整行所有元素
                    if len(df_sorted.loc[col_covered]) > 0:  # 查看竖线有没有涉及到行
                        rc_group = df_sorted.loc[col_covered].groupby('line_split_c').groups  # 逐个分组来判断 防止重复分组
                        for g in rc_group:  # (这里取的是分组的index)
                            # 情况1：是某一行、几行的侧边竖线 没有跨行
                            if min(df_sorted.loc[rc_group[g], 'center_column']) < col_line[0] < max(df_sorted.loc[rc_group[g], 'center_column']):
                                    # and len(df_sorted.loc[col_covered, 'line_split_r'].unique()) == 1:
                                # 列起始编号 如果之前没编号过 就从1开始 否则从之前最大的开始重新标 (使用正则匹配匹配下划线后面的数字)
                                c_n = 1 if max(df_sorted.loc[rc_group[g], 'line_split_c'].str.extract(r'_(\d+)$').astype(int)) == 0 else \
                                      max(df_sorted.loc[rc_group[g], 'line_split_c'].str.extract(r'_(\d+)$').astype(int))
                                common_indices = rc_group[g].intersection(df_sorted.loc[df_sorted['center_column'] <= col_line[0]].index)  # 同时满足 涉及的行+在左侧
                                df_sorted.loc[common_indices, 'line_split_c'] = df_sorted.loc[common_indices, 'line_split_r'] + '_' + str(c_n)  # 给划分编号
                                common_indices = rc_group[g].intersection(df_sorted.loc[df_sorted['center_column'] >= col_line[0]].index)  # 同时满足 涉及的行+在右侧
                                df_sorted.loc[common_indices, 'line_split_c'] = df_sorted.loc[common_indices, 'line_split_r'] + '_' + str(c_n + 1)  # 给划分编号
                            # 情况2：是多行的侧边竖线 没有分割 跨行
                            elif (col_line[0] <= min(df_sorted.loc[rc_group[g], 'center_column']) or
                                  col_line[0] >= max(df_sorted.loc[rc_group[g], 'center_column'])) and len(df_sorted.loc[col_covered, 'line_split_r'].unique()) > 1:
                                if min(df_sorted.loc[col_covered, 'line_merge_c']) == -1:  # 如果有某一行没有被标记合并
                                    m_n = max(df_sorted['line_merge_c']) + 1 if max(df_sorted.loc[col_covered, 'line_merge_c']) == -1 else \
                                    sorted(df_sorted.loc[col_covered, 'line_merge_c'])[1]  # 合并起始标号 如果都是-1就从0开始 否则从非-1编号开始重新编
                                    df_sorted.loc[col_covered, 'line_merge_c'] = m_n

        # # 下面根据线框解决各种布局冲突（列名：line_split_c）
        # df_sorted_copy = df_sorted.copy()
        # df_sorted = df_sorted_copy.copy()
        # 1 list_item 跨线问题 pair_to、group_pair也要改 根据划线分组 >1给list号
        for li in df_sorted['list_item'].unique():
            if li != -1:
                list_df = df_sorted.loc[df_sorted['list_item'] == li]  # 取这一项 list item
                line_split_c = list_df['line_split_c']  # 看线框分组有哪些
                if len(line_split_c.unique()) > 1:  # 线框分组不在一组
                    st_gp_name, st_li_name = max(df_sorted['group_pair']) + 1, max(df_sorted['list_item']) + 1  # 拆分 从最大开始编号
                    for lc in line_split_c.unique():
                        if len(list_df.loc[list_df['line_split_c'] == lc]) == 1:  # 如果拆完剩一个
                            df_sorted.loc[list_df.loc[list_df['line_split_c'] == lc].index,
                                          ['group_pair', 'pair_to', 'list_item']] = -1, -1, -1  # 清空分组
                        else:  # 如果不止一个
                            df_sorted.loc[list_df.loc[list_df['line_split_c'] == lc].index,
                                          ['group_pair', 'list_item']] = st_gp_name, st_li_name  # 更新编号
                            st_gp_name += 1  # 如果分成多组 还要+1
                            st_li_name += 1
                            for l_idx in list_df.loc[list_df['line_split_c'] == lc].index:
                                if list_df.loc[l_idx, 'pair_to'] not in list_df.loc[list_df['line_split_c'] == lc].index:  # 如果配对的不在了
                                    df_sorted.loc[l_idx, 'pair_to'] = list(set(list_df.loc[list_df['line_split_c'] == lc].index) - {l_idx})[0]  # pair_to 随便选一个组内的

        # 2 group（小组 不是sup_group）的跨线问题 根据横线分割直接拆分没有配对的组
        for g in df_sorted.loc[df_sorted['group_pair']==-1, 'group'].unique():
            if g != -1:
                group_df = df_sorted.loc[(df_sorted['group_pair']==-1) & (df_sorted['group'] == g)]
                if (len(group_df) > 1) and len(group_df['line_split_c'].unique()) > 1:
                    df_sorted.loc[(df_sorted['group_pair'] == -1) & (df_sorted['group'] == g), 'group'] = -1

        # 3 根据group_pair重新标记合并项
        gp_groups = df_sorted.groupby('group_pair').groups
        for gp in gp_groups:
            if gp != -1:
                group_pair_in_same_line = df_sorted.loc[(df_sorted['line_split_c'].isin(df_sorted.loc[gp_groups[gp], 'line_split_c'].unique())), 'line_merge_c']
                if max(group_pair_in_same_line) == -1:  # 如果没有本来的分组信息再根据list合并
                    df_sorted.loc[(df_sorted['line_split_c'].isin(df_sorted.loc[gp_groups[gp], 'line_split_c'].unique())), 'line_merge_c'] = max(df_sorted['line_merge_c']) + 1

        self.compos_dataframe = df_sorted.sort_index()

    '''
    ******************************
    ***** Icon Classification ****
    ******************************
    '''
    def icon_cls_by_clip(self, model_cls, preprocess, device, real_labels_dict, labels_for_read):
        # 一. 根据正则匹配 先还原被ocr误识别的搜索框idx
        pattern = r'^Q(?:\s+|(?![a-z]))'
        # 如果有识别到类似Q开头的
        Q_idx = []
        for t_idx in self.compos_dataframe.index:
            if self.compos_dataframe.loc[t_idx, 'center_row'] < self.img.shape[1]/3 and \
                    re.search(pattern, self.compos_dataframe.loc[t_idx, 'text_content']):  # 判断疑似Q开头的搜索框 (前提搜索框位于图片上1/3)
                Q_idx.append(t_idx)
        # --去掉条件 先分如果分类后没有 搜索放大 则启用分类
        # if len(self.compos_dataframe.loc[(self.compos_dataframe['text_content'].str.contains('搜索')) | (self.compos_dataframe['text_content'].str.lower()== 'search')]) == 0:
        for q_idx in Q_idx:
            # 需要更改：id column_min column_max width area center center_column text_content sub_class
            tx_idx = self.compos_dataframe.index.max() + 1
            self.compos_dataframe.loc[tx_idx] = self.compos_dataframe.loc[q_idx]  # 复制一行
            self.compos_dataframe.loc[tx_idx, 'id'] = tx_idx  # 更新分裂后文字编号
            # 分裂放大镜
            Q_width = self.compos_dataframe.loc[q_idx, 'width']/len(self.compos_dataframe.loc[q_idx, 'text_content'])  # 算出一个字平均宽度
            Q_col_min, Q_col_max, Q_area = self.compos_dataframe.loc[q_idx, 'column_min'], self.compos_dataframe.loc[q_idx, 'column_min'] + Q_width, int(Q_width * self.compos_dataframe.loc[q_idx, 'height'])
            Q_ccol = int((Q_col_min + Q_col_max)/2)
            Q_center, Q_text_content, Q_sub_class = (Q_ccol, int(self.compos_dataframe.loc[q_idx, 'center_row'])), '搜索、放大_', 'buttonicon'
            # 分裂文本
            tx_width = self.compos_dataframe.loc[q_idx, 'width'] - Q_width  # 剩余文字宽度
            tx_col_min, tx_col_max, tx_area = Q_col_max, self.compos_dataframe.loc[q_idx, 'column_max'], int(tx_width * self.compos_dataframe.loc[q_idx, 'height'])
            tx_ccol = int((tx_col_min + tx_col_max)/2)
            tx_center, tx_text_content, tx_sub_class = (tx_ccol, int(self.compos_dataframe.loc[q_idx, 'center_row'])), self.compos_dataframe.loc[q_idx, 'text_content'][1:], 'edittext'
            # 填入数值
            self.compos_dataframe.loc[q_idx, ['column_max', 'width', 'area', 'center_column', 'text_content', 'sub_class']] = \
                int(Q_col_max), int(Q_width), int(Q_area), Q_ccol, Q_text_content, Q_sub_class
            self.compos_dataframe.at[q_idx, 'center'] = Q_center
            self.compos_dataframe.loc[tx_idx, ['column_max', 'width', 'area', 'center_column', 'text_content', 'sub_class']] = \
                int(tx_col_min), int(tx_width), int(tx_area), tx_ccol, tx_text_content, tx_sub_class
            self.compos_dataframe.at[tx_idx, 'center'] = tx_center

        # 二. 前置处理单个奇怪文字
        emoji_idx = (self.compos_dataframe['text_content'].str.len() == 1) & ~(self.compos_dataframe['text_content'].str.isalpha())  # 删掉单个非文字的文本
        self.compos_dataframe.loc[emoji_idx, 'class'] = 'Compo'
        self.compos_dataframe.loc[emoji_idx, 'sub_class'] = 'buttonicon'
        self.compos_dataframe.loc[emoji_idx, 'text_content'] = ''

        # 三. 开始分类
        ori_df = self.compos_dataframe.copy()
        label = clip.tokenize(list(real_labels_dict.values())).to(device)
        # need_cls_df = pd.DataFrame()
        image_pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

        # 已配对图标加注释
        for paired_idx in self.compos_dataframe.loc[(((self.compos_dataframe['text_content'] == '') & ~((ori_df['width']/ori_df['height'] > 1.5) & (ori_df['area']/(self.img.shape[0] * self.img.shape[1]) < 0.2))) |
                                                     (self.compos_dataframe['text_content'] != '') & (self.compos_dataframe['sub_class'] == 'buttonicon') & ~(self.compos_dataframe['text_content'].str.endswith('_')))
                                                    & (self.compos_dataframe['pair_to'] != -1)].index:
            pair_to_text = self.compos_dataframe.loc[self.compos_dataframe.loc[paired_idx, 'pair_to'], 'text_content']
            if pair_to_text == '' or pair_to_text.endswith('_'):
                filtered_text = self.compos_dataframe.loc[(self.compos_dataframe['list_item'] == self.compos_dataframe.loc[paired_idx, 'list_item']) &
                                                          ((self.compos_dataframe['text_content'] != '') & (~self.compos_dataframe['text_content'].str.endswith('_'))), 'text_content'].values
                pair_to_text = ' '.join(filtered_text)
            self.compos_dataframe.loc[paired_idx, 'text_content'] = pair_to_text + '_'
        # 未配对图标找回（仅垂直）or分类 前提：没有配对 条件1：是按钮或者开关 不能有字 条件2：是图不能有字 条件3：单个文字的文本框 条件4：比例特定的 可以不遵守pair原则
        need_cls_df = ori_df.loc[(  ( ((ori_df['sub_class'] == 'buttonicon') | (ori_df['sub_class'] == 'switch')) &
                                      ((ori_df['text_content'] == '') | ~(ori_df['text_content'].str.endswith('_'))))
                                  | ((ori_df['sub_class'] == 'image') & (ori_df['text_content'] != ''))
                                  | ((ori_df['class'].str.lower() == 'text') & (ori_df['text_content'].str.len() == 1) & (ori_df['text_content'] != '我')))
                                 & (ori_df['pair_to']==-1) & (ori_df['text_content'] != 'i')
                                 | (((ori_df['sub_class'] == 'buttonicon') | (ori_df['sub_class'] == 'switch')) &
                                    (ori_df['width']/ori_df['height'] > 1.5) & (ori_df['area']/(self.img.shape[0] * self.img.shape[1]) < 0.2) & (ori_df['text_content']==''))]
        # 开始找回
        for i in range(len(need_cls_df)):
            cp = need_cls_df.iloc[i]
            curr_id = need_cls_df.iloc[i, 0]
            # 先默认两种找回模式：1.横向向左同一排（取消） 2.垂直居中向下 (都是中心阈值)
            curr_row, curr_col = cp['center_row'], cp['center_column']

            # 截图 调试用
            x1, x2, y1, y2 = cp['column_min'], cp['column_max'], cp['row_min'], cp['row_max']
            cp_img = image_pil.crop((int(x1), int(y1), int(x2), int(y2)))  # 截图
            import matplotlib.pyplot as plt
            plt.imshow(cp_img)
            plt.axis('off')  # 关闭坐标轴
            plt.show()

            # row_cand_id = self.compos_dataframe.loc[(curr_row-5 < self.compos_dataframe['center_row']) & (self.compos_dataframe['center_row'] < curr_row+5)
            #                                         & (self.compos_dataframe['text_content'] != '') & ~(self.compos_dataframe['text_content'].str.endswith('_'))].index  # 根据阈值+有字过滤
            # if len(row_cand_id) > 0 and min(self.compos_dataframe.loc[row_cand_id, 'center_column'] - curr_col) < 0:  # 有横向候选，选列最近的 + 必须向左
            #     row_cand_gap = self.compos_dataframe.loc[row_cand_id, 'center_column'] - curr_col
            #     row_cand_gap = row_cand_gap.loc[row_cand_gap < 0]  # 选出向左的控件
            #     self.compos_dataframe.loc[self.compos_dataframe['id'] == curr_id, 'text_content'] = \
            #         self.compos_dataframe.loc[abs(row_cand_gap).idxmin(), 'text_content'] + '_' + predicted_class + '_'  # 不用看阈值 直接匹配
            # else:
            # 横向匹配关

            # 纵向匹配找回
            col_cand_id = self.compos_dataframe.loc[(curr_col - 15 < self.compos_dataframe['center_column']) & (self.compos_dataframe['center_column'] < curr_col + 4)
                                                    & (self.compos_dataframe['center_row'] >= cp['center_row']) & (self.compos_dataframe['text_content'] != '')
                                                    & ~(self.compos_dataframe['text_content'].str.endswith('_')) & (self.compos_dataframe['id'] != curr_id)
                                                    & (self.compos_dataframe['line_split_c'] == cp['line_split_c'])].index  # 根据阈值+有字过滤 因为可能有脚标 左边便宜阈值加大
            find_the_text = False
            if len(col_cand_id) > 0 and max(self.compos_dataframe.loc[col_cand_id, 'center_row'] - curr_row) > 0 and 0.25 < cp['width']/cp['height'] < 4:  # 有竖向候选，选行最近的 + 必须向下 还要加上阈值 不能太远 必需是图标
                col_cand_gap = self.compos_dataframe.loc[col_cand_id, 'center_row'] - curr_row
                # col_cand_gap = col_cand_gap.loc[col_cand_gap > 0]  # 选出向下的控件
                if min(col_cand_gap) < 40:  # 有向下的 且间距小于
                    selected_idx = abs(self.compos_dataframe.loc[col_cand_id, 'center_row'] - curr_row).idxmin()
                    self.compos_dataframe.loc[self.compos_dataframe['id'] == curr_id, 'text_content'] = self.compos_dataframe.loc[selected_idx, 'text_content'] + '_'  # 成功匹配
                    self.compos_dataframe.loc[self.compos_dataframe['id'] == curr_id, 'pair_to'] = selected_idx  # 上配对户口
                    self.compos_dataframe.loc[self.compos_dataframe['id'] == selected_idx, 'pair_to'] = curr_id
                    self.compos_dataframe.loc[(self.compos_dataframe['id'] == curr_id) | (self.compos_dataframe['id'] == selected_idx), 'alignment_in_group'] = 'h'  # 修改为默认水平分布
                    self.compos_dataframe.loc[(self.compos_dataframe['id'] == curr_id) | (self.compos_dataframe['id'] == selected_idx), 'list_item'] = max(self.compos_dataframe['list_item']) + 1
                    # 如果同一框线内（line_split_c）人有现成的group_pair就加入 否则新建
                    line_group = self.compos_dataframe.loc[(self.compos_dataframe['id'] == curr_id) | (self.compos_dataframe['id'] == selected_idx), 'line_split_c']
                    st_gp_name = max(self.compos_dataframe.loc[self.compos_dataframe['line_split_c'] == line_group.unique()[0], 'group_pair'])
                    find_the_text = True
                    if len(line_group.unique())==1 and st_gp_name != -1 and abs(max(self.compos_dataframe.loc[self.compos_dataframe['group_pair'] == st_gp_name, 'center_row']) - cp['center_row']) < 100:
                        self.compos_dataframe.loc[(self.compos_dataframe['id'] == curr_id) | (self.compos_dataframe['id'] == selected_idx), 'group_pair'] = st_gp_name
                    else:
                        self.compos_dataframe.loc[(self.compos_dataframe['id'] == curr_id) | (self.compos_dataframe['id'] == selected_idx), 'group_pair'] = max(self.compos_dataframe['group_pair']) + 1
                # elif cp['class'].lower() == 'text':
                #     # 如果文本没有找回配对 那就不动了
                #     continue
            if not find_the_text:  # 如果无法找回 启用图标识别
                x1, x2, y1, y2 = cp['column_min'], cp['column_max'], cp['row_min'], cp['row_max']
                w, h = abs(x2 - x1), abs(y2 - y1)
                cp_img = image_pil.crop((int(x1), int(y1), int(x2), int(y2)))  # 截图
                image = preprocess(cp_img).unsqueeze(0).to(device)
                if cp['area']/(self.img.shape[0] * self.img.shape[1]) < 0.2:  # 面积不能太大
                    with torch.no_grad():  # clip分类预测
                        logits_per_image, logits_per_text = model_cls(image, label)  # 预测结束
                        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]  # 各类概率
                        predicted_class = labels_for_read[str(np.argmax(probs) + 1)]
                        if np.argmax(probs) + 1 == 148 and w/h >= 4:  # 错误补丁
                            predicted_class = '搜索框'
                        if np.argmax(probs) + 1 in [173, 174] and cp['area']/(self.img.shape[0] * self.img.shape[1]) < 0.0005:  # 错误补丁
                            predicted_class = '页面指示器'

                        # 预测类别的文字
                        if self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'].values[0] == '':
                            if not 0.05 <= w/h <= 20:  # 比例奇怪 是图片
                                self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] = '图片_'
                            # print('id' + str(need_cls_df.iloc[i, 0]), 'area', area_wh)
                            elif (sorted(probs)[-2] + sorted(probs)[-1]) <= 0.35:  # 1 置信度小的
                                if cp['area'] / (self.img.shape[0] * self.img.shape[1]) < 0.0015:  # 1.1 图片面积小/比例不奇怪 可能是图标
                                    self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] = '可能是' + predicted_class + '_'  # + str(max(probs))+ '_'  # 找回注释失败 使用原分类结果
                                    if sorted(probs)[-2] + sorted(probs)[-1] < 0.3:  # 如果置信度太低
                                        self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] = '图片_'
                                # os_dist_df = np.sqrt((self.compos_dataframe['center_column'] - self.compos_dataframe.loc[curr_id, 'center_column'])**2 + (self.compos_dataframe['center_row'] - self.compos_dataframe.loc[curr_id, 'center_row'])**2)
                                # xy_dist_df = np.minimum(abs(self.compos_dataframe['center_column']-self.compos_dataframe.loc[curr_id, 'center_column']), abs(self.compos_dataframe['center_row']-self.compos_dataframe.loc[curr_id, 'center_row']))
                                #
                                # combined_df = pd.concat([os_dist_df, xy_dist_df], axis=1)
                                # combined_df.columns = ['os_dist_df', 'xy_dist_df']  # 为列添加新的名称
                                # # 根据 'xy_dist_df' 和 'dist_df' 进行排序
                                # dist_df = combined_df.sort_values(by=['xy_dist_df', 'os_dist_df'], ascending=[True, True])
                                # find_exp = self.compos_dataframe.loc[dist_df[(self.compos_dataframe['text_content'] != '') & ~(self.compos_dataframe['text_content'].str.endswith('_')) & (xy_dist_df < 30) & (os_dist_df < 300)].index, 'text_content']
                                # # print(find_exp)
                                #
                                # if len(find_exp) >= 1:  # 1.1.1 如果能找到 就用释义名
                                #     self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] = find_exp.iloc[0] + '_'  # 选择距离最近的文字进行赋值
                                else:  #  1.2 图片面积大 可能是头像/图片
                                    self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] = '图片，或' + predicted_class + '_'  # 填入df
                                    if 0.975 <= w/h <= 1.025:  # 按照比例 判断头像
                                        # 头像需要满足面积+比例之外 还需要出现在左上（1/3）/右上（1/3）以及中间（阈值+-5）
                                        img_w, img_h = self.img.shape[1], self.img.shape[0]
                                        if cp['center_column'] < img_w/3 and cp['center_row'] < img_h/3 or cp['center_column'] > img_w*2/3 and cp['center_row'] < img_h*1/3 or img_w/2 - 5 < cp['center_column'] < img_w/2 + 5 and cp['center_row'] < img_h*1/3:
                                            self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] = '头像，或' + predicted_class + '_'  # + str(max(probs)) + '_' + str(area_wh) + '_'  + str(w/h) + '_'
                            else:  # 2 置信度大 是见过的图标+ 5
                                if predicted_class == '搜索框':
                                    if cp['area']/(self.img.shape[0] * self.img.shape[1]) < 0.01:  # 搜索框面积太小的也不对
                                        self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] = '图片' +'_'
                                    else:
                                        self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] = predicted_class + '_'
                                        self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'sub_class'] = "edittext"
                                else:
                                    self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] = predicted_class + '键' + '_'  # + str(max(probs)) + '_' # 填入df
                        else:
                            self.compos_dataframe.loc[self.compos_dataframe['id'] == need_cls_df.iloc[i, 0], 'text_content'] += '_'  # '-' + predicted_class + '_'  # 填入df

        # 四. 对单独一个Q的文字改回搜索hhh
        self.compos_dataframe.loc[self.compos_dataframe['text_content'] == 'Q', 'class'] = 'Compo'
        self.compos_dataframe.loc[self.compos_dataframe['text_content'] == 'Q', 'sub_class'] = 'buttonicon'
        self.compos_dataframe.loc[self.compos_dataframe['text_content'] == 'Q', 'text_content'] = '图标：搜索、放大_'

        # 五. 搜索框找回
        # 1. 根据搜索
        search_found = False
        search_icon = self.compos_dataframe.loc[self.compos_dataframe['text_content'].str.contains('搜索、放大')]
        for search_idx in range(len(search_icon)):
            curr_row, curr_col = search_icon.iloc[search_idx]['center_row'], search_icon.iloc[search_idx]['center_column']  # 搜索框中心位置
            # 选取同行 且长宽比>4的
            row_cand_id = self.compos_dataframe.loc[(((curr_row-5 < self.compos_dataframe['center_row']) & (self.compos_dataframe['center_row'] < curr_row+5)) |
                                                    ((search_icon.iloc[search_idx]['row_min'] < self.compos_dataframe['row_min']) & (self.compos_dataframe['row_max'] < search_icon.iloc[search_idx]['row_max']))) &
                                                    (self.compos_dataframe['width']/self.compos_dataframe['height'] > 4)].index
            if len(row_cand_id) > 0:  # 有横向候选，选列最近的 + 根据搜索框定左右
                row_cand_gap = self.compos_dataframe.loc[row_cand_id, 'center_column'] - curr_col
                row_cand_gap = row_cand_gap.loc[row_cand_gap < 0] if curr_col > self.img.shape[1]/2 else row_cand_gap.loc[row_cand_gap > 0]  # 根据搜索图标位置确定匹配方向
                if len(row_cand_gap) > 0:
                    self.compos_dataframe.loc[row_cand_gap.loc[row_cand_gap == sorted(abs(row_cand_gap))[0]].index, 'sub_class'] = 'edittext'  # 选择最近的一个长条改为文本框
                    search_found = True
        # 2. 根据图片搜索
        if not search_found:
            search_icon = self.compos_dataframe.loc[(self.compos_dataframe['text_content'].str.contains('相机、图片搜索、或截图、快门')) | (self.compos_dataframe['text_content'].str.contains('（前后置）摄像头切换'))]
            for search_idx in range(len(search_icon)):
                curr_row, curr_col = search_icon.iloc[search_idx]['center_row'], search_icon.iloc[search_idx]['center_column']  # 搜索框中心位置
                # 选取同行 且长宽比>4的
                row_cand_id = self.compos_dataframe.loc[(((curr_row - 5 < self.compos_dataframe['center_row']) & (self.compos_dataframe['center_row'] < curr_row + 5)) |
                                                         ((search_icon.iloc[search_idx]['row_min'] < self.compos_dataframe['row_min']) & (self.compos_dataframe['row_max'] < search_icon.iloc[search_idx]['row_max']))) &
                                                        (self.compos_dataframe['width'] / self.compos_dataframe['height'] > 4)].index
                if len(row_cand_id) > 0:  # 有横向候选，选列最近的 + 根据搜索框定左右
                    row_cand_gap = self.compos_dataframe.loc[row_cand_id, 'center_column'] - curr_col
                    row_cand_gap = row_cand_gap.loc[row_cand_gap < 0] if curr_col > self.img.shape[1] / 2 else row_cand_gap.loc[row_cand_gap > 0]  # 根据搜索图标位置确定匹配方向
                    if len(row_cand_gap) > 0:
                        self.compos_dataframe.loc[row_cand_gap.loc[abs(row_cand_gap) == sorted(abs(row_cand_gap))[0]].index, 'sub_class'] = 'edittext'  # 选择最近的一个长条改为文本框
                        search_found = True
        # 3. 根据"搜索"
        if not search_found:
            search_icon = self.compos_dataframe.loc[(self.compos_dataframe['text_content'] == '搜索') | (self.compos_dataframe['text_content'].str.lower() == 'search')]
            for search_idx in range(len(search_icon)):
                curr_row, curr_col = search_icon.iloc[search_idx]['center_row'], search_icon.iloc[search_idx]['center_column']  # 搜索框中心位置
                # 选取同行 且长宽比>4的
                row_cand_id = self.compos_dataframe.loc[(((curr_row - 5 < self.compos_dataframe['center_row']) & (self.compos_dataframe['center_row'] < curr_row + 5)) |
                                                         ((search_icon.iloc[search_idx]['row_min'] < self.compos_dataframe['row_min']) & (self.compos_dataframe['row_max'] < search_icon.iloc[search_idx]['row_max']))) &
                                                        (self.compos_dataframe['width'] / self.compos_dataframe['height'] > 4)].index
                if len(row_cand_id) > 0:  # 有横向候选，选列最近的 + 根据搜索框定左右
                    row_cand_gap = self.compos_dataframe.loc[row_cand_id, 'center_column'] - curr_col
                    row_cand_gap = row_cand_gap.loc[row_cand_gap < 0] if curr_col > self.img.shape[1] / 2 else row_cand_gap.loc[row_cand_gap > 0]  # 根据搜索图标位置确定匹配方向
                    if len(row_cand_gap) > 0:
                        self.compos_dataframe.loc[row_cand_gap.loc[abs(row_cand_gap) == sorted(abs(row_cand_gap))[0]].index, 'sub_class'] = 'edittext'  # 选择最近的一个长条改为文本框

        # 六. 适用于地图的加强edit_text找回
        self.compos_dataframe.loc[(self.compos_dataframe['text_content'] == '我的位置') | (self.compos_dataframe['text_content'].str.contains('输入')), 'sub_class'] = 'edittext'

        # empty_edittext_idx = (self.compos_dataframe['sub_class'] == 'edittext') & (self.compos_dataframe['text_content'] == '')  # 空输入框加字
        self.compos_dataframe.loc[(self.compos_dataframe['sub_class'] == 'edittext') | (self.compos_dataframe['sub_class'] == 'autocompletetextview'), 'text_content'] = \
            '默认文本: ' + self.compos_dataframe.loc[(self.compos_dataframe['sub_class'] == 'edittext') | (self.compos_dataframe['sub_class'] == 'autocompletetextview'), 'text_content']  # '在此处输入'  # ['在此输入'+str(edt_id) for edt_id in list(range(len(self.compos_dataframe.loc[empty_edittext_idx])))]


    '''
    *****************************
    ******* Visualization *******
    *****************************
    '''
    def visualize(self, img=None, gather_attr='class', name='board', show=True):
        if img is None:
            img = self.img.copy()
        return draw.visualize(img, self.compos_dataframe, attr=gather_attr, name=name, show=show)

    def visualize_fill(self, img=None, gather_attr='class', name='board', show=True):
        if img is None:
            img = self.img.copy()
        return draw.visualize_fill(img, self.compos_dataframe, attr=gather_attr, name=name, show=show)

    def visualize_cluster(self, show=True):
        board = draw.visualize_group_transparent(self.img.copy(), self.compos_dataframe, 'cluster_center_column', show=show)
        board = draw.visualize_group_transparent(board, self.compos_dataframe, 'cluster_center_row', 0.5, 1, color=(0, 0, 255), show=show)
        return board
