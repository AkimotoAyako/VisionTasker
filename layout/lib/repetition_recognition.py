import math
import numpy as np


def recog_repetition_nontext(compos, show=True, only_non_contained_compo=False):

    '''
    produced dataframe attributes: 'alignment', 'group_nontext'
    识别非文本元素的重复布局并返回相关信息 dbscan聚类
    '''

    compos_cp = compos.copy()
    compos_cp.select_by_class(['Compo', 'Background'], no_parent=only_non_contained_compo, replace=True)  # 选compos和背景 且只选择未包含在其他元素内的元素
    # if no text compo, return empty dataframe
    # 如果没有符合条件的元素，函数会返回一个空的数据框，其中包含两列 'alignment' 和 'group_nontext'，它们都初始化为 -1
    if len(compos_cp.compos_dataframe) == 0:
        compos_cp.compos_dataframe['alignment'] = -1
        compos_cp.compos_dataframe['group_nontext'] = -1
        return compos_cp.compos_dataframe

    # step1. cluster compos
    compos_cp.cluster_dbscan_by_attr('center_column', eps=3.5, show=show)  # 使用 DBSCAN 算法对元素进行聚类
    compos_cp.cluster_dbscan_by_attr('center_row', eps=3.5, show=show)  # 使用 'center_column' 和 'center_row' 属性进行垂直和水平方向的聚类
    compos_cp.cluster_dbscan_by_area('area', compos_cp.img.shape[0] * compos_cp.img.shape[1], eps=0.01, show=show)
    # compos_cp.cluster_area_by_relational_size(show=show)  # 并使用 'area' 属性进行面积的聚类

    # step2. group compos according to clustering
    compos_cp.group_by_clusters(cluster=['cluster_area', 'cluster_center_column'], alignment='v', show=show)  # 垂直方向分组
    # compos_cp.group_by_clusters(cluster=['cluster_area', 'cluster_center_row'], alignment='h', show=show)  # 垂直方向分组
    compos_cp.check_group_of_two_compos_validity_by_areas(show=show)  # 根据2.2倍面积以内原则进行分组检查
    compos_cp.group_by_clusters_conflict(cluster=['cluster_area', 'cluster_center_row'], alignment='h', show=show)  # 水平方向分组(冲突解决) 有问题
    # compos_cp.group_by_clusters_conflict(cluster=['cluster_area', 'cluster_center_column'], alignment='v', show=show)  # 水平方向分组(冲突解决) 有问题
    compos_cp.check_group_of_two_compos_validity_by_areas(show=show)  # 根据2.2倍面积以内原则进行分组检查 b
    compos_cp.compos_dataframe.rename({'group': 'group_nontext'}, axis=1, inplace=True)

    return compos_cp.compos_dataframe


def recog_repetition_text(compos, show=True, only_non_contained_compo=False):
    '''
    produced dataframe attributes: 'alignment', 'group_text'
    文本元素的dbscan聚类
    '''
    compos_cp = compos.copy()
    compos_cp.select_by_class(['Text'], no_parent=only_non_contained_compo, replace=True)
    # if no text compo, return empty dataframe
    if len(compos_cp.compos_dataframe) == 0:
        compos_cp.compos_dataframe['alignment'] = -1
        compos_cp.compos_dataframe['group_text'] = -1
        return compos_cp.compos_dataframe

    # step1. cluster compos
    compos_cp.cluster_dbscan_by_attr('row_min', 3, show=False)
    compos_cp.check_group_by_attr(target_attr='cluster_row_min', check_by='height', eps=15, show=show)
    compos_cp.cluster_dbscan_by_attr('column_min', 2, show=False)
    compos_cp.check_group_by_attr(target_attr='cluster_column_min', check_by='height', eps=30, show=show)
    # compos_cp.cluster_dbscan_by_attr('column_max', 5, show=False)
    # compos_cp.check_group_by_attr(target_attr='cluster_column_max', check_by='height', eps=30, show=show)
    
    # step2. group compos according to clustering
    compos_cp.group_by_clusters('cluster_row_min', alignment='h', show=show)  # 上对齐
    # compos_cp.check_group_of_two_compos_validity_by_areas(show=show)  # 根据2.2倍面积以内原则进行分组检查
    compos_cp.group_by_clusters_conflict('cluster_column_min', alignment='v', show=show)  # 左对齐
    # compos_cp.group_by_clusters_conflict('cluster_column_max', alignment='v', show=show)  # 右对齐
    # compos_cp.check_group_of_two_compos_validity_by_areas(show=show)  # 根据2.2倍面积以内原则进行分组检查
    compos_cp.regroup_left_compos_by_cluster('cluster_column_min', alignment='v', show=show)  # 对单元素集再次分组(保险）
    compos_cp.compos_dataframe.rename({'group': 'group_text'}, axis=1, inplace=True)

    return compos_cp.compos_dataframe


def calc_connections(compos):
    '''
    connection of two compos: (length, id_1, id_2) of the connecting line between two compos' centers
    return: connections between all compos
    用于计算一组元素（compos）之间的连接关系，其中连接关系由以下三元组表示：(length, id_1, id_2)，表示两个元素之间的连接线的长度以及这两个元素的唯一标识符（id）
    '''
    connections = []  # 创建一个空列表 connections 用于存储连接信息
    for i in range(len(compos) - 1):
        c1 = compos.iloc[i]
        for j in range(i + 1, len(compos)):  # 取出一对元素的信息
            c2 = compos.iloc[j]
            distance = int(math.sqrt((c1['center_column'] - c2['center_column'])**2 + (c1['center_row'] - c2['center_row'])**2))  # 计算欧式距离
            # slope = round((c1['center_row'] - c2['center_row']) / (c1['center_column'] - c2['center_column']), 2)
            connections.append((distance, c1['id'], c2['id']))  # 计算距离相似性
    # connections = sorted(connections, key=lambda x: x[0])
    return connections


def match_two_connections(cons1, cons2):
    '''
    input: two lists of connections [(length, id_1, id_2)]
        for a block having n elements, it has n*(n-1)/2 connections (full connection of all nodes)
    '''
    if abs(len(cons1) - len(cons2)) > 1:  # 比较两个连接列表的长度，如果它们的长度差距大于1，就返回 False。
        return False
    marked = np.full(len(cons2), False)  # 创建一个布尔数组 marked，其长度与 cons2 中连接的数量相同，用于标记 cons2 中的连接是否已经匹配。
    matched_num = 0  # 计算匹配的连接数量
    for c1 in cons1:  # 连接匹配
        for k, c2 in enumerate(cons2):
            # the two connections are matched
            if not marked[k] and max(c1[0], c2[0]) < min(c1[0], c2[0]) * 1.5:  # 匹配的条件是：两个连接的长度差异不大（max(c1[0], c2[0]) < min(c1[0], c2[0]) * 1.5），且 cons2 中的连接没有被标记为已匹配。
                marked[k] = True
                matched_num += 1
                break
    if matched_num == min(len(cons1), len(cons2)):
        return True
    return False


def recog_repetition_block_by_children_connections(children_list, connections_list, start_pair_id):
    # 比较子块之间的连接信息，识别出相似的块（容器）
    '''
    接收参数：
    children_list：一个包含多个数据框的列表，每个数据框表示一个块（容器）中的子元素。
    connections_list：一个包含多个连接信息的列表，每个连接信息是一个子块与其它子块之间的连接描述。
    start_pair_id：一个整数，表示开始分配的块配对的起始标识。
    eg: paired_blocks = rep.recog_repetition_block_by_children_connections(children_list, connections_list, start_pair_id)
    '''
    pairs = {}  # {'pair_id': [dataframe of children]}  存储块的配对关系，键是块的配对标识，值是具有相同配对标识的子块数据框列表。
    pair_id = start_pair_id  # 初始化 pair_id 为 start_pair_id，表示块配对的标识从该值开始递增。
    mark = np.full(len(children_list), False)  # 标记子块是否已经配对

    for i in range(len(children_list) - 1):  # 两层循环遍历所有子块，比较它们的连接信息
        connections1 = connections_list[i]  # 取出子块之间的连接信息
        children1 = children_list[i]  # 取出子元素信息
        for j in range(i + 1, len(children_list)):
            connections2 = connections_list[j]
            children2 = children_list[j]
            if match_two_connections(connections1, connections2):  # 连接信息是否匹配
                if not mark[i]:
                    # hasn't paired yet, creat a new pair
                    if not mark[j]:  # 如果i和j都没有匹配成功过 更新他们的pair_id 更新匹配状态
                        pair_id += 1
                        children1['group_pair'] = pair_id
                        children2['group_pair'] = pair_id
                        pairs[pair_id] = [children1, children2]
                        mark[i] = True
                        mark[j] = True
                    # if c2 is already paired, set c1's pair_id as c2's
                    else:  # 只给其中一个更新匹配信息 按照已经匹配好的另一个的组抄
                        children1['group_pair'] = children2.iloc[0]['group_pair']
                        pairs[children2.iloc[0]['group_pair']].append(children1)
                        mark[i] = True
                else:  # 只给其中一个更新匹配信息 按照已经匹配好的另一个的组抄
                    # if c1 is marked while c2 isn't marked
                    if not mark[j]:
                        children2['group_pair'] = children1.iloc[0]['group_pair']
                        pairs[children1.iloc[0]['group_pair']].append(children2)
                        mark[j] = True
                    # if c1 and c2 are all already marked in different group_pair, merge the two group_pairs together
                    else:  # 如果 children1 和 children2 分别属于不同的块配对，将这两个块配对合并为一个，即将所有的子块添加到一个新的块配对中，同时更新子块的配对标识
                        # merge all g2's pairing groups with g1's
                        if children1.iloc[0]['group_pair'] != children2.iloc[0]['group_pair']:
                            c1_pair_id = children1.iloc[0]['group_pair']
                            c2_pair_id = children2.iloc[0]['group_pair']
                            for c in pairs[c2_pair_id]:
                                c['group_pair'] = c1_pair_id
                                pairs[c1_pair_id].append(c)
                            pairs.pop(c2_pair_id)

    merged_pairs = None  # 将所有配对的子块合并为一个数据框 merged_pairs
    for i in pairs:
        for children in pairs[i]:
            if merged_pairs is None:
                merged_pairs = children
            else:
                merged_pairs = merged_pairs.append(children, sort=False)
    return merged_pairs

