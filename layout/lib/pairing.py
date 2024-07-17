import pandas as pd
import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN
import layout.lib.draw as draw


def calc_compos_distance(compo1, compo2):
    # compo1 is on the left of compo2
    if compo1['column_max'] <= compo2['column_min']:
        dist_h = compo2['column_min'] - compo1['column_max']
    # compo1 is on the right of compo2
    elif compo2['column_max'] <= compo1['column_min']:
        dist_h = compo1['column_min'] - compo2['column_max']
    # compo1 and compo2 align vertically
    else:
        dist_h = -1

    # compo1 is on the top of compo2
    if compo1['row_max'] <= compo2['row_min']:
        dist_v = compo2['row_min'] - compo1['row_max']
    # compo1 is below compo2
    elif compo2['row_max'] <= compo1['row_min']:
        dist_v = compo1['row_min'] - compo2['row_max']
    # compo1 and compo2 align horizontally
    else:
        dist_v = -1

    if dist_h == -1:
        # compo1 and compo2 are intersected, which is impossible as UIED has merged all intersected compoments
        if dist_v == -1:
            print('Impossible due to all intersected compos were merged')
            return False
        else:
            return dist_v
    else:
        if dist_v == -1:
            return dist_h
        # compo1 and compo2 dont align neither vertically nor horizontally
        else:
            dist = math.sqrt(dist_v ** 2 + dist_h ** 2)
            return dist


def calc_compos_y_distance(compo1, compo2):
    # if y-intersected, set the t-distance as -1 x或y相交就算
    if (max(compo1['row_min'], compo2['row_min']) < min(compo1['row_max'], compo2['row_max']) and
            (abs(compo1['center_row'] - compo2['center_row']) <= 5)) \
            or \
            (max(compo1['column_min'], compo2['column_min']) < min(compo1['column_max'], compo2['column_max']) and
             (abs(compo1['row_min'] - compo2['row_max']) <= 15 or abs(compo1['row_max'] - compo2['row_min']) <= 15)):
        return -1
    # # if not, calculate the y-distance
    # if compo1['row_min'] < compo2['row_min']:
    #     # c1 is above c2
    #     return compo2['row_min'] - compo1['row_max']
    # else:
    #     # c1 is under c2
    #     return compo1['row_min'] - compo2['row_max']
    # if not, calculate the y-distance
    if compo1['row_min'] < compo2['row_min']:
        # c1 is above c2
        return compo2['center_row'] - compo1['center_row']
    else:
        # c1 is under c2
        return compo1['center_row'] - compo2['center_row']

def match_angles(angles_all, max_matched_angle_diff=10):
    '''
    @angles_all : list of list, each element in the g1's angle with each element in the g2
                [[0.27, -12.88, 24.40],  # the 1st element in the g1's angles with all elements in the g2
                [13.00, 0.18, 13.06]]    # the 2ed element in the g1's angles with all elements in the g2
    Match if there are any similar angles for the 1st and 2ed element (e.g. 0.25 abd 0.18 in the example)
    '''
    for i in range(min(3, len(angles_all) - 1)):
        angles_i = angles_all[i]
        for k, an_i in enumerate(angles_i):
            if an_i is None: continue
            # match with others
            matched_num = 1
            paired_ids = [k]
            for j in range(i + 1, len(angles_all)):
                angles_j = angles_all[j]
                for p, an_j in enumerate(angles_j):
                    if an_j is not None and abs(an_i - an_j) < max_matched_angle_diff:
                        paired_ids.append(p)
                        matched_num += 1
                        break
            if matched_num == len(angles_all):
                return paired_ids
    return None


def calc_angle(c1, c2, anchor='corner'):
    '''
    @anchor: 'corner' -> calculated by top left, or 'center' -> calculate by center
    '''
    angle = None
    if anchor == 'corner':
        angle = int(math.degrees(math.atan2(c1['row_min'] - c2['row_min'], c1['column_min'] - c2['column_min'])))
    elif anchor == 'center':
        angle = int(math.degrees(math.atan2(c1['center_row'] - c2['center_row'], c1['center_column'] - c2['center_column'])))
    if angle < 0:
        angle += 180
    if angle > 90:
        angle -= 180
    return angle

# 在这里 修改 组和组匹配时 pair-main
def match_two_groups_by_angles_and_y_distance(g1, g2, diff_distance=1.2, diff_angle=5, match_thresh=0.8):
    '''
    As the text's length is variable, we don't count on distance if one or more groups are texts.
    In this situation, we count on the angles of the line between two possibly paired elements.
    '''
    assert g1.iloc[0]['alignment_in_group'] == g2.iloc[0]['alignment_in_group']  # 函数检查两个组的alignment_in_group是否相同如果不相同，直接返回 False。
    alignment = g1.iloc[0]['alignment_in_group']
    pairs = {}
    if alignment == 'h':  # 对两个组内的元素按照水平或垂直坐标进行排序，分别得到 g1_sort 和 g2_sort
        g1_sort = g1.sort_values('center_column')
        g2_sort = g2.sort_values('center_column')
    else:
        g1_sort = g1.sort_values('center_row')
        g2_sort = g2.sort_values('center_row')

    # max_height = (max(list(g1['height'])) + max(list(g2['height'])))/2  # 函数计算两个组中元素的最大高度，用于后续的距离计算
    max_height = 15
    swapped = False
    if len(g1) == len(g2):  # 如果两个组内的元素数量相同，函数将进行如下匹配操作 else在183行哦
        angles_cor = []  # 首先，计算每对元素之间的距离、角度等信息，然后进行匹配。
        angles_cen = []
        distances = []
        for i in range(len(g1_sort)):
            c1 = g1_sort.iloc[i]
            c2 = g2_sort.iloc[i]
            angles_cor.append(calc_angle(c1, c2, 'corner'))
            angles_cen.append(calc_angle(c1, c2, 'center'))
            distances.append(calc_compos_y_distance(c1, c2))
        # print(distances, 'Corner Angles:', angles_cor, 'Center Angles:', angles_cen)

        # match distances 匹配距离：函数计算两组元素中每对元素之间的距离，如果大多数元素的距离相似 差距小于10，且没有元素的距离过远1.2最小距离，就认为距离匹配成功。
        matched_number = 0
        for i, distance in enumerate(distances):
            dis_i = distances[i]
            # if the y-distance is too far, regard it as unmatched
            if distance > max_height or (abs(g1_sort.iloc[i]['center_column']-g2_sort.iloc[i]['center_column']) > max_height and alignment == 'h'):  # 距离控制
                continue
            for j in range(len(distances)):
                if i == j:
                    continue
                dis_j = distances[j]
                if abs(dis_i - dis_j) < 10 or max(dis_i, dis_j) < diff_distance * min(dis_i, dis_j):
                    matched_number += 1
                    break
        # print('distance match:', matched_number)
        if matched_number < len(distances) * match_thresh:
            return False

        # match angles both by corner and by center 匹配角度（tan）：函数计算每对元素之间的角度（分别为角点处的角度和中心点处的角度），如果大多数元素的角度相似，就认为角度匹配成功。
        match_num = 0
        for i in range(len(angles_cor)):  # 角点角度
            angle_i = angles_cor[i]
            for j in range(len(angles_cor)):
                if i == j:
                    continue
                angle_j = angles_cor[j]
                # compare the pair's distance and angle between the line and the x-axis
                if abs(angle_i - angle_j) < diff_angle:  # 如果很多角度都不匹配就算了
                    match_num += 1
                    break
        # print('corner angle match:', match_num)

        # if fail to match corner angle, try to match center angle
        if matched_number < len(angles_cor) * match_thresh:
            match_num = 0
            for i in range(len(angles_cen)):  # 中心点角度
                angle_i = angles_cen[i]
                for j in range(len(angles_cen)):
                    if i == j:
                        continue
                    angle_j = angles_cen[j]
                    # compare the pair's distance and angle between the line and the x-axis
                    if abs(angle_i - angle_j) < diff_angle:  # 如果很多角度都不匹配就算了
                        match_num += 1
                        break
            # print('center angle match:', match_num)
            if matched_number < len(angles_cen) * match_thresh:
                return False

        # record pairs if matched 如果前面的一对False之后闯关成功 就说明匹配上了 恭喜!
        for i in range(len(g1_sort)):
            pairs[g1_sort.iloc[i]['id']] = g2_sort.iloc[i]['id']

    else:  # 如果两个组内的元素数量不同，函数将进行一种特殊情况的匹配： 263行结束
        if max(len(g1_sort), len(g2_sort)) > min(len(g1_sort), len(g2_sort)) * 8:  # 首先两个组元素相差不能超过5倍
            return False
        # make sure g1 represents the shorter group while g2 is the longer one
        if len(g1_sort) > len(g2_sort):  # 首先，确保 g1 表示元素数量较少的组，如果不是，则交换 g1 和 g2
            temp = g1_sort
            g1_sort = g2_sort
            g2_sort = temp
            swapped = True

        distances = []  # 又是距离和角度的匹配
        angles_cor = []
        angles_cen = []
        # calculate the y-distances between each c1 in g1 and all c2 in g2 计算 g1 中的每个元素与 g2 中所有元素之间的距离和角度

        for i in range(len(g1_sort)):
            c1 = g1_sort.iloc[i]
            distance = None
            angle_cor = None
            angle_cen = None
            for j in range(len(g2_sort)):
                c2 = g2_sort.iloc[j]
                d_cur = calc_compos_y_distance(c1, c2)
                # match the closest
                if distance is None or distance > d_cur:
                    distance = d_cur
                    angle_cor = calc_angle(c1, c2, 'corner')
                    angle_cen = calc_angle(c1, c2, 'center')
                    pairs[c1['id']] = c2['id']
            distances.append(distance)
            angles_cor.append(angle_cor)
            angles_cen.append(angle_cen)
        # print(distances, 'Corner Angles:', angles_cor, 'Center Angles:', angles_cen)

        # match the distances # 函数计算 g1 中的元素与 g2 中的哪个元素距离最近，如果大多数元素的距离相似，且没有元素的距离过远，就认为距离匹配成功。
        match_num = 0
        g1m_set, g2m_set = [], []  # 保存匹配上的小小组是那几个
        for i in range(len(distances)):
            dis_i = distances[i]
            for j in range(len(distances)):
                if i == j:
                    continue
                dis_j = distances[j]
                # compare the pair's distance and angle between the line and the x-axis
                if -1 * max_height < max(dis_i, dis_j) < max_height and \
                        (abs(dis_i - dis_j) <= 10 or max(dis_i, dis_j) < diff_distance * min(dis_i, dis_j)):
                    match_num += 1
                    g1m_set += [g1_sort.loc[g1_sort['id'] == list(pairs.keys())[i], 'group'].values[0],
                                g1_sort.loc[g1_sort['id'] == list(pairs.keys())[j], 'group'].values[0]]  # 记录小组
                    g2m_set += [g2_sort.loc[g2_sort['id'] == list(pairs.values())[i], 'group'].values[0],
                                g2_sort.loc[g2_sort['id'] == list(pairs.values())[j], 'group'].values[0]]
                    break
        # print('distance match:', match_num)
        g1m_set, g2m_set = list(set(g1m_set)), list(set(g2m_set))
        g1m_len, g2m_len = sum([len(g1_sort.loc[g1_sort['group'] == gm_n]) for gm_n in g1m_set]), sum([len(g2_sort.loc[g2_sort['group'] == gm_n]) for gm_n in g2m_set])
        if not match_num > min(len(g1), len(g2)) * match_thresh and not ((match_num > min(g1m_len, g2m_len) * match_thresh) and not min(g1m_len, g2m_len) == 1):  # 小组内占比大的也可以匹配
            return False

        # match the angle by center or corner 函数计算每对元素之间的角度（分为角点处的角度和中心点处的角度），如果大多数元素的角度相似，就认为角度匹配成功。
        match_num = 0
        g1m_set, g2m_set = [], []
        for i in range(len(angles_cor)):
            angle_i = angles_cor[i]
            for j in range(len(angles_cor)):
                if i == j:
                    continue
                angle_j = angles_cor[j]
                # compare the pair's distance and angle between the line and the x-axis
                if abs(angle_i - angle_j) < diff_angle:
                    match_num += 1
                    g1m_set += [g1_sort.loc[g1_sort['id']==list(pairs.keys())[i], 'group'].values[0],
                                g1_sort.loc[g1_sort['id']==list(pairs.keys())[j], 'group'].values[0]]  # 记录小组
                    g2m_set += [g2_sort.loc[g2_sort['id'] == list(pairs.values())[i], 'group'].values[0],
                                g2_sort.loc[g2_sort['id'] == list(pairs.values())[j], 'group'].values[0]]
                    break
        g1m_set, g2m_set = list(set(g1m_set)), list(set(g2m_set))
        g1m_len, g2m_len = sum([len(g1_sort.loc[g1_sort['group'] == gm_n]) for gm_n in g1m_set]), sum([len(g2_sort.loc[g2_sort['group'] == gm_n]) for gm_n in g2m_set])
        # print('corner angle match:', match_num)

        # if fail to match corner angle, try to match center angle
        if not match_num > min(len(g1), len(g2)) * match_thresh and not ((match_num > min(g1m_len, g2m_len) * match_thresh) and not min(g1m_len, g2m_len) == 1):
            match_num = 0
            g1m_set, g2m_set = [], []
            for i in range(len(angles_cen)):
                angle_i = angles_cen[i]
                for j in range(len(angles_cen)):
                    if i == j:
                        continue
                    angle_j = angles_cen[j]
                    # compare the pair's distance and angle between the line and the x-axis
                    if abs(angle_i - angle_j) < diff_angle:
                        match_num += 1
                        g1m_set += [g1_sort.loc[g1_sort['id'] == list(pairs.keys())[i], 'group'].values[0],
                                    g1_sort.loc[g1_sort['id'] == list(pairs.keys())[j], 'group'].values[0]]  # 记录小组
                        g2m_set += [g2_sort.loc[g2_sort['id'] == list(pairs.values())[i], 'group'].values[0],
                                    g2_sort.loc[g2_sort['id'] == list(pairs.values())[j], 'group'].values[0]]
                        break
            g1m_set, g2m_set = list(set(g1m_set)), list(set(g2m_set))
            g1m_len, g2m_len = sum([len(g1_sort.loc[g1_sort['group'] == gm_n]) for gm_n in g1m_set]), sum([len(g2_sort.loc[g2_sort['group'] == gm_n]) for gm_n in g2m_set])
            # print('center angle match:', match_num)
            if not match_num > min(len(g1), len(g2)) * match_thresh and not ((match_num > min(g1m_len, g2m_len) * match_thresh) and not min(g1m_len, g2m_len) == 1):
                return False

    # print('Success:', g1.iloc[0]['group'], g2.iloc[0]['group'], distances, max_side)
    # 配对检查 丢弃明显大的
    drop_idx = []
    for data in [distances, angles_cor, angles_cen]:
        data = np.array(data)
        # 计算绝对值的最大值和最小值
        abs_max = np.max(np.abs(data))
        abs_min = np.min(np.abs(data))
        # 添加额外的条件来判断是否需要四分位数检测
        if (abs_max > 3 * abs_min and abs_min != 0) or (abs_max > 5 and abs_min == 0):
            # 计算四分位数
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            # 计算四分位距
            IQR = Q3 - Q1
            # 计算异常值的阈值
            lower_threshold = Q1 - 1.5 * IQR
            upper_threshold = Q3 + 1.5 * IQR
            # 找到异常值的索引
            outliers = np.where((data < lower_threshold) | (data > upper_threshold))[0]
            drop_idx += list(outliers)
    drop_idx += list(np.where(np.abs(distances) > max_height+10)[0])  # 单个距离过滤 放松10个的距离
    drop_idx = list(set(drop_idx))

    if len(set(pairs.keys()) - set(drop_idx))==0:
        return False

    for idx, i in enumerate(pairs):  # 返回配对关系
        if idx not in drop_idx:
            if not swapped:
                g1.loc[i, 'pair_to'] = pairs[i]
                g2.loc[pairs[i], 'pair_to'] = i
            else:
                g2.loc[i, 'pair_to'] = pairs[i]
                g1.loc[pairs[i], 'pair_to'] = i
        # print(pairs[i])

    return True


def match_two_groups_by_distance(g1, g2, diff_distance=1.2, diff_angle=10):
    assert g1.iloc[0]['alignment_in_group'] == g2.iloc[0]['alignment_in_group']
    alignment = g1.iloc[0]['alignment_in_group']
    pairs = {}
    if alignment == 'h':
        g1_sort = g1.sort_values('center_column')
        g2_sort = g2.sort_values('center_column')
        max_side = max(list(g1['height']) + list(g2['height']))
    else:
        g1_sort = g1.sort_values('center_row')
        g2_sort = g2.sort_values('center_row')
        max_side = max(list(g1['width']) + list(g2['width']))

    swapped = False
    if len(g1) == len(g2):
        distances = []
        angles = []
        for i in range(len(g1_sort)):
            c1 = g1_sort.iloc[i]
            c2 = g2_sort.iloc[i]
            distance = calc_compos_distance(c1, c2)
            angle = calc_angle(c1, c2, 'center')
            # mismatch if too far
            if distance > max_side * 2:
                return False
            # compare the pair's distance and angle between the line and the x-axis
            if i > 0:
                if (abs(distance - distances[i-1]) > 10 and max(distance, distances[i-1]) > diff_distance * min(distance, distances[i-1])) and \
                        abs(angle - angles[i - 1]) > diff_angle:
                    return False
            pairs[c1['id']] = c2['id']
            distances.append(distance)
            angles.append(angle)
    else:
        # make sure g1 represents the shorter group while g2 is the longer one
        if len(g1_sort) > len(g2_sort):
            temp = g1_sort
            g1_sort = g2_sort
            g2_sort = temp
            swapped = True

        distances = []
        angles = []
        marked = np.full(len(g2_sort), False)  # mark the matched compo in the g2
        # calculate the distances between each c1 in g1 and all c2 in g2
        for i in range(len(g1_sort)):
            c1 = g1_sort.iloc[i]
            distance = None
            angle = None
            matched_id = None
            for j in range(len(g2_sort)):
                if marked[j]: continue
                c2 = g2_sort.iloc[j]
                d_cur = calc_compos_distance(c1, c2)
                if distance is None or distance > d_cur:
                    distance = d_cur
                    angle = calc_angle(c1, c2, 'center')
                    pairs[c1['id']] = c2['id']
                    # mark the matched compo
                    marked[j] = True
                    # unmark the previously matched compo
                    if matched_id is not None:
                        marked[matched_id] = False
                    matched_id = j
            distances.append(distance)
            angles.append(angle)
        # match the distances and angles
        match_num = 1
        for i in range(len(distances)):
            dis_i = distances[i]
            angle_i = angles[i]
            for j in range(len(distances)):
                if i == j:
                    continue
                dis_j = distances[j]
                angle_j = angles[j]
                # compare the pair's distance and angle between the line and the x-axis
                if max(dis_i, dis_j) < max_side * 2 and\
                        (abs(dis_i - dis_j) <= 10 or max(dis_i, dis_j) < diff_distance * min(dis_i, dis_j)) and\
                        abs(angle_i - angle_j) < diff_angle:
                    match_num += 1
                    break
        # print(g1.iloc[0]['group'], g2.iloc[0]['group'], match_num, distances, max_side)
        if match_num < min(len(g1), len(g2)) * 0.8:
            return False

    # print('Success:', g1.iloc[0]['group'], g2.iloc[0]['group'], distances, max_side)
    for i in pairs:
        if not swapped:
            g1.loc[i, 'pair_to'] = pairs[i]
            g2.loc[pairs[i], 'pair_to'] = i
        else:
            g2.loc[i, 'pair_to'] = pairs[i]
            g1.loc[pairs[i], 'pair_to'] = i
    return True


def pair_matching_within_groups(groups, start_pair_id, new_pairs=True, max_group_diff=2):
    pairs = {}  # {'pair_id': [dataframe of grouped by certain attr]} 存储组之间的配对关系
    pair_id = start_pair_id  # 开始编号
    mark = np.full(len(groups), False)  # 标记每个组是否已经进行了配对操作
    if new_pairs:  # 如果 new_pairs 参数为 True，函数会在进行新的配对操作前，检查并删除已有的 group_pair 列，以确保每次配对都是全新的
        for group in groups:
            if 'group_pair' in group.columns:
                group.drop('group_pair', axis=1, inplace=True)
    for i, g1 in enumerate(groups):  # 函数遍历给定的组列表 groups，每次取出两个组进行比较
        # if mark[i]: continue
        # print(g1['group'])
        alignment1 = g1.iloc[0]['alignment_in_group']
        for j in range(i + 1, len(groups)):
            g2 = groups[j]
            # print(g2['group'])
            alignment2 = g2.iloc[0]['alignment_in_group']
            if alignment1 == alignment2:  # 首先，它检查这两个组的元素属性是否相同（alignment_in_group），如果不相同，则不进行配对 v还是h组
                if alignment1 == 'h' and abs(sum(g1['center_row'])/len(g1)-sum(g2['center_row'])/len(g2)) < 10:
                    continue
                if not match_two_groups_by_angles_and_y_distance(g1, g2):
                    continue  # 调用函数 进一步检查两个组是否具有相似的角度和垂直距离 如果不相似，则不进行配对
                # print(list(g1['group'])[0], mark[i], '-', list(g2['group'])[0], mark[j])
                if not mark[i]:
                    # hasn't paired yet, creat a new pair 如果两个组都没有被标记过（即未进行过配对），则创建一个新的配对编号 pair_id，
                    # 将这两个组标记为已配对，将它们的 group_pair 设置为同一个编号，并将它们添加到 pairs 字典中。
                    if not mark[j]:
                        pair_id += 1
                        g1['group_pair'] = pair_id
                        g2['group_pair'] = pair_id
                        pairs[pair_id] = [g1, g2]
                        mark[i] = True
                        mark[j] = True
                    # if g2 is already paired, set g1's pair_id as g2's 如果其中一个组已经被标记，
                    # 将另一个组的 group_pair 设置为已标记组的 group_pair，并将这个组添加到已标记组所属的配对中。
                    elif alignment1=='v':
                        g1['group_pair'] = g2.iloc[0]['group_pair']
                        pairs[g2.iloc[0]['group_pair']].append(g1)
                        mark[i] = True
                else:
                    # if gi is marked while gj isn't marked
                    if not mark[j] and alignment1=='v':
                        g2['group_pair'] = g1.iloc[0]['group_pair']
                        pairs[g1.iloc[0]['group_pair']].append(g2)
                        mark[j] = True
                    # if gi and gj are all already marked in different group_pair, merge the two group_pairs together
                    elif alignment1=='v':
                        # merge all g2's pairing groups with g1's 如果两个组都已经被标记，并且它们属于不同的配对，
                        # 将这两个配对合并为一个，并将其中一个配对从 pairs 字典中移除
                        if g1.iloc[0]['group_pair'] != g2.iloc[0]['group_pair']:
                            g1_pair_id = g1.iloc[0]['group_pair']
                            g2_pair_id = g2.iloc[0]['group_pair']
                            for g in pairs[g2_pair_id]:
                                g['group_pair'] = g1_pair_id
                                pairs[g1_pair_id].append(g)
                            pairs.pop(g2_pair_id)


    merged_pairs = None
    for i in pairs:
        for group in pairs[i]:
            if merged_pairs is None:
                merged_pairs = group
            else:
                merged_pairs = merged_pairs.append(group, sort=False)
    return merged_pairs


def pair_visualization(pairs, img, img_shape, show_method='line'):
    board = img.copy()
    if show_method == 'line':
        for id in pairs:
            pair = pairs[id]
            for p in pair:
                board = draw.visualize(board, p.compos_dataframe, img_shape, attr='group_pair', show=False)
    elif show_method == 'block':
        for id in pairs:
            pair = pairs[id]
            for p in pair:
                board = draw.visualize_fill(board, p.compos_dataframe, img_shape, attr='group_pair', show=False)
    cv2.imshow('pairs', board)
    cv2.waitKey()
    cv2.destroyAllWindows()


# *****************************************
# Add missed compos by checking group items
# *****************************************
def calc_compo_related_position_in_its_paired_item(compos, all_compos_in_pair):
    related_pos = None
    for i in range(len(compos)):
        compo = compos.iloc[i]
        # get the position of the item to which the compo belongs
        item = all_compos_in_pair[all_compos_in_pair['list_item'] == compo['list_item']]
        item_pos = item['column_min'].min(), item['row_min'].min()

        if related_pos is None:
            related_pos = compo['column_min'] - item_pos[0], compo['row_min'] - item_pos[1], compo['column_max'] - \
                          item_pos[0], compo['row_max'] - item_pos[1]
        else:
            related_pos = (min(related_pos[0], compo['column_min'] - item_pos[0]), min(related_pos[1], compo['row_min'] - item_pos[1]),
                           max(related_pos[2], compo['column_max'] - item_pos[0]), max(related_pos[3], compo['row_max'] - item_pos[1]))
    return related_pos


def calc_intersected_area(bound1, bound2):
    '''
    bound: [column_min, row_min, column_max, row_max]
    '''
    col_min_s = max(bound1[0], bound2[0])
    row_min_s = max(bound1[1], bound2[1])
    col_max_s = min(bound1[2], bound2[2])
    row_max_s = min(bound1[3], bound2[3])
    w = np.maximum(0, col_max_s - col_min_s)
    h = np.maximum(0, row_max_s - row_min_s)
    inter = w * h
    return inter


def find_missed_compo_by_iou_with_potential_area(potential_area, all_compos):
    unpaired_compos = all_compos[all_compos['group_pair'] == -1]
    for i in range(len(unpaired_compos)):
        up = unpaired_compos.iloc[i]
        inter = calc_intersected_area((up['column_min'], up['row_min'], up['column_max'], up['row_max']), potential_area)
        if inter > 0 and inter/up['area'] > 0.5:
            return up['id']
    return None
