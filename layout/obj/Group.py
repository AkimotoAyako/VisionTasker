import difflib
import numpy as np


def split_groups(compos):
    grps = {}
    groups = compos.groupby('group').groups
    for i in groups:
        if len(groups[i]) <= 1:
            continue
        grp = Group(i, compos.loc[groups[i]])
        grps[grp.id] = grp
    return grps


def is_valid_group_by_similar_interleave(grp_interleaves, need_rectify_compos):
    if len(grp_interleaves) == 1:
        return False
    for i in range(len(grp_interleaves) - 1):
        inter_a = list(grp_interleaves[i]['group'])
        inter_b = list(grp_interleaves[i + 1]['group'])
        # print(sw_a, sw_b)
        if len(inter_a) <= 1 and len(inter_b) <= 1:
            need_rectify_compos.append(grp_interleaves[i])
            if i == len(grp_interleaves) - 2:
                need_rectify_compos.append(grp_interleaves[i+1])
            continue
        sm = difflib.SequenceMatcher(None, inter_a, inter_b).ratio()
        if sm < 0.5:
            return False
    return True


def find_interleaves_in_group(group, compos_all):
    c_all = compos_all.sort_values('row_min')
    alignment = group.alignment
    interleaves = []
    for i in range(len(group.compos) - 1):
        c_a = group.compos.iloc[i]
        c_b = group.compos.iloc[i + 1]
        col_min = min(c_a['column_min'], c_b['column_min'])
        col_max = max(c_a['column_max'], c_b['column_max'])
        row_min = min(c_a['row_min'], c_b['row_min'])
        row_max = max(c_a['row_max'], c_b['row_max'])
        interleaving = c_all[c_all['group'] != group.id]
        if alignment == 'h':
            # filter elements not on the same horizontal level
            interleaving = interleaving[~((interleaving['row_max'] < row_min) | (interleaving['row_min'] > row_max))]
            # c_a is on the left of c_b
            interleaving = interleaving[(interleaving['column_min'] > c_a['column_max']) & (interleaving['column_max'] < c_b['column_min'])]
        elif alignment == 'v':
            # filter elements not on the same vertical level
            interleaving = interleaving[~((interleaving['column_max'] < col_min) | (interleaving['column_min'] > col_max))]
            # c_a is above c_b
            interleaving = interleaving[(interleaving['row_min'] > c_a['row_max']) & (interleaving['row_max'] < c_b['row_min'])]
        interleaves.append(interleaving)
    return interleaves


def check_valid_group_by_interleaving(compos_all, rectify_compos=True):
    groups = split_groups(compos_all)
    for gid in groups:
        grp = groups[gid]
        interleaves = find_interleaves_in_group(grp, compos_all)
        need_rectify_compos = []
        if is_valid_group_by_similar_interleave(interleaves, need_rectify_compos):
            if rectify_compos and len(need_rectify_compos) > 0:
                for c in need_rectify_compos:
                    if len(c) > 0 and c.iloc[0]['group'] == -1:
                        c = grp.add_compo(c)
                        compos_all.loc[list(c['id'])] = c
        else:
            compos_all.loc[compos_all[compos_all['group'] == gid].id, 'group_nontext'] = -1
            compos_all.loc[compos_all[compos_all['group'] == gid].id, 'group_text'] = -1
            compos_all.loc[compos_all[compos_all['group'] == gid].id, 'alignment_in_group'] = -1
            compos_all.loc[compos_all[compos_all['group'] == gid].id, 'group'] = -1
            print('invalid:', gid)


class Group:
    def __init__(self, id, compos):
        self.id = id
        self.compos = compos                                    # dataframe
        self.alignment = compos.iloc[0]['alignment_in_group']   # group element's alignment
        self.cls = compos.iloc[0]['class']

        self.top, self.bottom, self.left, self.right = self.group_boundary()
        self.width = self.bottom - self.top
        self.height = self.right - self.left
        self.area = self.width * self.height

        self.intersected_grps = []
        self.sort_group()

    def group_boundary(self):
        compo = self.compos
        top = compo['row_min'].min()
        left = compo['column_min'].min()
        bottom = compo['row_max'].max()
        right = compo['column_max'].max()
        return top, bottom, left, right

    def get_boundary(self):
        return self.top, self.bottom, self.left, self.right

    def is_intersected(self, group_b):
        top_a, bottom_a, left_a, right_a = self.get_boundary()
        top_b, bottom_b, left_b, right_b = group_b.get_boundary()
        top_s = max(top_a, top_b)
        bottom_s = min(bottom_a, bottom_b)
        left_s = max(left_a, left_b)
        right_s = min(right_a, right_b)
        w = max(0, bottom_s - top_s)
        h = max(0, right_s - left_s)
        if w == 0 or h == 0:
            return False
        return True

    def add_compo(self, compo):
        compo['class'] = self.cls
        compo['alignment_in_group'] = self.alignment
        compo['group'] = self.id
        compo['group_text'] = self.compos.iloc[0]['group_text']
        compo['group_nontext'] = self.compos.iloc[0]['group_nontext']
        self.compos.append(compo)
        self.sort_group()
        return compo

    def sort_group(self):
        if self.alignment == 'v':
            self.compos.sort_values('row_min', inplace=True)
        else:
            self.compos.sort_values('column_min', inplace=True)
