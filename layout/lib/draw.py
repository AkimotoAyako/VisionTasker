import cv2
import numpy as np
from random import randint as rint


def random_color():
    return rint(0, 255), rint(0, 255), rint(0, 255)


def draw_label(img, bound, color, text=None, line=2):
    cv2.rectangle(img, (bound[0], bound[1]), (bound[2], bound[3]), color, line)
    if text is not None:
        # put text with rectangle
        (w,h),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (bound[0], bound[1] - 20), (bound[0] + w, bound[1] - 20 + h), color, -1)
        cv2.putText(img, text, (bound[0] + 3, bound[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


def visualize_group_transparent(board, df, group_name, alpha=0.5, beta=0.6, color=(255,0,0), show=True):
    groups = df.groupby(group_name).groups
    mask = np.zeros(board.shape, dtype=np.uint8)
    for i in groups:
        if i == -1: continue
        left = df.loc[groups[i], 'column_min'].min()
        right = df.loc[groups[i], 'column_max'].max()
        top = df.loc[groups[i], 'row_min'].min()
        bottom = df.loc[groups[i], 'row_max'].max()
        cv2.rectangle(mask, (left, top), (right, bottom), color, -1)
    board = cv2.addWeighted(mask, alpha, board, beta, 1)
    if show:
        cv2.imshow(group_name, board)
        cv2.waitKey()
        cv2.destroyWindow(group_name)
    return board


def visualize(img, compos_df, resize_shape=None, attr='class', name='board', show=True):
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape)

    board = img.copy()
    for i in range(len(compos_df)):
        compo = compos_df.iloc[i]
        board = cv2.rectangle(board, (compo.column_min, compo.row_min), (compo.column_max, compo.row_max), (255, 0, 0))
        board = cv2.putText(board, str(compo[attr]), (compo.column_min + 5, compo.row_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    if show:
        cv2.imshow(name, board)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return board


def visualize_fill(img, compos_df, resize_shape=None, attr='class', name='board', show=True):  # group pair用
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape)
    colors = {}
    board = img.copy()
    for i in range(len(compos_df)):
        compo = compos_df.iloc[i]
        if compo[attr] == -1:
            # board = cv2.rectangle(board, (compo.column_min, compo.row_min), (compo.column_max, compo.row_max), random_color(), -1)
            continue
        else:
            # compo[attr] = compo[attr].replace('nt', 'c')
            if compo[attr] not in colors:
                colors[compo[attr]] = random_color()
        if compo['class'] == 'Block':
            board = cv2.rectangle(board, (compo.column_min, compo.row_min), (compo.column_max, compo.row_max), colors[compo[attr]], 2)
        else:
            board = cv2.rectangle(board, (compo.column_min, compo.row_min), (compo.column_max, compo.row_max), colors[compo[attr]], -1)
        board = cv2.putText(board, str(compo[attr]), (compo.column_min + 5, compo.row_min + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    if show:
        cv2.imshow(name, cv2.resize(board, (500, 800)))
        cv2.waitKey()
        cv2.destroyWindow(name)
    return board
