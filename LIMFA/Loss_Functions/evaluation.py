# -*-coding:utf-8-*-
"""
@Project    : LIMFA
@Time       : 2022/3/28 14:39
@Author     : Danke Wu
@File       : evaluation.py
"""
def count_2class(prediction, y):
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]

        ## for class 1
        if Act == 0 and Pre == 0: TP1 += 1
        if Act == 0 and Pre != 0: FN1 += 1
        if Act != 0 and Pre == 0: FP1 += 1
        if Act != 0 and Pre != 0: TN1 += 1
        ## for class 2
        if Act == 1 and Pre == 1: TP2 += 1
        if Act == 1 and Pre != 1: FN2 += 1
        if Act != 1 and Pre == 1: FP2 += 1
        if Act != 1 and Pre != 1: TN2 += 1

    return TP1, FN1, FP1, TN1, TP2, FN2, FP2, TN2


def evaluationclass( conter_dict, num_sample):  # 2 dim

    ACC_dict = {}

    ## print result
    ACC_dict['Acc_all'] = round(float(conter_dict['TP1'] + conter_dict['TP2'] ) / float(num_sample), 4)
    ACC_dict['Acc1'] = round(float(conter_dict['TP1']  + conter_dict['TN1'] ) / float(conter_dict['TP1'] + conter_dict['TN1'] + conter_dict['FN1'] + conter_dict['FP1'] ), 4)
    if (conter_dict['TP1']  + conter_dict['FP1'] )==0:
        ACC_dict['Prec1'] =0
    else:
        ACC_dict['Prec1'] = round(float(conter_dict['TP1'] ) / float(conter_dict['TP1']  + conter_dict['FP1']), 4)
    if (conter_dict['TP1']  + conter_dict['FN1']  )==0:
        ACC_dict['Recll1'] =0
    else:
        ACC_dict['Recll1'] = round(float(conter_dict['TP1'] ) / float(conter_dict['TP1']+ conter_dict['FN1']), 4)
    if (ACC_dict['Prec1'] + ACC_dict['Recll1'] )==0:
        ACC_dict['F1'] =0
    else:
        ACC_dict['F1'] = round(2 * ACC_dict['Prec1'] * ACC_dict['Recll1'] / (ACC_dict['Prec1'] + ACC_dict['Recll1'] ), 4)

    ACC_dict['Acc2'] = round(float(conter_dict['TP2']  + conter_dict['TN2']) / float(conter_dict['TP2'] + conter_dict['TN2'] + conter_dict['FN2'] + conter_dict['FP2'] ), 4)
    if (conter_dict['TP2']  + conter_dict['FP2'] )==0:
        ACC_dict['Prec2'] =0
    else:
        ACC_dict['Prec2'] = round(float(conter_dict['TP2'] ) / float(conter_dict['TP2'] + conter_dict['FP2'] ), 4)
    if (conter_dict['TP2']  + conter_dict['FN2'] )==0:
        ACC_dict['Recll2'] =0
    else:
        ACC_dict['Recll2'] = round(float(conter_dict['TP2'] ) / float(conter_dict['TP2'] + conter_dict['FN2'] ), 4)
    if (ACC_dict['Prec2'] + ACC_dict['Recll2'] )==0:
        ACC_dict['F2'] =0
    else:
        ACC_dict['F2'] = round(2 * ACC_dict['Prec2'] * ACC_dict['Recll2'] / (ACC_dict['Prec2'] + ACC_dict['Recll2'] ), 4)

    return ACC_dict


