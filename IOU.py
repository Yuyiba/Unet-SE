# -*- coding:utf-8 -*-
from pathlib import Path
import argparse
import cv2
import numpy as np

   # intersection = (y_true * y_pred).sum()                # TP真正类 也是 交集
   # FP = (y_true_f * y_pred).sum()                        # FP假正类
   # FN = (y_true * y_pred_f).sum()                        # FN假反类
   # TN = (y_true_f * y_pred_f).sum()                      # TN真反类 
   # union = y_true.sum() + y_pred.sum() - intersection    # 并集   

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection+ 1e-15) / (union+ 1e-15)
def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum()+ 1e-15) / (y_true.sum() + y_pred.sum()+ 1e-15)
def Recall(y_true,y_pred,y_pred_f):
    intersection = (y_true * y_pred).sum()
    FN = (y_true * y_pred_f).sum()
    return((intersection)+ 1e-15) / ((intersection + FN)+ 1e-15)
def Precision(y_true,y_pred,y_true_f):        #查准率（TP/TP+FP）
    intersection = (y_true * y_pred).sum()
    FP = (y_true_f * y_pred).sum()                    
    return(intersection+ 1e-15) / (intersection + FP+ 1e-15)
def FP(y_pred,y_true_f,y_pred_f):              
    FP = (y_true_f * y_pred).sum()                        
    TN = (y_true_f * y_pred_f).sum()                      
    return(FP+ 1e-15 ) / (FP + TN + 1e-15)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--train_path', type=str, default='/home/yus/Documents/u_net_liver-master/result',help='image')#jpg
    arg('--target_path', type=str, default='/home/yus/Documents/u_net_liver-master/data/val_label', help='GT')#png
    args = parser.parse_args()
    result_dice = []
    result_jaccard = []
    result_Recall = []
    result_Precision = []
    result_FP = []
    for file_name in (Path(args.train_path)).glob('*'):


        y_pred = (cv2.imread(str(file_name), 0) >= 127).astype(np.uint8)
        #print(cv2.imread(str(file_name), 0))

        y_pred= cv2.resize(y_pred,(256,256))

        y_pred_f = (cv2.imread(str(file_name), 0) < 127).astype(np.uint8)

        y_pred_f= cv2.resize(y_pred_f,(256,256))

        pred_file_name = str((Path(args.target_path) / file_name.name)).replace('jpg', 'png')

        
        y_true = (cv2.imread(str(pred_file_name), 0) >= 127).astype(np.uint8)


        y_true= cv2.resize(y_true,(256,256))

        y_true_f=(cv2.imread(str(pred_file_name), 0) < 127).astype(np.uint8)       
        y_true_f = cv2.resize(y_true_f,(256,256))


        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]
        result_Recall += [Recall(y_true,y_pred,y_pred_f)]
        result_Precision += [Precision(y_true,y_pred,y_true_f)]
        result_FP += [FP(y_pred,y_true_f,y_pred_f)]

    print('Precision = ', np.mean(result_Precision))
    print('Recall = ', np.mean(result_Recall))
    print('IOU = ', np.mean(result_jaccard)) 
    print('Dice = ', np.mean(result_dice))         
    print('FP = ', np.mean(result_FP))
