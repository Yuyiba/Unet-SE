# -*- coding:utf-8 -*-
from pathlib import Path
import argparse
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--predimage', type=str, default='/home/yus/Documents/Unet-SE-master/result',help='image')
    arg('--groudtruth', type=str, default='/home/yus/Documents/Unet-SE-master/data/test_label', help='GT')
    args = parser.parse_args()

    cmatrix = np.zeros((2,2))
    all_acc = 0

    for file_name in (Path(args.train_path)).glob('*'):

        y_pred = (cv2.imread(str(file_name), 0) >= 127).astype(np.uint8)
        y_pred= cv2.resize(y_pred,(512,512))

        # Change as needed groundtruth images name
        tmp_file_name = str((Path(args.target_path) / file_name.name))[:-4]
        tmp_file_name = tmp_file_name + '_gt.png'    

        y_true = (cv2.imread(str(tmp_file_name), 0) >= 127).astype(np.uint8)
        y_true= cv2.resize(y_true,(512,512))

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        cmatrix=cmatrix+confusion_matrix(y_true,y_pred)
        '''
        # TN真反类  c00
        # FP假正类  c01
        # FN假反类  c10
        # TP真正类  c11
        '''
    all_acc=(cmatrix/cmatrix.sum()).trace()
    Recall = cmatrix[1,1]/(cmatrix[1,1]+cmatrix[1,0])
    Precision = cmatrix[1,1]/(cmatrix[1,1]+cmatrix[0,1])
    Iou = cmatrix[1,1]/(cmatrix[1,1]+cmatrix[0,1]+cmatrix[1,0])

    print('  Iou    ={:0.5f}'.format(Iou))
    print('all_acc  ={:0.5f}'.format(all_acc))
    print(' Recall  ={:0.5f}'.format(Recall))
    print('Precision={:0.5f}'.format(Precision))


