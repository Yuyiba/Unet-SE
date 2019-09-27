# -*- coding:utf-8 -*-
import torch.utils.data as data
import PIL.Image as Image
import os


# def make_dataset(data_path):
#     list_volume = []  # 读样本数据的代号
#     for line in open(data_path):
#         list_volume.append(line.strip())
#
#     ALL_PATH = "/share/share/data/LTIS/LTISEXTRACT_90"
#     imgs = []
#     # 遍历segmentations下的所以liver2文件夹，若存在一个liver_mask，则添加原文件
#     for volume in list_volume:
#         human_id = volume.split("-")[-1]  # 获取human_id
#         # 若是tumor则修改liver2即可,ct_id_dir=ct图片对应的id号
#         ct_id_dir = os.path.join(ALL_PATH, 'segmentations', 'segmentation-%s' % human_id, 'liver')
#         if os.path.exists(ct_id_dir):
#             for ct_id in os.listdir(ct_id_dir):
#                 ct_path = os.path.join(ALL_PATH,
#                                        'images/volume-%s/volume-%s.nii_%s.png' % (human_id, human_id, ct_id))
#                 mask_path = os.path.join(ALL_PATH,
#                                          'segmentations/segmentation-%s/liver/%s/liver_mask.png' % (human_id, ct_id))
#                 fsize = os.path.getsize(mask_path)
#                 if fsize>1024:
#                     imgs.append((ct_path, mask_path))
#     print(len(imgs))
#     import shutil
#     for i, (ct, mask) in enumerate(imgs[:400]):
#         shutil.copyfile(ct, 'data/train/%03d.png' % i)
#         shutil.copyfile(mask, 'data/train/%03d_mask.png' % i)
#
#     for i, (ct, mask) in enumerate(imgs[-20:]):
#         shutil.copyfile(ct, 'data/val/%03d.png' % i)
#         shutil.copyfile(mask, 'data/val/%03d_mask.png' % i)
#     return imgs


def make_dataset(root):
    imgs=[]
    #n=len(os.listdir(root))//2
    img=os.listdir(root)
    for i in img:
        #img=os.path.join(root,"%03d.jpg"%i)
        #mask=os.path.join(root,"%03d_mask.png"%i)
        img=os.path.join(root,i)
        print(i[:-4])
        mask=os.path.join(root+"_label/"+i[:-4]+".png") ### 1
        imgs.append((img,mask))           #加载图像
    return imgs


class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_x = img_x.resize((256, 256))
        img_y = Image.open(y_path)
        img_y = img_y.resize((256, 256))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y, x_path

    def __len__(self):
        return len(self.imgs)
