# -*- coding:utf-8 -*-
import torch.utils.data as data
import PIL.Image as Image
import os

def make_dataset(root):
    imgs=[]
    #n=len(os.listdir(root))//2
    img=os.listdir(root)
    for i in img:
        #img=os.path.join(root,"%03d.jpg"%i)
        #mask=os.path.join(root,"%03d_mask.png"%i)
        img=os.path.join(root,i)
        mask=os.path.join(root+"_label/"+i[:-4]+".png")
        imgs.append((img,mask))           
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
        img_x = img_x.resize((512, 512))
        img_y = Image.open(y_path)
        img_y = img_y.resize((512, 512))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y, x_path

    def __len__(self):
        return len(self.imgs)
