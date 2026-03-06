import os
import torch
import numpy as np

from PIL import Image

from tools.general.json_utils import read_json
from tools.dataset.voc_utils import get_color_map_dic


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        return data


class VOC_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, with_id=False, with_mask=False):
        self.root_dir = root_dir

        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(self.root_dir, 'SegmentationClass')

        self.image_id_list = [
            image_id.strip() for image_id in open('./data/%s.txt' % domain).readlines()
        ]

        self.with_id = with_id
        self.with_mask = with_mask

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        image = Image.open(image_path).convert('RGB')
        return image

    def get_mask(self, image_id):
        mask_path = os.path.join(self.mask_dir, image_id + '.png')
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def __getitem__(self, index):
        image_id = self.image_id_list[index]

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_mask:
            data_list.append(self.get_mask(image_id))

        return data_list


class VOC_Dataset_For_WSSS(VOC_Dataset):
    """
    Dataset for training the segmentation network with pseudo labels.
    Returns:
        image, pseudo_mask
    """
    def __init__(self, root_dir, domain, pred_dir, transform=None):
        super().__init__(root_dir, domain, with_id=True, with_mask=False)
        self.pred_dir = pred_dir
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    def __getitem__(self, index):
        image, image_id = super().__getitem__(index)

        pseudo_mask_path = os.path.join(self.pred_dir, image_id + '.png')
        pseudo_mask = Image.open(pseudo_mask_path)

        if self.transform is not None:
            input_dic = {'image': image, 'mask': pseudo_mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            pseudo_mask = output_dic['mask']

        return image, pseudo_mask


class VOC_Dataset_For_Segmentation(VOC_Dataset):
    """
    Dataset for validation/testing with ground-truth masks.
    Returns:
        image, gt_mask
    """
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image': image, 'mask': mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']

        return image, mask


class VOC_Dataset_For_Evaluation(VOC_Dataset):
    """
    Dataset for inference/evaluation.
    Returns:
        image, image_id, gt_mask
    """
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_id=True, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    def __getitem__(self, index):
        image, image_id, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image': image, 'mask': mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']

        return image, image_id, mask