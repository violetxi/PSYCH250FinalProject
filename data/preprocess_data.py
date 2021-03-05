import os
import torch
import numpy as np
from PIL import Image


# used to aggregate categories into 4 super classes
LABEL_MAP = {'faces':0, 'bodies':1, 'places':2, 'characters':3}
SUPER_SUB_CLASSES = {
    'faces': ['adult', 'child'], 'bodies': ['body', 'limb'],
    'places': ['house', 'corridor'], 'characters': ['word', 'number']}


class RawStimuliProcessor(object):
    def __init__(self,
                 raw_folder,
                 processed_folder,
                 num_train_per_sub_cate=120):
        self.raw_folder = raw_folder
        self.processed_folder = processed_folder
        self.num_train_per_sub_cate = num_train_per_sub_cate
        self.get_category_image_paths()
        self.split_train_val()
        self.save_train_val()
        
    def get_category_image_paths(self):
        def add_full_path(sub_category, image_files):
            return [os.path.join(
                self.raw_folder, sub_category, image_file) \
                    for image_file in image_files]        
        self.category_files = {
            category : [] for category in SUPER_SUB_CLASSES
1;95;0c        }
        subcategories = os.listdir(self.raw_folder)
        for super_category in SUPER_SUB_CLASSES:
            for sub_category  in SUPER_SUB_CLASSES[super_category]:
                image_files = os.listdir(os.path.join(
                    self.raw_folder, sub_category)
                )
                image_paths = add_full_path(
                    sub_category, image_files)
                self.category_files[super_category].extend(image_paths)
        return

    def load_images(self, image_paths):
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((256, 256))
            image = torch.from_numpy(np.array(image))
            image = image.permute(0, 1, 2)
            images.append(image)
        return images
    
    def split_train_val(self):
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        for category in self.category_files:
            files = np.random.permutation(self.category_files[category])
            num_sub_categories = len(SUPER_SUB_CLASSES[category])
            train = files[: self.num_train_per_sub_cate * num_sub_categories]
            val = files[self.num_train_per_sub_cate * num_sub_categories :]            
            train_labels = [LABEL_MAP[category]] * len(train)
            train_images = self.load_images(train)
            val_labels = [LABEL_MAP[category]] * len(val)
            val_images = self.load_images(val)
            self.train_images.extend(train_images)
            self.train_labels.extend(train_labels)
            self.val_images.extend(val_images)
            self.val_labels.extend(val_labels)
            
        self.train_images = torch.stack(self.train_images)
        self.val_images = torch.stack(self.val_images)
        self.train_labels = torch.Tensor(self.train_labels)
        self.val_labels = torch.Tensor(self.val_labels)
        assert self.train_images.shape[0] == self.train_labels.shape[0] \
            and self.val_images.shape[0] == self.val_labels.shape[0], \
            'number of images and labels needs to match!'        

    def save_train_val(self):
        print(self.train_images.shape, self.val_images.shape)
        print(self.train_labels.shape, self.val_labels.shape)
        torch.save(
            [self.train_images, self.train_labels],
            os.path.join(self.processed_folder, 'train.pt'))
        torch.save(
            [self.val_images, self.val_labels],
            os.path.join(self.processed_folder, 'val.pt'))
        
        
if __name__ == '__main__':
    raw_folder  = './stimuli'
    processed_folder = './processed'
    processor = RawStimuliProcessor(
        raw_folder, processed_folder)
    
