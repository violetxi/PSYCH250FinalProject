import os
import pdb
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision.utils import save_image

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
        self.save_resized_train_test()
        
    def get_category_image_paths(self):
        def add_full_path(sub_category, image_files):
            return [os.path.join(
                self.raw_folder, sub_category, image_file) \
                    for image_file in image_files]        
        self.category_files = {
            category : {'train_paths': [], 'test_paths': []} \
            for category in SUPER_SUB_CLASSES
        }
        subcategories = os.listdir(self.raw_folder)
        for super_category in SUPER_SUB_CLASSES:
            for sub_category  in SUPER_SUB_CLASSES[super_category]:
                image_files = os.listdir(os.path.join(
                    self.raw_folder, sub_category)
                )
                image_paths = add_full_path(
                    sub_category, image_files)
                train_paths = image_paths[ : self.num_train_per_sub_cate]
                test_paths = image_paths[self.num_train_per_sub_cate : ]
                self.category_files[super_category]['train_paths'].extend(train_paths)
                self.category_files[super_category]['test_paths'].extend(test_paths)
        return            
    
    def save_resize_image(self, src_paths, dst_folder, label, condition):
        try:
            os.makedirs(dst_folder)
        except:
            pass
        for src_path in src_paths:            
            image = Image.open(src_path).convert('RGB')
            image = image.resize((256, 256))
            dst_path = os.path.join(dst_folder, os.path.basename(src_path))
            self.meta[condition].append([dst_path, label])
            image.save(dst_path)
        return

    def save_resized_train_test(self):
        self.meta = {'train': [], 'test': []}
        for class_ in self.category_files:
            label = LABEL_MAP[class_]
            train_paths = self.category_files[class_]['train_paths']
            test_paths = self.category_files[class_]['test_paths']
            train_class_folder = os.path.join(self.processed_folder, 'train', class_)
            test_class_folder = os.path.join(self.processed_folder, 'test', class_)            
            self.save_resize_image(train_paths, train_class_folder, label, 'train')
            self.save_resize_image(test_paths, test_class_folder, label, 'test')
        meta_path = os.path.join(self.processed_folder, 'meta.pkl')
        pickle.dump(self.meta, open(meta_path, 'wb'))
        
        
if __name__ == '__main__':
    raw_folder  = 'data/stimuli'
    processed_folder = 'data/processed'
    processor = RawStimuliProcessor(
        raw_folder, processed_folder)
    
