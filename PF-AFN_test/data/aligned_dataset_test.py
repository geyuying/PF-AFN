import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import linecache

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.fine_height=256
        self.fine_width=192

        self.dataset_size = len(open('demo.txt').readlines())

        dir_I = '_img'
        self.dir_I = os.path.join(opt.dataroot, opt.phase + dir_I)

        dir_C = '_clothes'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)

        dir_E = '_edge'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)

    def __getitem__(self, index):        

        file_path ='demo.txt'
        im_name, c_name = linecache.getline(file_path, index+1).strip().split()

        I_path = os.path.join(self.dir_I,im_name)
        I = Image.open(I_path).convert('RGB')

        params = get_params(self.opt, I.size)
        transform = get_transform(self.opt, params)
        transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        I_tensor = transform(I)

        C_path = os.path.join(self.dir_C,c_name)
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform(C)

        E_path = os.path.join(self.dir_E,c_name)
        E = Image.open(E_path).convert('L')
        E_tensor = transform_E(E)

        input_dict = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor}
        return input_dict

    def __len__(self):
        return self.dataset_size 

    def name(self):
        return 'AlignedDataset'
