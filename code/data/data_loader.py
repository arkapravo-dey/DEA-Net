import os, random
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop


def get_clear_name(hazy_name):
    if hazy_name.startswith(('thin_', 'moderate_', 'thick_')):
        return hazy_name.split('_', 1)[1]   # remove prefix only
    
    elif '_' in hazy_name:
        return hazy_name.split('_')[0] + '.png'
    
    else:
        return hazy_name


class TrainDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TrainDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = sorted(os.listdir(hazy_path))

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = get_clear_name(hazy_image_name)

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        # random crop
        crop_params = RandomCrop.get_params(hazy, [256, 256])
        hazy = crop(hazy, *crop_params)
        clear = crop(clear, *crop_params)

        # random rotation
        rotate_params = random.randint(0, 3) * 90
        hazy = rotate(hazy, rotate_params)
        clear = rotate(clear, rotate_params)

        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear

    def __len__(self):
        return len(self.hazy_image_list)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = sorted(os.listdir(hazy_path))

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = get_clear_name(hazy_image_name)

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear, hazy_image_name

    def __len__(self):
        return len(self.hazy_image_list)


class ValDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(ValDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = sorted(os.listdir(hazy_path))

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = get_clear_name(hazy_image_name)

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return {'hazy': hazy, 'clear': clear, 'filename': hazy_image_name}

    def __len__(self):
        return len(self.hazy_image_list)