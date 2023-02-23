from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Image_and_Masks(Dataset):
    def __init__(
            self,
            root_dir,
            image_size=192,
            crop_size=192,
    ):
        super().__init__()

        self.image_folder = Path(root_dir).joinpath('images')
        self.mask_folder = Path(root_dir).joinpath('masks')
        self.crop_size = crop_size
        self.image_size = image_size
        self.csv_path = Path(root_dir).joinpath('dataset.csv')  # can change
        types_dict = {'filename': str, 'orientation': int}
        df = pd.read_csv(self.csv_path, dtype=types_dict)
        self.images = df['filename'].values
        self.orientations = df['orientation']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_name = self.images[item]
        image_path = Path(self.image_folder).joinpath(image_name)
        mask_path = Path(self.mask_folder).joinpath(image_name)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        orientation = self.orientations[item]
        return image, mask, orientation


if __name__ == "__main__":
    dataset = Image_and_Masks(root_dir='E:/GitHub Repos/segmenter_model_data/checkpoint.pth')
    print(dataset.mask_folder)
