import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class ClothDataset(Dataset):
    
    """
    Custom dataset for close articles images.

    Args:
        root (str): Root directory containing the images.
        dataframe (pd.DataFrame): DataFrame containing the dataset.
        transform (torchvision.transforms.Compose): Image transformation pipeline.

    """

    def __init__(self, root, dataframe, df, transform=None):
        super(ClothDataset, self).__init__()
        self.dataframe = dataframe
        self.root = root
        self.cat_list = df['articleType'].unique()
        self.cat2num = {cat:i for i, cat in enumerate(self.cat_list)}
        self.num2cat = {i:cat for i, cat in enumerate(self.cat_list)}
        if transform is None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor()
            ])
        self.transform = transform
        
    def __getitem__(self, idx):
        line = self.dataframe.iloc[idx]
        cat = line.articleType
        cat_id = self.cat2num[cat]
        img_path = os.path.join(self.root, str(line.id)+'.jpg')
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, cat_id
    
    def get_classes(self):
        return len(self.cat_list)

    def __len__(self):
        return len(self.dataframe)