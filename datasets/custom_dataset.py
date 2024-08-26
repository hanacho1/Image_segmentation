import os
import numpy as np
from PIL import Image
import torch.utils.data as data

class CustomSegmentation(data.Dataset):
    """Custom Segmentation Dataset.
    Args:
        root (string): Root directory of the dataset.
        image_set (string, optional): Select the image_set to use, e.g., ``train``, ``val``
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self, root, image_set='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set
        
        image_dir = os.path.join(self.root, 'JPEGImages')
        mask_dir = os.path.join(self.root, 'SegmentationClass')
        splits_dir = os.path.join(self.root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError('Specified image_set does not exist.')

        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def decode_target(self, mask):
        """Decode segmentation class labels into a color image."""
        colormap = {
            0: (0, 0, 0),       # Background
            1: (255, 0, 0),     # Class 1
            2: (0, 255, 0),     # Class 2
            3: (0, 0, 255),     # Class 3
            4: (255, 255, 0),   # Class 4
            5: (0, 255, 255),   # Class 5
        }
        mask = mask.squeeze()  # Remove singleton dimensions if present
        color_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label, color in colormap.items():
            color_image[mask == label] = color
        return color_image
        
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        #mask = Image.open(self.masks[index])
        mask = Image.open(self.masks[index]).convert('RGB')  # Ensure mask is in RGB       
        mask = self.rgb_to_mask(mask)
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.images)

    def rgb_to_mask(self, mask_rgb):
        """Convert RGB mask to a class index mask.
        Args:
            mask_rgb (PIL.Image): RGB mask image.

        Returns:
            numpy.ndarray: An array representing the class index mask.
        """
        # Define the colormap for corresponding classes including background
        colormap = {
            (0, 0, 0): 0,       # Background
            (255, 0, 0): 1,     # Class 1
            (0, 255, 0): 2,     # Class 2
            (0, 0, 255): 3,     # Class 3
            (255, 255, 0): 4,   # Class 4
            (0, 255, 255): 5    # Class 5
        }
        # Convert the RGB mask image to a numpy array
        mask_rgb = np.array(mask_rgb)
        mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.int32)

        for rgb, index in colormap.items():
            matches = (mask_rgb == rgb).all(axis=2)
            mask[matches] = index

        return mask
        