from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root):
    images_list = []
    n = len(os.listdir(root)) // 2
    for i in range(n):
        img = os.path.join(root, "%03d.png" % i)
        mask = os.path.join(root, "%03d_mask.png" % i)
        images_list.append((img, mask))
    return images_list


class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        images_list = make_dataset(root)
        self.images = images_list
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.images[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.images)
