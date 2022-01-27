import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
import matplotlib.pyplot as plt
from matplotlib import animation
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.empty_cache()

# check cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# preprocess
x_transforms = transforms.Compose([
    # nump.ndarray (H, W, C) ==> (C, H, W) and / 255 for normalization [0, 1]
    transforms.ToTensor(),
    # (x-mean)/std ==> [-1, 1]
    transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
])

# transfer mask to tensor
y_transforms = transforms.ToTensor()


def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/ {}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # get the value of loss
            epoch_loss += loss.item()
            print("%d/%d, train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model


def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train", transform=x_transforms, target_transform=y_transforms)
    data_loaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, data_loaders)


# print the result of the model
def test(args):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    data_loaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    plt.ion()
    with torch.no_grad():
        # count = 0
        for x, _ in data_loaders:
            y = model(x).sigmoid()
            img_y = torch.squeeze(y).numpy()
            plt.imshow(img_y)
            # plt.savefig(f'./images/{count}.jpg')
            # count += 1
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    #  parse args
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
