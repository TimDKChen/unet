# Unet Review

<p>
  <!-- Web -->
  <a href="https://docs.expo.dev/workflow/web/">
    <img alt="Supports Expo Web" longdesc="Supports Expo Web" src="https://img.shields.io/badge/web-4630EB.svg?style=flat-square&logo=GOOGLE-CHROME&labelColor=4285F4&logoColor=fff" />
  </a>
</p>

<!-- Tables -->

| Name     | Email                 |
| :------- | --------------------- |
| Tim Chen | tim20136202@gmail.com |

CT segmentation snapshots

<img src=".\CT.gif" alt="Home" style="zoom: 100%;" />


## 📝 What I have learn

##### 1. Unet for biological images segmentation ...

##### 2. nn.BCEWithLogitsLoss, nn.ConvTranspose2d, nn.BatchNorm2d ...


## Environment

#### 1. If out of memory happen, 
#####  minimize the batch_size
#####  torch.no_grad() at the error line

## Train
```shell
python main.py train
```

## Test
load the last saved weight
```shell
python main.py test --ckpt=weights_19.pth
```