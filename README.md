# AstroSWIN
![astroswin_tiny](https://github.com/user-attachments/assets/b0771b0d-2189-4a60-9c5f-a404aa7dadb8)

SWIN Transformer adaptation for sharpening astrophotography images

## Contents

This repository contains jupyter notebooks for adaptation of the pretrained SWIN transformer for superresolution to astrophotography images sharpening task. Here I include five main notebooks:
- collect_astrobin_data.ipynb, in which I download IOTD's from astrobin.com. Unfortunately, I was too lazy to investigate into their API, so I just wrote a buch of code to parse astrobin's responses and retrieve urls to download high-res pictures (not in full res to spend less time)
- collect_hubble.ipynb, in which I collect images from esahubble archive. This notebook contains an issue which lead me to download only a couple of all the mighty images of Hubble
- preprocessing.ipynb, in which I cutted down all downloaded images into 256x256 patches and splitted them into train and test subsets to make preprocessing and training process cheaper. Unfortunately, now it's obvious to me that it will lead to overfitting and I had to cut them into 512x512 or 1024x1024 patches and then to RandomCrop while training, but now as I didn't tune my model for too long, I didn't face that problem
- train_aswin.ipynb, which contains all loss functions, custom blur functions and the training process, which can be further improved as the next step
- infer_astroswin.ipynb, which contains all necessary functions to download and run the model out of the box in google colab environment

## Important notes

Some parts of code that I included in my jupyter notebooks were generated using Deepseek-R1 model in deep think mode. It helped me a lot in various tasks, as I don't know much about computer vision specifics. Sometimes it produced code with bugs, which were fixed by me, but nontheless I cannot be 100% sure all the code is written as intended.

## Training process

All the training process is splitted into two big stages:
1. Domain adaptaion on noisy and sometimes low quality data
2. Tuning on clean preprocessed data

I trained the model on my home laptop MSI P65 Creator 9SD with NVIDIA GeForce 1660Ti Max-Q (6GB), so it was very hard not to get OOM. On every occurred OOM I moved further to the next epoch.

### First stage

At the first stage I focused primarily on making model work at least:
- I used data only from astrobin IotD's (Image of the Day), and of course I used only miniatures in jpeg format
- I started with just simple MSE + MAE losses, corrupted images random crops with Gaussian Blur filter from torchvision, trained full first epoch
- During cherry-picking model's results on the evaluation set I noticed that it can reproduce dusty parts of image, but screws up on stars
- I added loss over brightness gradient to help model focus on sudden changes in brightess -- it helped a lot, but model started to spoil colors
- After that I added histogram loss to fix model's output color space
- And finally, on the 4th and 5th epochs I trained the model with gradient loss, histogram loss and combined MSE + MAE losses, applying them to sigmoid or 1 minus sigmoid to focus on bright or dark locations respectively.

### Second stage

The first stage produced very harsh model which oversharpened images so it was only useable when I made linear combination of raw input image and processed image like $\alpha \times raw + (1 - \alpha) \times processed$. So at the second stage I decided to focus on making the model's output more smooth. I asked deepseek for advice and it suggested me to use adaptive homogenity background loss. TLDR: it helped to find a tradeoff, but I found that the more I tune the model with this loss function, the more smooth results it produces.

- After the first stage I decided to clean up my data -- I removed low-quality images from astrobin IotD folder and added as much ESA hubble photos as I could gather. Unfortunately, during the hubble's images collection I missed that sometimes one image does not contain LARGE image option, but contains PUBLICATION-READY. I had to modify my downloading script, but I was too lazy to do this.
- I also did full preprocessing to avoid CUDA OOM's during the training process -- I sliced all images into patches 256x256 and splitted patches themselves into two sets -- train and test. Now in a retrospective view I admit this was a very very bad idea, but it worked and, I hope, the next iteration will be more fair and clean.
- The 6th epoch I tuned with preference to AHBG loss. It helped model to become more smooth overall, without losing its ability to restore details in dusty parts of the image. But sometimes it still spoils color, making it more neutral. The resulting checkpoint is saved as astroswin_v06.
- The 7th epoch I tuned with preference to Histogram loss, but also included AHBG loss. The model increased its smoothness, but I managed to fix its output colors.

## Further work

As a continuation, I see mastering the training process:
- Fixing data to avoid potential model overfitting
- Fixing losses weights to get more suitable results
- Adding noise to image patches or combining blurry distortions to make the task harder.
