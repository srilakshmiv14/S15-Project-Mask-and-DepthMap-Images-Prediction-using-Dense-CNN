# S15-Project-Mask-and-DepthMap-Images-Prediction-using-Dense-CNN <br>

## Getting Started ## 
## Rewind with Dataset Statistics: ##

**Here we are going to have a dataset module, where already creation part done as Session 14/15 A Assignment.** <br>

**Dataset Colab (pynb):** https://drive.google.com/file/u/0/d/1gVyUY93azAIvZVuts5Pm1J1WG76rYgoA/edit <br>
**Github link:** https://github.com/srilakshmiv14/EVA-Session-14-15A/blob/master/Session15A_Dataset_Creation.ipynb <br>
**DepthMap Creation (pynb) Colab:** https://colab.research.google.com/drive/1BvpvWvAAWcUBBtRws20h5am1DiLQsTG3 <br>
**Github Link:** https://github.com/srilakshmiv14/EVA-Session-14-15A/blob/master/Session_15A_Depthmaps_Dataset_Creation.ipynb <br>


## Dataset and Data Loaders ##
**Dataset:** Here we will construct a Dataset class which takes input of all 4 images, where to the Network Passing Background and Foreground- Background images as Input to Convolutional Blocks. Target images are Masks and Depth Images. <br>
**Transformations:** We will apply some scale transformations before loading the data.
As both the inputs are of 224 * 224 size, we will resize it as 128 * 128 or 64 * 64
-Grayscale Transformations has been done. <br>
**Dataloaders:** Here we will fetch images in batches apply transforms to them & then returns dataloders for train and validation phases.

**Dataset Utilities:** https://github.com/srilakshmiv14/S15-Project-Mask-and-DepthMap-Images-Prediction-using-Dense-CNN/tree/master/utils <br>

## Mask and DepthMap Images Prediction using Dense CNN Architecture ##
![Mask and DepthMap Images Prediction using Dense CNN Architecture](https://github.com/srilakshmiv14/S15-Project-Mask-and-DepthMap-Images-Prediction-using-Dense-CNN/blob/master/DepthMap%20and%20Mask%20Image%20Predictor%20CNN%20Architecture.png)


**Architecture Description**  <br>
***Input Layer:***  IN channels = 6 (Bg – 3, BgFg – 3), OUT channels – 64, Kernel Size – 3 X 3  <br>
***Layer 1:*** IN = 64, OUT = 128, Kernel Size – 3 X 3 <br>
***Residual Block 1:*** IN = 128, OUT=128, Kernel Size – 3*3 (2 Times) <br>
***Layer 2:*** IN = 128, OUT=128, Kernel Size – 3 X 3 <br>
***Layer 3:*** IN = 128, OUT=256, Kernel Size – 3 X 3 <br>
***Residual Block 2:*** IN = 256, OUT=256, Kernel Size – 3 X 3 (2 Times) <br>

**Convolution Blocks**
***Conv 1:*** IN = 3, OUT = 32, Kernel Size – 3 X 3 <br>
***Conv 2:*** IN = 32, OUT = 32, Kernel Size – 3 X 3, Groups - 32 <br>
              IN = 32, OUT = 64, Kernel Size –3 X 3 <br>
***Conv 3:*** IN = 128, OUT = 256, Kernel Size – 3 X 3 <br>
***Conv 4:*** IN = 256, OUT = 256, Kernel Size – 3 X 3 <br>

**Forward Block**
•	Concatenation of both the input images – Bg and BgFG images <Br>
•	x = Input Layer <Br>
•	Passing Layer 1 <Br>
•	Appending Residual Block1<Br>
•	Passing Layer 2 <Br>
•	Passing Layer 3<Br>
•	Appending Residual block <Br>
•	Passing Layer 4<Br>
•	Making Out  2 forward layers<Br>
•	Concatenating of 2 forward layers <Br>
•	Returns Output [0] and Output[1] <Br>

**Number of Parameters : 2752064**

**Loss functions:**

We are using BCE Loss function and SSIM Loss function.

**SSIM Loss function:**
Default loss function in encoder-decoder based image reconstruction had been L2 loss. Previously, Caffe only provides L2 loss as a built-in loss layer. Generally, L2 loss makes reconstructed image blurry because minimizing L2 loss means maximizing log-likelihood of Gaussian. As you know Gaussian is unimodal.

L1 gains a popularity over L2 because it tends to create less blurry images. However, using either L1 or L2 loss in learning takes enormous time to converge. Both losses are pointwise, error is back-propagated by pixel by pixel.

Recently have discovered using SSIM Loss in github for image restructuring:

https://github.com/arraiyopensource/kornia

SSIM loss compares local region of target pixel between reconstructed and original images, whereas L1 loss compares pixel by pixel.

I compare perceptual loss and perceptual loss + SSIM loss in reconstruction of images. We can see perceptual + SSIM loss outperforms only perceptual loss. You can inspect more about SSIM in neural network field in Arxiv. 

**Basic Usage of Loss Function**

 import pytorch_ssim <br>
import torch <br>
from torch.autograd import Variable <br>

img1 = Variable(torch.rand(1, 1, 256, 256)) <br>
img2 = Variable(torch.rand(1, 1, 256, 256)) <br>

if torch.cuda.is_available(): <br>
    img1 = img1.cuda() <br>
    img2 = img2.cuda() <br>
 
print(pytorch_ssim.ssim(img1, img2)) <br>

ssim_loss = pytorch_ssim.SSIM(window_size = 11) <br>

print(ssim_loss(img1, img2)) <br>
