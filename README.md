# S15-Project-Mask-and-DepthMap-Images-Prediction-using-Dense-CNN <br>

## Getting Started ## 
## Rewind with Dataset Statistics: ##

**Here we are going to have a dataset module, where already creation part done as Session 14/15 A Assignment.** <br>
|**S.No**|**Image Description**|**Quantity**|
|----|-----------------|--------|
|**1.**|**Background Images**|**100**|
|**2**|**Foreground Images**|**100**|
|**3**|**Foreground Images Masks**|**100**|
|**4**|**Bg_Fg Images**|**400000**|
|**5**|**Bg_Fg Images Masks**|**400000**|
|**6**|**Bg_Fg DepthMap Images**|**400000**|

|**S.No**|**Dataset**|**Link**|
|--------|-----------------|--------|
|**1.**|**Dataset Colab (pynb):**| https://drive.google.com/file/u/0/d/1gVyUY93azAIvZVuts5Pm1J1WG76rYgoA/edit|
|**2.**|**Github link:** https://github.com/srilakshmiv14/EVA-Session-14-15A/blob/master/Session15A_Dataset_Creation.ipynb|
|**3.**|**DepthMap Creation (pynb) Colab:** https://colab.research.google.com/drive/1BvpvWvAAWcUBBtRws20h5am1DiLQsTG3|
|**4.**|**Github Link:** https://github.com/srilakshmiv14/EVA-Session-14-15A/blob/master/Session_15A_Depthmaps_Dataset_Creation.ipynb|


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

**Dense CNN Py File:** https://github.com/srilakshmiv14/S15-Project-Mask-and-DepthMap-Images-Prediction-using-Dense-CNN/blob/master/DNN%20Model/net1.py

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


SSIM is a built in method part of Sci-Kit’s image library so we can just load it up. 

MSE will calculate the mean square error between each pixels for the two images we are comparing. Whereas SSIM will do the opposite and look for similarities within pixels; i.e. if the pixels in the two images line up and or have similar pixel density values. 

The only issues are that MSE tends to have arbitrarily high numbers so it is harder to standardize it. While generally the higher the MSE the least similar they are, if the MSE between picture sets differ appears randomly, it will be harder for us to tell anything. SSIM on the other hand puts everything in a scale of -1 to 1 (but I was not able to produce a score less than 0). A score of 1 meant they are very similar and a score of -1 meant they are not similar. 

SSIM was already imported through skimage, no need to manually code it. Now we create a function that will take in two images, calculate it’s mse and ssim and show us the values all at once.different. In my opinion this is a better metric of measurement.


**Binary Cross Entropy – Logit Loss Function/ BCE Loss Function:**

![Binary Cross Entropy – Logit Loss Function/ BCE Loss Function](https://github.com/srilakshmiv14/S15-Project-Mask-and-DepthMap-Images-Prediction-using-Dense-CNN/blob/master/Binary%20Cross%20Entropy.png)

## Training the Model ##

Here the input images i.e Bg images and BgFg images are passed through forward propagation and loss functions as Criterion 1 and Criterion 2.

Computing Loss between (Output[1], Mask images) as Criterion1 Loss function and Computing Loss between (Output[2], Depth images) as Criterion 2 Loss function.

Overall loss is calculated as 2 * loss 1 + loss 2 

**Inferences and Showing Outputs as 5 Layers:**

1.	Loss 1 of trained model (epochs wise when loss gets decreased this layer will be shown black)
2.	Loss 2 of trained model (epochs wise when loss gets decreased this layer will be shown black)
3.	Predicted Mask Images by obtaining overall loss
4.	Predicted Depth Images by obtaining overall loss
5.	Predicted BgFg Images by obtaining overall loss

## Final Output ##
![](https://github.com/srilakshmiv14/S15-Project-Mask-and-DepthMap-Images-Prediction-using-Dense-CNN/blob/master/Final%20Output%20Complete.png)

## Project Colab Files ##

|**S.No**|**File**|**Link**|
|----|----|----|
|**1.**|**Session15b_Model1-sample1.pynb :**| https://colab.research.google.com/drive/145XcTkhonq7ibvx1VIICpbD3LW0JoFUm|
|**2.**|**Session15b_Model1-sample2.ipynb :**| https://colab.research.google.com/drive/1J8tuFIh64XbQsvyAVTnelbics8CqdlqK|
|**3.**|**Session15b_Model1-sample3.pynb :**| https://colab.research.google.com/drive/1WBZpeQwdJqWDQplD9WI5vj4FwWAvuBxg|
|**4.**|**Session15b_Model1-sample4.pynb :**| https://colab.research.google.com/drive/19ZcXkcEx6IdAnPb9hbWJQHlzGHcItkVP|


## ***Submitted by: Srilakshmi V*** ##
