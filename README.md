# S15-Project-Mask-and-DepthMap-Images-Prediction-using-Dense-CNN <br>

## Getting Started ## 
## Rewind with Dataset Statistics: ##

**Here we are going to have a dataset module, where already creation part done as Session 14/15 A Assignment.** <br>

[Dataset Colab (pynb):] (https://drive.google.com/file/u/0/d/1gVyUY93azAIvZVuts5Pm1J1WG76rYgoA/edit) <br>
[Github link:] (https://github.com/srilakshmiv14/EVA-Session-14-15A/blob/master/Session15A_Dataset_Creation.ipynb) <br>
[DepthMap Creation (pynb) Colab: ] (https://colab.research.google.com/drive/1BvpvWvAAWcUBBtRws20h5am1DiLQsTG3) <br>
[Github Link:] (https://github.com/srilakshmiv14/EVA-Session-14-15A/blob/master/Session_15A_Depthmaps_Dataset_Creation.ipynb) <br>


## Dataset and Data Loaders ##
Dataset: Here we will construct a Dataset class which takes input of all 4 images, where to the Network Passing Background and Foreground- Background images as Input to Convolutional Blocks.
And Target images are Masks and Depth Images.
Transformations: We will apply some scale transformations before loading the data.
As both the inputs are of 224 * 224 size, we will resize it as 128 * 128 or 64 * 64
-Grayscale Transformations has been done.
Dataloaders: Here we will fetch images in batches apply transforms to them & then returns dataloders for train and validation phases.

Dataset Utilities (https://github.com/srilakshmiv14/S15-Project-Mask-and-DepthMap-Images-Prediction-using-Dense-CNN/tree/master/utils) <br>
