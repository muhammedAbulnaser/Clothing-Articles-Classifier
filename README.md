# Clothing Articles Classifier Repo structure
<pre>
├── Clothing Classifier
│   ├── src
│   │   ├── ClothDataset.py
│   │   ├── ModelLoader.py
│   │   ├── ModelTrainerEvaluater.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   │ 
│   ├── data
│   │   ├── styles.csv
│   │   └── image folder
│   │ 
│   ├── notebooks
│   │   └──  simpleEDA.ipynb
│   │ 
│   ├── weights
│   │   └── saved weights
│   │ 
│   ├── main.py
│   │ 
│   └── README.md </pre>

This repository contains a solution for the clothing articles classifier task.

## Problem Statement

The task is to build a classifier for clothing articles. Given an input image of a clothing item, the model should predict the corresponding class or category.

## Quick Start
1. Clone the repo `https://github.com/muhammedAbulnaser/Clothing-Classifier`
2. Download the data[data]([URL](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)) in the data folder with this structure 
<pre>
├── data
│   ├── images "Contains the images" 
|   └── styles.csv "Contains the labels"
</pre>

3. run the `main.py` python file to train and evaluate the model of your choose
<pre>
python main.py --model_type "mobilenet" --data_path "data/" --num_classes 10  --epochs 5 
                              "vgg16" 
                             "resnet" 
</pre>  
 
## Approach

1. **Data Collection**: The dataset provided is a comprehensive collection of data from the e-commerce industry. It includes professionally captured high-resolution product images, along with manually-entered label attributes and descriptive text. Each product is uniquely identified by an ID, and the mapping between products and images is available in the styles.csv file. Additionally, key product categories and their display names are provided for easy reference. The dataset contains around `45k images`with `143 class`. [Link To the Dataset]([URL](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small))

2. **Data Preprocessing Steps**:                                                                                                                                 
    - ***Cleaning Unexisting Files:*** This step ensures the integrity of the dataset by removing records where the corresponding image file does not exist. It iterates over each row in the DataFrame and checks if the image file path exists. Rows with non-existing image files are dropped from the DataFrame.
    - ***Calculate Category “article type ”Counts:*** The code calculates the count of each unique articleType category in the DataFrame. This provides insights into the distribution of categories within the dataset.
    - ***Filter Dataset based on Category Counts:*** The dataset is filtered to focus on categories that have a count greater than `1000`. Rows with article types that do not meet this threshold are removed, resulting in a filtered DataFrame. This step is to minimize the huge imbalancing found in this dataset resulting to only `10 classes` out of `143 classes` originally in the dataset.
    - ***Split Dataset into Training and Testing Sets:*** The filtered DataFrame is split into a training dataset and a testing dataset. The train_test_split function is used, where `80%` of the data is allocated for training and `20%` for testing. Where the random seed ensures reproducibility.
    - ***Define ClothDataset Class:*** The ClothDataset class is defined as a subclass of torch.utils.data.Dataset. It represents the dataset of images and their corresponding article types. The class takes the root directory, the DataFrame (train or test), and an optional transform argument (e.g., resizing and converting to tensors).

3. **Model Architecture**:
For the clothing articles classifier, three model architectures were used: VGG16, ResNet18, and MobileNet. Here's an overview of each model: 
    - ***VGG16:*** VGG16 is a deep convolutional neural network architecture that was introduced by the Visual Geometry Group at the University of Oxford. It consists of `16` convolutional layers, including five max-pooling layers and three fully connected layers. The model has a fixed input size of 224x224 pixels. In PyTorch, the VGG16 model is available through the 'torchvision.models' module.
    - ***ResNet18:*** ResNet18 is a variant of the ResNet (Residual Network) architecture developed by Microsoft Research. It consists of `18` layers, including convolutional layers, batch normalization, and residual connections. Residual connections allow the network to learn residual mappings, which can alleviate the degradation problem in deep neural networks. The ResNet18 model also has an input size of 224x224 pixels and is available in PyTorch's 'torchvision.models' module.
    - ***MobileNet:*** MobileNet is a lightweight convolutional neural network architecture designed for mobile and embedded devices. It utilizes depthwise separable convolutions, which split the standard convolutional operation into a depthwise convolution and a pointwise convolution. This reduces computational complexity and model size while maintaining accuracy. MobileNet is also available in the 'torchvision.models' module.

Each of these models was pretrained on large-scale datasets such as ImageNet. By leveraging transfer learning, the pretrained models' weights were used as a starting point for the clothing articles classification task, and only the final classification layer was modified to match the number of clothing categories in the dataset, `10 classes` after filteration.

4. **Training**:

    - Optimization Algorithm: The SGD optimizer stands for Stochastic Gradient Descent. 

    - Learning Rate Schedule: The learning rate is set to 0.001.
    
    - The loss function: The CrossEntropyLoss used for multi-class classification tasks and combines the softmax activation function and the negative log-likelihood loss. 
    
    - Batch Size: The batch size used for training is 16.

    - Number of Training Epochs: The number of training epochs is 3 due to the lack of resources "Working on free google colab"

5. **Evaluation Metrics**: 
Due to imbalanced dataset
    - ***Micro F1 Score:*** `Micro F1 score` calculates the F1 score by considering the total number of true positives, false positives, and false negatives across all classes. It treats the `imbalanced dataset` as a single classification task and gives equal weight to each sample. Micro F1 score is suitable when you want to `prioritize overall performance` and treat each sample equally.

6. **Results**:

Results of the Micro F1 Score 3 used models:
| Metric type | VGG16 | Resnet18 | MobileNet |
|:-----------:|:-----:|:--------:|:---------:|
|Micro F1 Score|  94.7 % |  91.8 % |  93.3 %  |


## Receptive Field

The receptive field is a concept in convolutional neural networks (CNNs) that refers to the region in the input space that influences the output of a particular neuron or feature map. It represents the effective area of the input that is taken into account when computing the response of a neuron.

Results of the receptive field of 3 used models:
| VGG16 | Resnet18 | MobileNet |
|:-----:|:--------:|:---------:|
|  212 x 212 |  99 x 99 |  315 x 315  |

An example of an overall receptive field of ***resnet18*** model:
![image](https://github.com/muhammedAbulnaser/Clothing-Classifier/assets/63162632/2e915bed-5727-43b0-b5c8-417bffc74631)

Another example of an overall receptive field of ***VGG16*** model:
![image](https://github.com/muhammedAbulnaser/Clothing-Classifier/assets/63162632/e48004f5-926a-449a-98be-969082126424)

 
To increase or decrease the receptive field of the VGG16 model ***for example***, modification to the architecture by adjusting the size of the convolutional kernels and the stride should be done.
    - ***Changing Kernel Size:*** `Increasing` the size of the convolutional kernels in the layers will increase the receptive field size. while `Decreasing` kernel sizes such as 1x1 can decrease the receptive field as it focuses on local information.
    - ***Modifying Stride:*** `Increasing` the stride of the convolutional layers will reduce the spatial resolution leading to decreasing in the receptive field. while `Decreasing` the stride, on the other hand, will increase the receptive field.

## FLOPS and MACCs Calculation

Results of total `FLOPS` and `MACCs`
| Metric type | Params (M) | FLOPs (G) | MACs (G) |
|:-----------:|:-----:|:--------:|:---------:|
|VGG16|  138.36 |  30.96 | 15.47  |		
|Resnet18|  11.69 |  3.64 |  1.82  |		
|MobileNet|  3.5 |  0.63 |  0.31  | 		

where:
    - (M) refers to milion
    - (G) Giga
    
***to generate FLOPS and MACCs per layer as in the next example***
Run the `FlopsMaccsCalculator.py` from the code folder 
![image](https://github.com/muhammedAbulnaser/Clothing-Classifier/assets/63162632/83af0f12-33b7-4afa-add5-9f1bf1ad282c)


## Conclusion

Considering the deployment requirements of minimal model size and number of computations, MobileNet stands out as the most efficient model. It has significantly fewer parameters, FLOPs, and MACs compared to VGG16 and ResNet18 while still achieving a competitive Micro F1 Score of `93.3%`.

Therefore, if the goal is to prioritize efficiency and minimize computational resources, MobileNet would be a suitable choice. However, the final model selection should also consider other factors such as the specific application requirements and the trade-off between model performance and computational cost.


