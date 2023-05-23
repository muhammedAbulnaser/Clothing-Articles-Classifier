# Repository structure
<pre>
├── Clothing Classifier
│   ├── code
│   │   ├── models.py
│   │   ├── utils.py
│   │   └── preprocessing.py
│   ├── data
│   │   ├── dataset.csv
│   │   └── image folder
│   ├── notebooks
│   │   └──  exploration.ipynb
│   ├── README.md </pre>

This repository contains a solution for the clothing articles classifier task.

## Problem Statement

The task is to build a classifier for clothing articles. Given an input image of a clothing item, the model should predict the corresponding class or category.

## Approach

1. **Data Collection**: The dataset provided is a comprehensive collection of data from the e-commerce industry. It includes professionally captured high-resolution product images, along with manually-entered label attributes and descriptive text. Each product is uniquely identified by an ID, and the mapping between products and images is available in the styles.csv file. Additionally, key product categories and their display names are provided for easy reference. The dataset contains around ***45k images*** with ***143 class***. [Link To the Dataset]([URL](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small))

2. **Data Preprocessing**:                                                                                                                                        - Data Preprocessing Steps:
    - Cleaning Unexisting Files: This step ensures the integrity of the dataset by removing records where the corresponding image file does not exist. It iterates over each row in the DataFrame and checks if the image file path exists. Rows with non-existing image files are dropped from the DataFrame.
    - Calculate Category “article type ”Counts: The code calculates the count of each unique articleType category in the DataFrame. This provides insights into the distribution of categories within the dataset.
    - Filter Dataset based on Category Counts: The dataset is filtered to focus on categories that have a count greater than 1000. Rows with article types that do not meet this threshold are removed, resulting in a filtered DataFrame. This step is to minimize the huge imbalancing found in this dataset resulting to only ***10 classes*** out of ***143 classes*** originally in the dataset.
    - Split Dataset into Training and Testing Sets: The filtered DataFrame is split into a training dataset and a testing dataset. The train_test_split function is used, where 80% of the data is allocated for training and 20% for testing. Where the random seed ensures reproducibility.
    - Define ClothDataset Class: The ClothDataset class is defined as a subclass of torch.utils.data.Dataset. It represents the dataset of images and their corresponding article types. The class takes the root directory, the DataFrame (train or test), and an optional transform argument (e.g., resizing and converting to tensors).
    - 

3. **Model Architecture**: Provide an overview of the chosen model architecture for the clothing classifier. If you used a pre-trained model, mention the specific model and any modifications made. Include the number of parameters in the model.

4. **Training**: Describe the training process, including the optimization algorithm used, the learning rate schedule, the batch size, and the number of training epochs. Mention any regularization techniques applied, such as dropout or weight decay.

5. **Evaluation Metrics**: Clearly state the metrics used to evaluate the performance of the classifier. Explain why you chose these metrics and their relevance to the problem at hand.

6. **Results**: Present the results of your tests, including the accuracy or other relevant metrics achieved on the testing set. Provide insights into the performance of the classifier and any observations made during evaluation.

## Receptive Field

Discuss the overall receptive field of your model. Explain how receptive field size impacts the model's ability to capture global and local information in the images. Provide examples to demonstrate the receptive field at different stages of the model.

Explain how the receptive field can be increased or decreased based on the model architecture or modifications.

## FLOPS and MACCs Calculation

Report the estimated or calculated number of FLOPS (floating-point operations) and MACCs (multiply-accumulate operations) per layer in your model, focusing mainly on the convolutional and fully connected layers. Highlight the most computationally expensive layers in terms of FLOPS and MACCs.

Discuss strategies to decrease the number of FLOPS and MACCs in the model, such as model pruning, quantization, or using alternative lightweight architectures. Provide examples to support your recommendations.

## Repository Structure

Describe the structure of your repository, including the organization of code files, data directories, and any additional resources. Provide instructions for running the code, reproducing the results, and any dependencies or package versions required.

## Conclusion

Summarize your approach, key findings, and any limitations or challenges faced during the development of the clothing classifier. Discuss possible future improvements or extensions to the model or evaluation process.

Feel free to reach out if you have any questions or suggestions!

