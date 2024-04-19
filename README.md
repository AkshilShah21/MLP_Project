An Architecture Combining Convolutional Neural Network (CNN) and Linear Support Vector Machine (SVM) for Image Classification
===

*This project was inspired based of research paper[arXiv.org](https://arxiv.org/abs/1712.03541). and the project code [github](https://github.com/AFAgarap/cnn-svm)*

## Abstract

The aim of the study is to explore the effectiveness of replacing the Softmax function with a linear SVM in a CNN architecture for image classification and to compare the performance of the CNN-SVM model against the traditional CNN-Softmax model on three different datasets. In this report, we aim to replicate the methodology and results presented in the research paper titled "An Architecture Combining Convolutional Neural Network (CNN) and Support Vector Machine (SVM) for Image Classification." Furthermore, we will extend the study by applying the same methodology and parameters to new datasets.
Empirical findings reveal that the CNN-SVM model achieved a test accuracy of approximately 99.99% using the MNIST dataset. In comparison, the CNN-Softmax model attained a slightly higher test accuracy of approximately 99.83% on the same dataset. Both models underwent testing on the Fashion-MNIST dataset, known for its increased complexity compared to MNIST. Here, the CNN-SVM model demonstrated a test accuracy of around 99.96%, while the CNN-Softmax model achieved approximately 98.83% accuracy.

## Usage

First, clone the project.
```bash
git clone https://github.com/AkshilShah21/MLP_Project.git
```

Make sure you have all the required packages mentioned in requirement.txt. If not, then Install it. 

Then, go to the repository's directory, and run the `main.py` 
```bash
cd MLP_Project
python3 main.py 
```

Subsequently, it will ask
```bash
Choose the dataset:
1. MINST
2. Fashion-MINST
3. Dogs vs cat
Enter your choice (1/2/3):
```

After the selected dataset, Both models are trained on the particular dataset. 

On completion of excution  generate the result into logs and figures

## Results

The hyperparameters used in this project were manually assigned, and not through optimization.

|Hyperparameters|CNN-Softmax|CNN-SVM|
|---------------|-----------|-------|
|Batch size|128|128|
|Learning rate|1e-3|1e-3|
|Steps|100|100|
|SVM C|N/A|1|

**The experiments were conducted on an Apple Mac M1 pro chip, 32 GB RAM and Apple Metal 14-core GPU**

#### Test Accuracy Comparison

| Model             | MNIST    | Fashion-MNIST | cat vs dog |
| :---------------- | :------: | :------------:|:----------:|
| CNN-Softmax       |  99.99%  |     99.96%    |   99.96%   |
| CNN-SVM           |  99.83%  |     98.83%    |   99.99 %  |


#### Test loss Comparison

| Model             | MNIST    | Fashion-MNIST | cat vs dog |
| :---------------- | :------: | :------------:|:----------:|
| CNN-Softmax       |    0%    |     00.02%    |   00.02%   |
| CNN-SVM           |  90.30%  |     90.80%    |   00.10 %  |

