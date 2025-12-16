# DL
Deep learning experiments for face analysis tasks (verification, recognition, and attribute classification). Contains full training and evaluation pipelines.





# 🏗️ VGG19 Transfer Learning for Celebrity Classification

This documentation outlines the process of adapting the pre-trained VGG19 model for our specific image classification task (31 classes of famous personalities) using Transfer Learning.

---

## 1. Objective and Methodology

* **Goal:** To leverage the powerful feature extraction capabilities of a VGG19 model pre-trained on the vast ImageNet dataset and transfer this knowledge to our smaller, specific dataset.
* **Methodology:** We utilize **Transfer Learning** through a technique called **Feature Extraction**. This approach trains only the newly added top layers, keeping the pre-trained foundational layers fixed.

---

## 2. Architectural Structure (Feature Extraction)

The model is divided into two main, independently treated components:

| Component | Description | Action Taken |
| :--- | :--- | :--- |
| **Feature Extractor** (`VGG19.features`) | The convolutional base (Conv Blocks 1-5) responsible for learning hierarchical visual features (edges, textures, shapes). | **Freezing:** The weights of these layers are frozen (`param.requires_grad = False`). |
| **Classifier** (`VGG19.classifier`) | The final fully connected layers responsible for making the class prediction. | **Replacement & Training:** The original classifier is discarded and replaced with a new set of layers optimized for 31 classes. |

### New Classifier Architecture

The replacement classifier structure is designed to adapt the extracted features to the new set of 31 classes, incorporating Dropout for regularization:

$$\text{Input} \xrightarrow{\text{Flatten}} (512 \times 7 \times 7) \xrightarrow{\text{FC}_1} 4096 \xrightarrow{\text{ReLU} + \text{Dropout}(0.5)} 4096 \xrightarrow{\text{FC}_2} 4096 \xrightarrow{\text{ReLU} + \text{Dropout}(0.5)} 4096 \xrightarrow{\text{FC}_3} 31$$

---

## 3. Fine-Tuning Implementation Steps

The core steps in the PyTorch code ensure that only the new top layers are trained:

| Step | Code Implementation | Rationale |
| :--- | :--- | :--- |
| **1. Load Pre-Trained Model** | `vgg19_tl = models.vgg19(pretrained=True)` | Loads the VGG19 architecture initialized with weights learned from ImageNet. |
| **2. Freeze Base Layers** | `for param in vgg19_tl.features.parameters(): param.requires_grad = False` | Prevents the backpropagation algorithm from updating the weights in the feature extraction layers, preserving the generic learned features. |
| **3. Replace Classifier** | `vgg19_tl.classifier = nn.Sequential(...)` | The final layer is set to output 31 values, matching our specific task. |
| **4. Define Optimizer** | `optimizer = torch.optim.Adam(vgg19_tl.classifier.parameters(), lr=1e-4)` | Crucially, the optimizer is instructed to only update the parameters of the newly added `.classifier` layers. |

---

## 4. Training Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | VGG19 | The underlying CNN architecture. |
| **Classes** | 31 | The number of output categories (celebrities). |
| **Optimizer** | Adam | The optimization algorithm used for weight updates. |
| **Learning Rate** | $1e-4$ | The step size for weight updates. |
| **Loss Function** | CrossEntropyLoss | Standard loss function for multi-class classification. |
| **Epochs** | 10 | The total number of full passes over the training dataset. |
| **Device** | CPU (used in the script) | The processing unit used for training. |

---

## 5. Performance Metrics (Snapshot)

The use of Transfer Learning allowed for rapid and effective convergence:

| Metric | Value (After 10 Epochs) | Interpretation |
| :--- | :--- | :--- |
| **Validation Accuracy (Val Acc)** | $\approx 82.72\%$ | The percentage of correct classifications on the unseen validation dataset. |
| **AUC (Area Under the Curve)** | $0.9955$ | Excellent value, indicating a strong capability to distinguish between the 31 classes. |
| **Performance Trend** | Improved from $37\%$ (Epoch 1) to $82\%$ (Epoch 10) | Demonstrates the high efficiency and fast convergence achieved by leveraging pre-trained weights. |


### 5.1. Visualizations

The following graphs illustrate the model's performance and training stability:

#### Accuracy and Loss Trends
Shows strong convergence over 10 epochs.

Accuracy Curve
![<img width="662" height="505" alt="Accuracy" src="https://github.com/user-attachments/assets/2e131cc2-73e1-4818-876c-182fc08c5afe" />
]

Loss Curve
![<img width="687" height="512" alt="loss" src="https://github.com/user-attachments/assets/5dee80c8-3579-40c5-ad97-e64fc7123417" />
]

#### Detailed Classification Metrics
Provides a deeper look into the model's predictive power across the 31 classes.

Confusion Matrix
![<img width="847" height="769" alt="ConfusionMatrix" src="https://github.com/user-attachments/assets/6cb3c9cd-d5dd-4447-b8ee-9e38eecb4b51" />
]

Multi-Class ROC Curve
![<img width="825" height="658" alt="RocCurve" src="https://github.com/user-attachments/assets/5b5de693-d3d1-4911-96b1-4c403bed7fef" />
]
