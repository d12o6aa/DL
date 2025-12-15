# DL
Deep learning experiments for face analysis tasks (verification, recognition, and attribute classification). Contains full training and evaluation pipelines.
This project focuses on face classification, where the goal is to identify a person from an input facial image. The task is formulated as a multi-class classification problem with 31 classes, each representing a different individual.

To study the impact of different deep learning architectures, four well-known models were selected and evaluated:

VGG-19 (from scratch)

ResNet-50 (Transfer Learning)

Inception V1 (Transfer Learning)

MobileNet (Transfer Learning)

These architectures were chosen because they represent different design philosophies in convolutional neural networks, ranging from very deep sequential models to lightweight and efficient networks.

ResNet Model Architecture
1.Base Network: ResNet-50
The backbone of the model is ResNet-50, a 50-layer deep convolutional neural network introduced by He et al. (2016). ResNet introduces residual connections, allowing gradients to flow directly through identity shortcuts and mitigating the vanishing gradient problem in deep networks.
Key components of ResNet-50 include:
    • Convolutional layers with Batch Normalization and ReLU activation
    • Residual blocks with skip connections
    • Global Average Pooling
    • Fully Connected (FC) classification layer
2.Transfer Learning Strategy
To adapt ResNet-50 to the face recognition task:
    1. Pre-trained weights from ImageNet (IMAGENET1K_V1) were loaded.
    2. All convolutional layers were frozen, preventing their weights from being updated during training.
    3. The final fully connected layer was replaced to match the number of target classes.
Original FC layer: 2048 → 1000
Modified FC layer: 2048 → 31
Only the parameters of the new classification layer were trained, significantly reducing computational cost and overfitting risk.
3.Final Architecture Flow
Input Image (RGB)
↓
Convolution + Residual Blocks (Frozen ResNet-50)
↓
Global Average Pooling
↓
Fully Connected Layer (31 outputs)
↓
Softmax → Class Probabilities

4.Training Configuration
 Hardware
    • Device: CPU
 Optimization Setup
    • Loss Function: Cross-Entropy Loss
    • Optimizer: Adam
    • Learning Rate: 1e-4
    • Epochs: 10
Only the parameters of the final fully connected layer were optimized.

5. Evaluation Metrics
To comprehensively evaluate model performance, the following metrics were computed:
    • Accuracy: Overall classification correctness
    • Precision (Weighted): Class-wise correctness of positive predictions
    • Recall (Weighted): Ability to correctly identify samples from each class
    • F1-score (Weighted): Harmonic mean of precision and recall
    • Confusion Matrix: Visualization of class-wise prediction errors
    • ROC Curve & AUC (One-vs-Rest): Discriminative ability across all classes

6. Experimental Results
6.1 Training and Validation Performance
Across 10 epochs, the model showed steady improvement:
    • Final Training Accuracy: 64.76%
    • Final Validation Accuracy: 68.28%
The validation accuracy consistently tracked training accuracy, indicating good generalization and limited overfitting.
6.2 Confusion Matrix Analysis
The confusion matrix reveals that:
    • Many classes achieved strong diagonal dominance (high correct classification rates).
    • Misclassifications often occurred between visually similar individuals.
This behavior is expected in face recognition tasks without extensive fine-tuning of deeper layers.
6.3 ROC Curve and AUC
A multi-class ROC analysis using a One-vs-Rest strategy produced:
    • Overall AUC: 0.98
This high AUC score demonstrates excellent separability between classes, even when classification accuracy is moderate.

7. Learning Curves
    • Accuracy Curve: Shows consistent improvement across epochs for both training and validation.
    • Loss Curve: Indicates stable convergence without oscillation or divergence.
These curves confirm that the optimization process was stable and effective.

8.Strengths
    • Effective use of transfer learning
    • High AUC indicating strong class separability
    • Stable training with minimal overfitting
    • Computationally efficient due to frozen layers
9.Limitations
    • Training performed on CPU only
    • Feature extractor not fine-tuned
    • Performance limited by dataset size and class imbalance
