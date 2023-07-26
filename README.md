# CNN-Image-Classification-and-Denoising
In this deep learning project, we explore various techniques to handle noisy images and perform image classification. We start by loading and normalizing the MNIST Fashion dataset and add noise to both training and test images. A CNN classifier model is defined and trained on clean images to achieve high accuracy in classifying fashion items. We evaluate the classifier on clean and noisy test images to observe the effect of noise on classification performance.

To address the issue of noisy images, we design an autoencoder model that learns to denoise the input images. The autoencoder is trained using noisy images, and we evaluate its performance on denoising the noisy test images. We plot examples of the noisy and denoised images side-by-side to observe the denoising capabilities of the autoencoder.

Furthermore, we construct an end-to-end denoising-classifier network that combines the denoising autoencoder and the image classifier. This network simultaneously denoises the noisy images and performs image classification. We evaluate its performance on the test data and report the accuracy and loss of the combined tasks.

Here's how we can break this for easier interpretation:

Deliverable 1: ROC Analysis for Traffic Sign Detector
The project begins with loading the ground-truth values and detector outputs. The ROC curve is then computed and plotted to evaluate the traffic sign detector's performance.

Deliverable 2: Autoencoder Denoising for Fashion MNIST Classification
The project proceeds with the MNIST fashion dataset, where images are loaded, normalized, and subjected to simulated noise. A CNN classifier is created and trained using clean images. The classifier's accuracy on clean test images is reported. Subsequently, the classifier is evaluated on noisy test images, revealing the impact of noise on classification performance.

Deliverable 3: Autoencoder Denoising and Classification
A CNN-based autoencoder model is defined to denoise noisy images. The autoencoder is trained using noisy images and the denoising effectiveness is demonstrated by denoising and plotting noisy test images. The classifier's accuracy on autoencoder denoised test images is reported.

Deliverable 4: End-to-End Denoising-Classification Network 
An advanced denoising-classification model is constructed, combining an encoder for feature extraction, two branches for classification and denoising, and a decoder for image reconstruction. The model is trained end-to-end using noisy images, and classification accuracy on noisy test images is presented.

Overall, this project showcases the power of deep learning in handling noisy data and demonstrates the effectiveness of CNNs for image classification and denoising tasks.
