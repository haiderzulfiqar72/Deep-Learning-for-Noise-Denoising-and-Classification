# CNN-Image-Classification-and-Denoising
In this deep learning project, we explore various techniques to handle noisy images and perform image classification. We start by loading and normalizing the MNIST Fashion dataset and add noise to both training and test images. A CNN classifier model is defined and trained on clean images to achieve high accuracy in classifying fashion items. We evaluate the classifier on clean and noisy test images to observe the effect of noise on classification performance.

To address the issue of noisy images, we design an autoencoder model that learns to denoise the input images. The autoencoder is trained using noisy images, and we evaluate its performance on denoising the noisy test images. We plot examples of the noisy and denoised images side-by-side to observe the denoising capabilities of the autoencoder.

Furthermore, we construct an end-to-end denoising-classifier network that combines the denoising autoencoder and the image classifier. This network simultaneously denoises the noisy images and performs image classification. We evaluate its performance on the test data and report the accuracy and loss of the combined tasks.

Overall, this project showcases the power of deep learning in handling noisy data and demonstrates the effectiveness of CNNs for image classification and denoising tasks.
