# Model Justification
### We are using pretrained models in order to classifly and identify male or female faces.
Our Model
-
    - Model MLP
        - 512 (for vggface2, casia) or 4096 (for vgg16) input Neurons
        - 1000 Hidden Layer
        - 1000 Hidden Layer
        - 1 Output Layer
        - ReLU internal activation, Sigmoid output activation.
    - Trained using LR reduction on Plateau and Early Stopping to avoid over training
    - Model with lowest validation loss is saved.
    
We decided that the MLP model should be good for classification because we are using a transfer learning technique. It's the easiest thing to link up. Also, 2 hidden layers should be enough, as long as the size is sufficient. (We do have to be wary of overfitting, but we handle this by saving the point with lowest validation loss.)

https://hackmd.io/VBzas-lZRfCWEQVNPkBKaA#By using a pretrained MLP model we were able to have the sophisication of features extracted by larger models that have millions of training points and weights without doing the training, data gathering, and preprocessing ourselves. With training on our basic data set alone, we were unlikely to get the accuracy we achieved. 

We used vggface2, casia, and vgg16 pretrained models for our results.

vggface2 has 3.31 million face images with "large variations in pose, age, illumination, ethnicity and profession."

Casia-Webface has 500,000 face images.

VGG-16 has 14 million images from 22,000 categories.

To curate, download, store, process, and then train this much data would be insurmountable for this project. But, with these pretrained models we are able to get higher levels of accuracy without repeating the same work or incuring the same costs.




# Classification Accuracy
Our training and validation accuracies are below. Our goal was to calculate the percentage of correct predictions we made.

| VGGface2    | First Calculated Epoch | Final Epoch |
| ----------- | ---------------------- | ----------- |
| train\_loss | 0.9921875              | 0.0200529   |
| train\_acc  | 93.75%                 | 99.22%      |
| val\_loss   | 0.9736601              | 0.9736601   |
| val\_acc    | 90.27%                 | 90.27%      |

![](https://i.imgur.com/CgQtVUX.jpg)

| Casia       | First Calculated Epoch | Final Epoch |
| ----------- | ---------------------- | ----------- |
| train\_loss | 0.0809043              | 0.0040720   |
| train\_acc  | 96.09%                 | 100.00%     |
| val\_loss   | 0.0865717              | 0.1616313   |
| val\_acc    | 96.92%                 | 97.14%      |

![](https://i.imgur.com/gEmqyq7.jpg)

| VGG16       | First Calculated Epoch | Final Epoch |
| ----------- | ---------------------- | ----------- |
| train\_loss | 39.8437500             | 42.9687500  |
| train\_acc  | 60.16%                 | 57.03%      |
| val\_loss   | 38.7486801             | 38.7486801  |
| val\_acc    | 61.25%                 | 61.25%      |

![](https://i.imgur.com/kYjSfF7.jpg)

From Casia we can see that we achieved extremely high validation accuracy at the final epoch at 97.14%. This was the highest from the three models. VGGface2 came in second at 90% accuracy.

The interesting one was VGG16-Imagenet. The other pretrained models were only trained on facial data, which would make sense why they ranked much, much better than VGG16. The high loss calculated demonstrates how poorly the pretrained model fitted our data. This would make sense however, since VGG16-Imagenet is meant to be a larger more cohesive model for multiple different categories of images.


# Improvements
Some improvements we could have added were:
- We could also include Regularization layers like Dropout or use L2/L1. We used Sigmoid to get a 0 - 1 probability.
- We could have also have found better data sets out there with more pictures. We did have a large data set but they did lack some ethnic diversity along with being of only celebrities and the biases that come with only sampling stars.

We also tried taking VGG16 imagenet and removing the final fully connected layers to fine tune the model for our specific use case. We ended up trying this out but did not get the results we desired. This was mostly due to the prior mentioned fact that it includes images from thousands of different categories (22,000) and not just facial images.


# Sample
To run a sample of the code that we have discussed, you can look into:
```
Dir Structure
    SexSwap
        classification
            Results.ipynb    
```
Make sure to follow the directions in the main README.md to set up conda and the other tools you need to run this sample code.

Embeddings are generated using embedding_gen.ipynb from the data to preprocess for faster training