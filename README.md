# Computer Vision Project: Gender Swap

Vito Lin and John Lee

## Goal
The goal of our project is to be able to take an image of a person and discern the perceived biological sex of the individual. We want to be able to increase the feminine or masculine facial traits to change the perceived sex of the person. The goal is to be able to discern features that are associated with a feminine or masculine face and then use that information to make predictions on a new face and predict the faces perceived sex. We also want to be able to use these feminine and masculine features extracted from our training to also adjust the features on a face to make it perceived as more feminine or masculine.
These features can be extracted using methods such as the one shown in class with Gabor wavelets where we get features extracted from the convolutional layers. Not using Gabor wavelets since there are better solutions available but this is just a first intuition.
We also plan on adapting an already existing application for face morphing online in order to complete the genderswap of an inputted face. Our first intuition is to recreate the face using the facial morph that was shown to us in class and then adjust features of the face until it looks more feminine or masculine. 

### Tasks:
In order to accomplish this goal we have to first gather a large dataset of faces with tags on the perceived gender of all the faces. We would want the data set to be consistent in terms of angle of face, illumination, and area that is cropped (or other image augmentations). From here we would want to feed into some machine learning algorithm these tagged photos and be able to discern the masculine and feminine features to predict the perceived gender of the face. 
    We would also want to use the Morphable Model for the Synthesis of 3D faces that was introduced in class in order to generate these new feminine or masculine faces. 
    We also want to minimize the amount of features that may be associated with a feminine or masculine face that may not be from the actual face but perhaps artifacts from the sources we gather the faces from. This would just mean screening the pictures ahead of time to see if there are artifacts that exist (maybe something as silly as the word male or female written on the photos).

### Learning:
* Learn how to use machine learning in order to categorize images.
* Learn how to use machine learning for facial recognition.
* Learn how to use facial morphing repositories that already exist and fit it with our parameters for masculinity or femininity.
* Learn how to curate good datasets for training, validation, and testing. 
* Learn about different machine learning algorithms in order to find an algorithm that creates the best results for our project.

## Datasets:
* Large data set or sets of faces with tagged perceived genders.
* Be able to preprocess these data sets to that way the are consistent to be used in our algorithm
* We want to gather a large amount of faces from different sites that hold faces and then preprocess them all to work within our model. We can then split the model into training, validation, and testing.


### Source
We will utilize the Celeb Dataset from Kaggle found here: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download.

### Description
The original paper this dataset was created for is:
S. Yang, P. Luo, C. C. Loy, and X. Tang, "From Facial Parts Responses to Face Detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015
The dataset contains 202599 facial images of celebrities, cropped and aligned. Each image comes with 40 (-1, 1) binary attribute annotations (including Gender) and 5 landmark locations (eyes, nose, left/right mouth). Of the 202599 images, there are 10,177 unique anonymous identities.

The images are taken in many different backgrounds, with large variations in lighting, and are taken at no consistent angle. Each picture has a relatively unobstructed view of the celebrity's face and are taken with no consistent camera (focal lense, quality, color, etc)

### Implementation
Since some images contain distracting annotated features such as eyewear, we will likely reconsider their usage in the dataset.
Although the dataset provides a suggested train/val/test distribution roughly approximating to 80/10/10, if we remove certain images mentioned above, we will randomly repartition the remaining images into the same ratio. Furthermore, we may wish to treat the males and females as 2 different datasets. In that case, we will certianly repartition the data.

There will be little difference between the training and validation data sets as we are randomly sampling from the same source and dividing the total dataset to a 80/10/10 ratio (training, testing, validation). We want to keep the training and validation data sets consistent as to not bias the model in any way.