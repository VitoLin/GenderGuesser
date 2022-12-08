# Dataset
Before we started out project we separated the CelebFaces dataset into train, val, and test. At the end after we had trained out models we tested the model with the hidden test data set.

We thought this was good "unknown data" since it was from the same source that testing and validation came from. This would prevent the sourcing of the images affect the prediction. This means things such as lighting, quality, angle, or partial occlusion would not affect the guess.

We also though that the differences were good enough since they were not seen by the models at all.


However, we also included more data from Fairface. From the Fairface paper it is cited that "most existing large scale face databases are biased towards “lighter skin” faces (around 80%), e.g. White, compared to “darker” faces, e.g. Black."

The goal of the Fairface Dataset is "to mitigate the race bias in the existing face datasets." The dataset is comprised of "108,501 facial images" that are made up of "7 race groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino." The data is balanced between these 7 defined groups.

We chose this dataset since the previous dataset we had trained on was the CelebFaces Attributes dataset. From a quick look, we saw that the dataset lacked the diversity that we have in the Fairface Dataset. From this, we wanted to see how our sex classification model would fair with different ethnicities. We believed that this was a good way to test ethnic bias. We trained on just the white ethnicity and then tested it with different ethnicities.

The paper can be found here: 
https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf

# Results
### Celebrity Face Attributes
This is our results from our hidden test data on CelebrityFace Attributes.

| Model    | test\_acc | val\_acc |
| -------- | --------- | -------- |
| VGGface2 | 98.77%    | 98.60%   |
| Casia    | 97.46%    | 97.23%   |
| VGG16    | 60.98%    | 61.25%   |

This ended up having very similar accuracies to the data from the best epoch's val_acc.

### FairFace
This test set is from training on only the white ethnicity from Fairface and then running on different ethnicities.

| Ethnicity       | loss         | acc    |
| --------------- | ------------ | ------ |
| White           | 0.5150375962 | 92.91% |
| Middle Eastern  | 0.6889208555 | 92.21% |
| Latino          | 0.4966644347 | 90.34% |
| Indian          | 0.5177448392 | 87.18% |
| Southeast Asian | 0.5198197961 | 83.81% |
| East Asian      | 0.5003252029 | 82.05% |
| Black           | 0.5000362396 | 75.41% |

We can see from this that the White ethnicity performed the best with the rest following. This makes sense since we trained the model purely on White ethnicity.

We know that the 80% racial bias exists from other face databases. We also see that there is a large difference in accuracies from the best performer (White) at 92.91% to the worst (Black) at 75.41%. We imagine this phenomenon probably affects other models as well. There is often comments of how certain face filters don't work on POC online but by doing this project we actually able to quantify it in the space of gender identification.

# Demo
As mentioned before, there is a slight bias on white faces in both the models. One due to the implicit 80% white bias (Celeb) in most facial databases and one due to the purposeful 100% white bias we created (Fface).

![](https://i.imgur.com/ckkTDL8.jpg)

As we tested our demo code, we found that Fface was more likely to incorrectly predict the face of our members. The face above is an Asian male and predicts to be female most of the time.

Running our demo:
To run our demo, make sure you have the source activated. Then go to the classification folder and run
```
python demo.py
```
in the directory.

A vide0 demo can be found here:
https://youtu.be/q2MieQZme-c

# Improvements
We could improve our models for this ethnic bias by using the fairface data set to train our models, including all ethnicities. But even then, we see that with this dataset we only achieved 92.91% accuracy on white faces which is still lower than our CelebA dataset with the same pretrained model.

