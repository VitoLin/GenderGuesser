### Part 3
#### Overview
Since we using deep learning as part of our project we do not need much preprocessing. The features will be extracted as part of our machine learning algorithm. Hence we also perform an embedding extractiong on 2 pretrained models - InceptionResnet w/ VGGFace2 or Casia-webface.


#### Preprocessing
The limited preprocessing we will use comes in as filtering out certain data we want to use as part of our training or testing sets. This can something directly applicable such as percieved sex ie. male or female. This can also be something to limit the type of information being fed in such as excluding images with hats or glasses or things that would cover the face. This filtering will also be used to see if training on certain data sets and testing on another will create interesting results. For example, if we test only on celebrity images and then test on a data set of a certain ethnicity, will we get similar results to a celebrity testing set?

While we have implemented this filtering, we have not solidly defined exactly what we will filter out. This is because down the road, we may wish to use previously filtered out data in order to quantify our model's uncertainty.

#### Functions
We load our datasets using 2 Pytorch Dataloaders `FairFaceData` and `CelebData`. Vito Lin wrote the code for CelebData and John Lee did the work for FairFaceData (However, we largely discussed what we needed to implement for the filtering, so contribution is largely shared for both parts).
They contain the following methods:

##### Celebrity

###### `get_all_filter(self)`
returns list of attributes: 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', etc.

###### `set_filter(self, filter_)`
sets filter that determines the Truth output tensor of the dataloader. We may not want all the binary attributes. For example, we may only want truth values for a Male or Female, but not for whether they have a big nose.

###### `__len__(self)`
returns number of rows in the df

###### `__getitem__(self, idx)`
gets the image as a tensor and gets the target a tensor array of desired binaries

###### `filter_dataset(self, filter_)`
return a subset of data with the filtered index 
filter: Dict of column: value to filter out
We might wish to subset our data depending on what data to filter out. This is the key filtering step.

##### Ethnicity
Includes the same functions as the celebrity data set but also includes

###### `get_attr_map(self)`
Since the Ethnicity dataset is labeled using strings, we encode the strings from 0 - n.
returns the map of the attributes and their encodings such as: Male = 0/Female = 1

##### Running Data
We provide a sample dataset under data_sample which contains 10 images from the CelebA dataset and 
While the full dataset is currently preloaded in [data_viz.ipynb](https://github.com/VitoLin/GenderSwap/blob/main/data_viz.ipynb) notebook, to run it yourself, please change sample = True in the sample section.

Furthermore, here are some code samples bits for certain operations:

Getting list of attributes
<img src="https://github.com/VitoLin/GenderSwap/tree/main/img/example1.png" alt="example1" title="Optional title">

Setting filter
<img src="https://github.com/VitoLin/GenderSwap/tree/main/img/example2.png" alt="example2" title="Optional title">

Getting filter results
<img src="https://github.com/VitoLin/GenderSwap/tree/main/img/example3.png" alt="example3" title="Optional title">

#### Embedding extraction

Since our project relies upon deep learning techniques to implement feature extraction, we supplement the preprocessing step with an embedding extraction of our face data on 1 model (InceptionResNet) pretrained using 2 different datasets (VGGFace2 and Casia-webface). John Lee implemented the VGGFace2 code and Vito Lin completed it by implementing the Casia-webface code (However, we largely worked togethor to figure out the extraction. Hence "implement" just means wrote the code.).  

Our evaluation was done by first splitting the FairFace dataset into Male/Female. For each sex, we ran about 2000 images through the model, extracting a 512 dimensional vector of embeddings. For the males, we computed the centroid of the feature space. Then for both females and males, we computed the euclidian distance for each sample from the male_centroid and plotted a histogram. In an ideal classification, we would see the euclidian distance for males be quite small, whereas for the females, they would be large enough to generate a seperate histogram. However, since the pre-trained models we used were for the task of facial recognition (which is an orthogonal task to gender classification since they group togethor characteristics that distinguish genders) the histograms for female and male euclidian distances roughly overlap. This overlap will serve as the baseline for future models. 

The reason this step was important was to have access to the embeddings. For work down the road, we could either finetune the models for gender classification so the euclidian distances actually construct two non-overalapping histograms. Alternatively, we could train a new classifier uppon the embeddings of the model to seperate the two histograms.



##### Running Pretrained models
To run the pretrained models yourself, you can find a notebook [pretrained_model.ipynb](https://github.com/VitoLin/GenderSwap/blob/main/pretrained_model.ipynb) to do so by setting sample=True in the notebook. The current loaded information on github is the complete run, and the corresponding plots can be found below. However, if you run using sample mode, the final histograms will be completely off due to a lack of data. Running on sample mode is merely for testing the code, rather than actual data.

##### Plots
vggface2-pretrained
<img src="https://github.com/VitoLin/GenderSwap/tree/main/img/vggface2.png" alt = "vggface2" title="Optional title">

casia-webpage-pretrained
<img src="https://github.com/VitoLin/GenderSwap/tree/main/img/casia-webpage.png" alt = "casia-webpage" title="Optional title">