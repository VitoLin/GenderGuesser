### Preprocessing
#### Overview
Since we using deep learning as part of our project we do not need much preprocessing. The features will be extracted as part of our machine learning algorithm.

The preprocessing we will use comes in as filtering out certain data we want to use as part of our training or testing sets. This can something directly applicable such as percieved sex ie. male or female. This can also be something to limit the type of information being fed in such as excluding images with hats or glasses or things that would cover the face. This filtering will also be used to see if training on certain data sets and testing on another will create interesting results. For example, if we test only on celebrity images and then test on a data set of a certain ethnicity, will we get similar results to a celebrity testing set?

#### Functions
Our functions are:

##### Celebrity

###### `get_all_filter(self)`
returns list of attributes: 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', etc

###### `set_filter(self, filter_)`
sets filter to check if the encoding on a certain filter is -1 or 1

###### `__len__(self)`
returns number of rows in the df

###### `__getitem__(self, idx)`
gets the image as a tensor and gets the target as a filter

###### `filter_dataset(self, filter_)`
return a subset of data with the filtered index 
filter: Dict of column: value to filter out

##### Ethnicity
Includes the same functions as the celebrity data set but also includes

###### `get_attr_map(self)`
returns the map of the attributes and their encodings such as: the age as a range

##### Example
You can look in the [data_viz.ipynb](https://github.com/VitoLin/GenderSwap/blob/main/data_viz.ipynb) notebook for examples of the filters in use

Getting list of attributes
![alt text](https://github.com/VitoLin/GenderSwap/img/example1.png?raw=true)

Setting filter
![alt text](https://github.com/VitoLin/GenderSwap/img/example2.png?raw=true)

Getting filter results
![alt text](https://github.com/VitoLin/GenderSwap/img/example3.png?raw=true)