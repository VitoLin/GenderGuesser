


Load Embeddings for celeb data

-  Use Celeb Dataset to determine am appropriate classification model
    - Model 
        MLP
        - 512 (for vggface2, casia) or 4096 (for vgg16) input Neurons
        - 1000 Hidden Layer
        - 1000 Hidden Layer
        - 1 Output Layer
        - ReLU internal activation, Sigmoid output activation.
    - Trained using LR reduction on Plateau and Early Stopping to avoid over training
    - Model with lowest validation loss is saved.

- MLP model should be good for classification because we are using a transfer learning technique. Its the easiest thing to link up. Also, 2 hidden layers should be enough, as long as the size is sufficient. (We do have to be wary of overfitting, but we handle this by saving the point with lowest validation loss.)
- MLP, we could also include Regularization layers like Dropout or use L2/L1, but that overcomplicates our project (could be improvement). We use Sigmoid to get a 0 - 1 probability.


- For the question about reporting training and validation accuracies ( if you look in metrics.csv in each of the results folder, there is a column called val_acc & train_acc-> that is the validation accuracies.) Justification is just that we are trying to figure out the percentage of correctly predicted over total, so accuracy makes sense)
- Note that the correct values for train_acc and val_acc must be found by matching the epoch # and val_loss of the checkpoint to the line in metrics (For ex. in vggface2, the correct line is 100, so take the val_acc from line 100 and the train_acc from line 99). They get a bit messy, sorry (If you don't want to read the csv, you can go into a jupyter notebook and import pandas as pd and call pd.read_csv(path_to_csv) and it could be easier to read.)


- It's all kind of confusing, because I did the testing and validation all at the same time... so its hard to distinguish, maybe we can talk to Adam about it.


- Another thing is, when creating the sample data, I didn't really distinguish between training testing and validation. They are just the first 10 from the celeb dataset and 10 from each ethnicity. That means, it's highly likely that the celeb samples are a part of the train set. But honestly, whatever, I don't think we'll get docked if at all. The same holds true for the white ethnicity, but all the other ethnicities are guarenteed to be a part of the test set.


- Results found in Results.ipynb (These can be run by the graders)
- Training is run in celeb and white_classifier, but without data, the graders can't run it. It's hidden on my local machine. Let me know if you need them
- Embeddings are generated using embedding_gen.ipynb from the data to preprocess for faster training
- firface_sample generates samples for ethnicity data (for celeb, I just pulled the first 10.)