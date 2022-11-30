


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
    - Results found in Results.ipynb