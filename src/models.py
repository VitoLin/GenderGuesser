from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
from torchvision import transforms
import torch.nn as nn

def transformation(model):
    ''' necessary transformations corresponding to model'''
    if model == 'vggface2' or model == 'casia':
        return transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
        ])
    elif model == 'vgg16':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"{model} is an invalid model")
    
def get_pretrained_model(model_name, device):
    if model_name == 'vggface2':
        # Define Models in embedding mode.
        return  InceptionResnetV1(pretrained = 'vggface2', device = device, classify=False).eval()
    elif model_name == 'casia':
        return InceptionResnetV1(pretrained = 'casia-webface', device = device, classify=False).eval()
    elif model_name == 'vgg16':
        # model 3 is a vgg16, ImageNet trained
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # remove all but first fully connected
        model.classifier = model.classifier = nn.Sequential(model.classifier[0])
        model = model.to(device)
        return model.eval()