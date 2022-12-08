import cv2
import torch
from celeb_classifier import MLPClassifierWrapper
from src.models import get_pretrained_model, transformation

from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

celeb_model = MLPClassifierWrapper.load_from_checkpoint('../results/celeb_results_vggface2/lightning_logs/version_0/checkpoints/epoch=3-val_loss=0.05.ckpt')
celeb_model.trained_model = get_pretrained_model('vggface2', device)
celeb_model.to(device)
celeb_model.eval()

fface_model = MLPClassifierWrapper.load_from_checkpoint('../results/fface_results_vggface2/lightning_logs/version_0/checkpoints/epoch=1-val_loss=0.19.ckpt')
fface_model.trained_model = get_pretrained_model('vggface2', device)
fface_model.to(device)
fface_model.eval()  


def predictCeleb(img, model):
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    with torch.no_grad():
        # Transform
        input = transformation('vggface2')(img)

        # Predict
        input.to(device)
        output = celeb_model(input.unsqueeze(0))
        output = round(output.item())

    return output

def predictFface(img, model):
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    with torch.no_grad():
        # Transform
        input = transformation('vggface2')(img)

        # Predict
        input.to(device)
        output = fface_model(input.unsqueeze(0))
        output = round(output.item())

    return output


cap = cv2.VideoCapture(0)

padding = 50

while True:
    # Read the frame
    _, img = cap.read()
    img = cv2.flip(img, 1)

    # Detect the faces
    faces = face_cascade.detectMultiScale(img, 1.1, 4, minSize=(150,150))

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        
        # Padding
        if not(x - padding < 0 or y - padding < 0 or x + w + padding > img.shape[1] or y + h + padding > img.shape[0]):
            x = x - padding
            y = y - padding
            w = w + padding * 2
            h = h + padding * 2

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = img[y:y+h, x:x+w]

        # Predict
        textCeleb = "female"
        if predictCeleb(face, celeb_model) == 1:
            textCeleb = "male"
        
        textFface = "female"
        if predictFface(face, fface_model) == 1:
            textFface = "male"

        # Display text
        cv2.putText(img, "Celeb", org = (x,y-30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 0), thickness = 2)
        cv2.putText(img, textCeleb, org = (x,y), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 0), thickness = 2)

        cv2.putText(img, "Fface", org = (x+w-100,y-30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 0, 255), thickness = 2)
        cv2.putText(img, textFface, org = (x+w-100,y), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 0, 255), thickness = 2)



    # Display
    cv2.imshow('img', img)
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()