import cv2
import torch
from celeb_classifier import MLPClassifierWrapper
from src.models import get_pretrained_model, transformation

from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vggface_model = MLPClassifierWrapper.load_from_checkpoint('../results/celeb_results_vggface2/lightning_logs/version_0/checkpoints/epoch=3-val_loss=0.05.ckpt')
vggface_model.trained_model = get_pretrained_model('vggface2', device)
vggface_model.to(device)
vggface_model.eval()

def predict(img, model):
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    with torch.no_grad():
        # Transform
        input = transformation('vggface2')(img)

        # Predict
        input.to(device)
        output = vggface_model(input.unsqueeze(0))
        output = round(output.item())

    return output


cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    img = cv2.flip(img, 1)

    # Detect the faces
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = img[y:y+h, x:x+w]

        # Predict
        text = "female"
        if predict(face, vggface_model) == 1:
            text = "male"

        # Display text
        cv2.putText(img, text, org = (x,y), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 0), thickness = 2)
        

    # Display
    cv2.imshow('img', img)
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()