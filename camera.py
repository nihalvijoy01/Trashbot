import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_and_prepare_image(img, target_size=(100, 100)):
    """
    Prepare an image for prediction.
    Args:
    img (array): Image array (captured from the camera).
    target_size (tuple): The target size of the image input for the model.
    
    Returns:
    img_array (numpy array): Processed image array.
    """
    # Resize the image to the required input size
    img = cv2.resize(img, target_size)
    # Convert the image to array format
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model input
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image array (if required by your model, assuming here it's needed)
    img_array /= 255.0
    return img_array

def classify_image(model, img_array):
    """
    Classify the image using the loaded model.
    Args:
    model (Keras Model): The pre-loaded model.
    img_array (numpy array): The prepared image for classification.
    
    Returns:
    str: Predicted category.
    """
    # Predicting the class
    prediction = model.predict(img_array)
    # Assuming the output is a softmax layer, get the index of the highest probability
    class_index = np.argmax(prediction, axis=1)
    # Map index to class labels as per your training
    classes = ['paper', 'plastic', 'metal']
    return classes[class_index[0]]

def main():
    # Load the pre-trained model
    model = load_model('entharo/trashbot.h5')
    
    # Start the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('Press "c" to classify, "q" to quit', frame)

        # Press 'c' for capture and classify
        if cv2.waitKey(1) == ord('c'):
            # Prepare the image
            img_array = load_and_prepare_image(frame)
            
            # Classify the image
            prediction = classify_image(model, img_array)
            
            # Print the prediction
            print(f"The image is classified as: {prediction}")

        # Press 'q' to exit
        elif cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
