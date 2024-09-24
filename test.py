from keras.models import load_model

try:
    # Load the pre-trained model
    model = load_model('traffic_classifier.h5')

    # Print the summary of the model architecture
    model.summary()
except Exception as e:
    print("Error loading the model:", e)
