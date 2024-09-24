import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import cv2
import pytesseract
from deep_translator import GoogleTranslator
from keras.models import load_model
import pyttsx3
import cv2
import numpy as np

# Load the trained model to classify signs
model = load_model('my_model.h5')

# Dictionary to label all traffic signs class
classes = {1: 'Speed limit (20km/h)', 2: 'Speed limit (30km/h)', 3: 'Speed limit (50km/h)', 4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)', 6: 'Speed limit (80km/h)', 7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)', 9: 'Speed limit (120km/h)', 10: 'No passing',
           11: 'No passing veh over 3.5 tons', 12: 'Right-of-way at intersection', 13: 'Priority road',
           14: 'Yield', 15: 'Stop', 16: 'No vehicles', 17: 'Veh > 3.5 tons prohibited', 18: 'No entry',
           19: 'General caution', 20: 'Dangerous curve left', 21: 'Dangerous curve right', 22: 'Double curve',
           23: 'Bumpy road', 24: 'Slippery road', 25: 'Road narrows on the right', 26: 'Road work',
           27: 'Traffic signals', 28: 'Pedestrians', 29: 'Children crossing', 30: 'Bicycles crossing',
           31: 'Beware of ice/snow', 32: 'Wild animals crossing', 33: 'End speed + passing limits',
           34: 'Turn right ahead', 35: 'Turn left ahead', 36: 'Ahead only', 37: 'Go straight or right',
           38: 'Go straight or left', 39: 'Keep right', 40: 'Keep left', 41: 'Roundabout mandatory',
           42: 'End of no passing', 43: 'End no passing veh > 3.5 tons'}


classes_with_context = {
    1: 'Speed limit (20km/h): \n\nReduce speed to 20 km/h, often seen in school zones or residential areas.',
    2: 'Speed limit (30km/h): \n\nMaintain a speed of 30 km/h, typically in areas with high pedestrian activity.',
    3: 'Speed limit (50km/h): \n\nMax speed allowed is 50 km/h, usually within city limits.',
    4: 'Speed limit (60km/h): \n\nStay under 60 km/h, often found in suburban roads or minor highways.',
    5: 'Speed limit (70km/h): \n\nLimit your speed to 70 km/h, often seen on roads approaching major highways.',
    6: 'Speed limit (80km/h): \n\nMaximum speed is 80 km/h, typically on main roads connecting cities.',
    7: 'End of speed limit (80km/h): \n\nIndicates that the 80 km/h limit no longer applies; check for new speed limits.',
    8: 'Speed limit (100km/h): \n\nYou can drive up to 100 km/h, commonly found on highways.',
    9: 'Speed limit (120km/h): \n\nDrive no faster than 120 km/h, typically on expressways or motorways.',
    10: 'No passing: \n\nOvertaking other vehicles is prohibited; stay in your lane.',
    11: 'No passing veh over 3.5 tons: \n\nHeavy vehicles over 3.5 tons are not allowed to overtake other vehicles.',
    12: 'Right-of-way at intersection: \n\nYou have priority at the upcoming intersection; other vehicles must yield.',
    13: 'Priority road: \n\nYou are on a priority road; vehicles from other roads must yield to you.',
    14: 'Yield: \n\nGive the right of way to other road users at the intersection ahead.',
    15: 'Stop: \n\nCome to a complete stop at the sign; proceed only when it is safe.',
    16: 'No vehicles: \n\nNo motor vehicles are allowed beyond this point.',
    17: 'Veh > 3.5 tons prohibited: \n\nVehicles over 3.5 tons are not allowed to enter.',
    18: 'No entry: \n\nEntry is prohibited for all vehicles from this direction.',
    19: 'General caution: \n\nBe cautious of potential hazards; slow down and be alert.',
    20: 'Dangerous curve left: \n\nThere’s a sharp curve to the left ahead; slow down and take caution.',
    21: 'Dangerous curve right: \n\nThere’s a sharp curve to the right ahead; slow down and be cautious.',
    22: 'Double curve: \n\nTwo dangerous curves ahead, first to the left then to the right; reduce speed.',
    23: 'Bumpy road: \n\nThe road surface is uneven ahead; drive slowly to avoid damage.',
    24: 'Slippery road: \n\nThe road ahead may be slippery due to weather conditions; reduce speed.',
    25: 'Road narrows on the right: \n\nThe right lane is narrowing; move left if needed.',
    26: 'Road work: \n\nRoad construction ahead; slow down and watch for workers or machinery.',
    27: 'Traffic signals: \n\nTraffic lights are ahead; be prepared to stop.',
    28: 'Pedestrians: \n\nWatch out for pedestrians crossing the road; slow down and yield if necessary.',
    29: 'Children crossing: \n\nChildren may be crossing the road ahead, usually near schools; reduce speed.',
    30: 'Bicycles crossing: \n\nCyclists may be crossing or sharing the road; drive carefully.',
    31: 'Beware of ice/snow: \n\nRoad may be icy or covered in snow; reduce speed and drive cautiously.',
    32: 'Wild animals crossing: \n\nBe alert for animals crossing the road, especially in rural areas.',
    33: 'End speed + passing limits: \n\nThe previous speed and passing restrictions no longer apply; follow standard rules.',
    34: 'Turn right ahead: \n\nA mandatory right turn is coming up; get into the correct lane.',
    35: 'Turn left ahead: \n\nA mandatory left turn is ahead; prepare to turn.',
    36: 'Ahead only: \n\nYou are only allowed to go straight ahead; no turns allowed.',
    37: 'Go straight or right: \n\nYou can either continue straight or take a right turn at the next junction.',
    38: 'Go straight or left: \n\nYou can either go straight or turn left at the next intersection.',
    39: 'Keep right: \n\nStay to the right side of the road or lane divider.',
    40: 'Keep left: \n\nKeep to the left, often seen on divided roads or near obstacles.',
    41: 'Roundabout mandatory: \n\nYou must enter the roundabout ahead; give way to traffic already in the circle.',
    42: 'End of no passing: \n\nThe no-passing zone is ending; overtaking is now allowed.',
    43: 'End no passing veh > 3.5 tons: \n\nEnd of no-passing restrictions for vehicles over 3.5 tons.'
}

# Initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    # Convert the image to RGB format if it's not already in that format
    image = image.convert('RGB')
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)

    # Make predictions
    predictions = model.predict(image)

    # Get the class with the highest probability for each prediction
    predicted_classes = numpy.argmax(predictions, axis=1)

    sign = classes[predicted_classes[0] + 1]
    sign_context = classes_with_context[predicted_classes[0]+1]
    # Translate the sign text according to the selected language
    target_language = lang_option.get()
    if target_language != "english":
        try:
            sign = GoogleTranslator(source='english', target=target_language).translate(sign)
            sign_context = GoogleTranslator(source='english', target=target_language).translate(sign_context)
        except Exception as e:
            print("Translation error:", e)
    speak_text(sign)
# Use different fonts and add a newline for separation
    label.configure(foreground='#011638', font=('arial', 15, 'bold'), text=sign + "\n\n")  # Sign text in bold
    label.configure(foreground='#011638', font=('arial', 12, 'italic'), text=sign_context)  # Context text in italic


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def select_language(text):
    global lang_option
    lang_option = tk.StringVar(top)
    lang_option.set("english")  # default value
    lang_menu = OptionMenu(top, lang_option, "english", "spanish", "french", "german", "italian")
    lang_menu.place(relx=0.79, rely=0.53)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
        select_language(file_path)  # Call select_language to display language options after selecting an image
    except Exception as e:
        print(e)

# Voice 
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# In the classify function, after the sign is recognized:

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Know Your Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()