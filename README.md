# Real-Time Traffic Sign Analysis

## Project Overview
The Real-Time Traffic Sign Analysis project aims to develop a traffic sign classification system that identifies and translates traffic signs in real-time. This application enhances road safety by providing drivers with immediate recognition and understanding of road signs.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- Real-time detection and classification of traffic signs using a convolutional neural network (CNN).
- User-friendly graphical interface for easy image uploads and sign classification.
- Multilingual support for sign translations using the Google Translator API.
- Visualization of accuracy and loss metrics during model training.

## Technologies Used
- **Programming Languages**: Python
- **Libraries/Frameworks**: Keras, TensorFlow, OpenCV, Matplotlib, Pandas, Tkinter, Google Translator API
- **Model Architecture**: Convolutional Neural Network (CNN)


## Dataset

### Step 1: Download the Dataset
You can download the required dataset from the Google Drive link below:

[Download the dataset from Google Drive](https://drive.google.com/drive/folders/15wMqDhP7fknfp1XK2Rk8tLId9WpxGfPi?usp=drive_link)

### Step 2: Extract the Dataset
After downloading, extract the files and place them in the `data/` folder in your project directory. 

- **Folder structure**:
    - `Meta/`
    - `Train/`
    - `Test/`
  
### Step 3: Verify Dataset Placement
Ensure that the extracted files are placed correctly:
```plaintext
your-project-folder/
    ├── Meta/
    ├── Train/
    ├── Test/
    ├── your-code-files/
```
⚙️ Installation
Follow the steps below to set up and run the project:

1. Clone the Repository
```
bash
git clone https://github.com/yourusername/Real-Time-Traffic-Sign-Analysis.git
```
2. Navigate to the Project Directory
```
bash
cd Real-Time-Traffic-Sign-Analysis
```
4. (Optional) Create and Activate a Virtual Environment

```bash
python -m venv env
env\Scripts\activate    # On Windows
# source env/bin/activate   # On macOS/Linux
```
4. Install Dependencies

```
bash
pip install -r requirements.txt
```
Note: If requirements.txt doesn't exist, manually install:
```
bash
pip install tensorflow keras scikit-learn matplotlib pandas pillow opencv-python tk googletrans==4.0.0-rc1
```
5. Train the Model
```
bash
python traffic_sign.py
```
This will take a few minutes depending on your machine.

6. Run the Application
```
bash
python fgui.py
```
