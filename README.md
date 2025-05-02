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

[Download the dataset from Google Drive](https://drive.google.com/drive/folders/?usp=drive_link)

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

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Real-Time-Traffic-Sign-Analysis.gitw
2. Navigate to the project directory:
   ```bash
    cd Real-Time-Traffic-Sign-Analysis
3. Install the required packages
   ```bash
     pip install tensorflow keras sklearn matplotlib pandas pil
4. Run the MODEL first
   ```bash
   python traffic_sign.py

It will take some time
  
5. Run Main file Now
  ```bash
    python fgui.py
