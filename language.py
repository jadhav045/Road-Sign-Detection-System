import cv2
import pytesseract
# from deep_translator import GoogleTranslator
# from deep_translator import single_detection
from deep_translator import GoogleTranslator, single_detection
from cryptography.fernet import Fernet

# Load the previously generated key
with open("secret.key", "rb") as key_file:
    key = key_file.read()

cipher_suite = Fernet(key)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("SpanishTextRecognition.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(pytesseract.image_to_string(img))
boxes = pytesseract.image_to_data(img)
#hImg, wImg = img.shape
hImg, wImg, _ = img.shape


#print(boxes)
for x, b in enumerate(boxes.splitlines()):
    if x != 0:
        b = b.split()
        # print(b)
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 1)
            text = b[11].encode()  # Encrypt the text before translation
            encrypted_text = cipher_suite.encrypt(text)
            try:
                # lang = single_detection(encrypted_text.decode())  # Decrypt before language detection
                lang = single_detection(encrypted_text.decode(), api_key='64a6937ed27d4e100736152b81596a10')

                try:
                    # translated_text = GoogleTranslator(source=lang, target='english').translate(text)
                    translated_text = GoogleTranslator(source='spanish', target='english').translate(text.decode())
                    cv2.putText(img, translated_text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
                except deep_translator.exceptions.InvalidSourceOrTargetLanguage:
                    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            except IndexError:
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

cv2.imshow('Result', img)
cv2.waitKey(0)