import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

class RealTimeEmotionDetection:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('800x600')
        self.root.title('Real-Time Emotion Detector')
        self.root.configure(background='#CDCDCD')

        self.label = Label(self.root, background='#CDCDCD', font=('arial', 15, 'bold'))
        self.label.pack(side='bottom', expand='True')

        self.capture_button = Button(self.root, text="Capture Video", command=self.capture_video, padx=10, pady=5)
        self.capture_button.configure(background="#364156", foreground='white', font=('arial', 12, 'bold'))
        self.capture_button.pack(side='bottom', pady=20)

        self.model = self.load_model("model_RTED.json", "model_weights_RTED.h5")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_model(self, json_file, weights_file):
        with open(json_file, "r") as file:
            loaded_model_json = file.read()
            model = model_from_json(loaded_model_json)

        model.load_weights(weights_file)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def capture_video(self):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture video.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y + h, x:x + w]
                resized_roi = cv2.resize(face_roi, (48, 48))
                resized_roi = np.expand_dims(np.expand_dims(resized_roi, -1), 0)
                prediction = self.model.predict(resized_roi)
                emotion_label = np.argmax(prediction)
                emotion = EMOTIONS_LIST[emotion_label]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Real-Time Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Netural","Sad", "Surprise"]
    app = RealTimeEmotionDetection()
    app.root.mainloop()
