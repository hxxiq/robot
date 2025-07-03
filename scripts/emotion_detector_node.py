#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import cv2
import numpy as np
import time
import os
import random
from collections import Counter
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import speech_recognition as sr

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotions(duration=7):
    rospy.loginfo("📍 Starting emotion detection...")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    try:
        emotion_model = load_model('emotion_model.hdf5')
    except Exception as e:
        rospy.logerr(f"❌ Failed to load model: {e}")
        return None

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        rospy.logerr("❌ Could not access the webcam.")
        return None

    start_time = time.time()
    emotion_counter = []

    while time.time() - start_time < duration and not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("❌ Failed to read frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi, verbose=0)[0]
            emotion = emotion_labels[np.argmax(prediction)]
            emotion_counter.append(emotion)

            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not emotion_counter:
        return None

    freq = Counter(emotion_counter)
    top_emotions = freq.most_common(2)
    return {"all": dict(freq), "most_common": top_emotions}

def decide_emotion(result):
    if not result:
        return None

    freq = result["all"]
    top = result["most_common"]

    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]

    tied = [emo for emo, count in freq.items() if count == top[0][1]]
    if "Neutral" in tied:
        tied.remove("Neutral")
    return random.choice(tied) if tied else None

def ask_user(emotion):
    question = f"I see you're feeling {emotion}. What made you feel that way?"
    os.system(f'espeak "{question}"')

    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        rospy.loginfo(f"🗣️ User: {user_input}")
        return user_input
    except:
        rospy.logwarn("❌ Could not understand audio.")
        return "Sorry, I couldn't hear that clearly."

def emotion_node():
    rospy.init_node('emotion_detector_node')
    emotion_pub = rospy.Publisher('/emotion_result', String, queue_size=10)

    rate = rospy.Rate(0.1)  # 1 loop every 10s
    while not rospy.is_shutdown():
        result = detect_emotions(duration=7)
        final_emotion = decide_emotion(result)

        if final_emotion:
            rospy.loginfo(f"🧠 Final Emotion: {final_emotion}")
            emotion_pub.publish(final_emotion)
            ask_user(final_emotion)
        else:
            rospy.logwarn("⚠️ No emotion detected.")

        rate.sleep()

if __name__ == '__main__':
    try:
        emotion_node()
    except rospy.ROSInterruptException:
        pass
