import cv2
import numpy as np
import random
import time
import os
from collections import Counter
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import speech_recognition as sr

# Emotion labels expected by the model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotions(duration=7):
    print("📍 Starting emotion detection...")

    # Load face detection and emotion model
    print("📍 Loading models...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    try:
        emotion_model = load_model('emotion_model.hdf5')
    except Exception as e:
        print(f"❌ Failed to load emotion model: {e}")
        return None

    cap = cv2.VideoCapture(2)  # Adjust camera index if needed
    if not cap.isOpened():
        print("❌ Could not access the webcam.")
        return None

    print(f"🧠 Capturing emotions for {duration} seconds...")
    start_time = time.time()
    emotion_counter = []

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi, verbose=0)[0]
            max_index = np.argmax(prediction)
            emotion = emotion_labels[max_index]
            emotion_counter.append(emotion)

            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Quit pressed.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if not emotion_counter:
        print("⚠️ No emotions were detected.")
        return {"all": {}, "most_common": []}

    emotion_freq = Counter(emotion_counter)
    top_emotions = emotion_freq.most_common(2)

    print("✅ Emotion detection complete.")
    print("All detected:", dict(emotion_freq))
    print("Most common:", top_emotions)

    return {
        "all": dict(emotion_freq),
        "most_common": top_emotions
    }

def decide_emotion(result):
    if not result:
        return None

    emotion_freq = result.get("all", {})
    most_common = result.get("most_common", [])

    if not emotion_freq or not most_common:
        return None

    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        return most_common[0][0]

    max_count = most_common[0][1]
    tied_emotions = [emo for emo, count in emotion_freq.items() if count == max_count]

    if "Neutral" in tied_emotions:
        tied_emotions.remove("Neutral")

    if tied_emotions:
        return random.choice(tied_emotions)
    else:
        return None

def ask_user_about_emotion(emotion):
    if not emotion:
        return None

    question = f"I see you're feeling {emotion}. What made you feel that way?"
    print("🤖 Asking:", question)
    os.system(f'espeak "{question}"')

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🎤 Listening for user response...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        print(f"🗣️ User said: {user_input}")
        return user_input
    except sr.UnknownValueError:
        print("❌ Could not understand the audio.")
    except sr.RequestError:
        print("❌ Could not request results from Google Speech Recognition.")

    return None

def generate_offline_response(user_input, emotion):
    responses = {
        "Sad": [
            "I'm sorry you're feeling sad. Remember, it's okay to feel this way. Try talking to someone you trust.",
            "Everyone has tough days. Maybe doing something you enjoy could lift your spirits.",
            "Sadness doesn't last forever. Be kind to yourself and take it one step at a time."
        ],
        "Angry": [
           "Anger is a valid emotion. Try taking deep breaths or going for a short walk to clear your mind.",
            "It helps to express anger in healthy ways. Maybe writing your thoughts down could help.",
            "Would you like to try a calming technique to manage your anger?"
        ],
        "Fear": [
           "Facing fear is brave. Remind yourself that you're safe right now.",
           "Try grounding yourself with your surroundings — name 5 things you can see or hear.",
           "You're not alone. It's okay to feel fear, but you are stronger than you think."
        ],
        "Happy": [
            "That's wonderful! Happiness is worth savoring. Keep doing what makes you feel good.",
            "I’m so glad you’re feeling happy. Maybe you can spread this joy to someone else too.",
            "Happiness is powerful — take a moment to be grateful for what’s going well."
        ],
        "Surprise": [
            "Surprises can catch us off guard. It's okay to take a moment to process it.",
            "Whether it’s a good or bad surprise, I'm here to help you reflect on it.",
            "Sometimes unexpected things bring new opportunities. Stay open to possibilities."
        ],
        "Disgust": [
            "That sounds unpleasant. Try to remove yourself from things that make you uncomfortable.",
            "Your feelings are valid. Focus on something that brings you comfort or peace.",
            "It’s okay to feel this way. Maybe a change of scenery or a favorite song can help reset your mood."
        ],
        "Neutral": [
            "Even neutral days are a part of life. You can use this time to relax or recharge.",
            "If you’re unsure how you feel, maybe journaling or deep breathing can help clarify things.",
            "Want to explore something meaningful together, like setting a small goal or trying a new hobby?"
        ]
    }

    

    reply = random.choice(responses.get(emotion, ["Thank you for sharing."]))
    print(f"🤖 Robot Response: {reply}")
    os.system(f'espeak "{reply}"')

if __name__ == '__main__':
    while True:
        result = detect_emotions(duration=7)
        print("🧾 Raw Result:", result)

        final_emotion = decide_emotion(result)
        print(f"🧠 Final Emotion Decided: {final_emotion}")

        if final_emotion:
            print("✅ Proceeding to ask user about emotion...")
            user_reply = ask_user_about_emotion(final_emotion)
            if user_reply:
                generate_offline_response(user_reply, final_emotion)
        else:
            print("⚠️ No valid emotion detected. Retrying...")

        print("\n🔁 Do you want to detect emotion again? (y/n): ", end="")
        choice = input().strip().lower()
        if choice != 'y':
            print("👋 Exiting program.")
            break
