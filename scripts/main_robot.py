#!/usr/bin/env python3

import rospy
import time
import pyttsx3

def speak(text):
    """Use TTS to speak a message"""
    engine = pyttsx3.init(driverName='espeak')
    

    voices = engine.getProperty('voices')
    for v in voices:
        print(f"Voice name: {v.name}, ID: {v.id}")

    engine.say(text)
    engine.runAndWait()

def activate_robot():
    """Robot activation routine"""
    rospy.init_node('main_robot_node', anonymous=True)
    rospy.loginfo("Robot is starting up...")

# Greet the user
    speak("Hello! I am online and ready to assist you.")
    rospy.loginfo("Greeting completed.")

# Simulate some startup wait
    time.sleep(1)
    rospy.loginfo("System initialized. Proceeding to main function...")

if __name__ == '__main__':
    try:
        activate_robot()
    except rospy.ROSInterruptException:
        pass
