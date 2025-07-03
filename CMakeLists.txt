cmake_minimum_required(VERSION 3.0.2)
project(robot)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  sound_play
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/main_robot.py
  scripts/face_detector.py
  scripts/emotion_detector.py
  scripts/speech_input.py
  scripts/response_generator.py
  scripts/activity_helper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
