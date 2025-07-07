# AutoRobot
# Emotional Robot Assistant

## 2. System Requirements

### 2.1 Hardware Requirements

**Minimum Requirements:**
- 💻 **Computer**: Intel i5 (8th gen) or AMD Ryzen 5 equivalent
- 🧠 **Memory**: 8GB RAM (16GB recommended)
- 💾 **Storage**: 50GB free space
- 📷 **Camera**: USB webcam with 720p resolution or higher
- 🎤 **Microphone**: USB microphone or built-in laptop microphone
- 🔊 **Speakers**: External speakers or headphones
- 🌐 **Internet**: Stable connection (10 Mbps recommended)

**Recommended Hardware:**
- 📷 **Camera**: Logitech C920 or similar 1080p webcam
- 🎤 **Microphone**: Blue Yeti or similar USB microphone
- 🔊 **Speakers**: Dedicated desktop speakers for clear audio
- 💻 **Graphics**: NVIDIA GTX 1650 or better (for faster processing)

### 2.2 Software Requirements

**Operating System:**
- 🐧 **Ubuntu 20.04 LTS** (Primary support)
- 🪟 **Windows 11** (Limited support via WSL2)
- 🍎 **macOS 12+** (Experimental support)

**Required Software:**
- ROS Noetic (Robot Operating System)
- Python 3.8+
- OpenCV 4.5+
- TensorFlow 2.8+

### 2.3 Internet Services

**Required Services:**
- Google Speech Recognition API (for speech-to-text)
- Google Text-to-Speech API (for robot voice)
- Optional: OpenAI API (for advanced conversations)

---

## 3. Installation Guide

### 3.1 Step-by-Step Installation

#### Step 1: Prepare Your System

```bash
# Update your system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git python3-pip

# Install ROS Noetic
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install -y ros-noetic-desktop-full
```

#### Step 2: Download the Robot Software

```bash
# Create workspace directory
mkdir -p ~/emotional_robot_ws/src
cd ~/emotional_robot_ws/src

# Clone the repository
git clone https://github.com/emotional-robot/emotional-robot-assistant.git

# Navigate to workspace
cd ~/emotional_robot_ws

# Build the project
catkin_make
```

#### Step 3: Install Dependencies

```bash
# Install Python dependencies
pip3 install -r src/emotional-robot-assistant/requirements.txt

# Install ROS dependencies
rosdep install --from-paths src --ignore-src -r -y
```

#### Step 4: Configure Environment

```bash
# Add to your shell configuration
echo "source ~/emotional_robot_ws/devel/setup.bash" >> ~/.bashrc
echo "export ROS_WORKSPACE=~/emotional_robot_ws" >> ~/.bashrc

# Reload configuration
source ~/.bashrc
```

#### Step 5: Download AI Models

```bash
# Download emotion detection model
cd ~/emotional_robot_ws/src/emotional-robot-assistant/models/
wget https://github.com/emotional-robot/models/releases/download/v1.0/emotion_model.h5

# Verify download
ls -la *.h5
```

### 3.2 Hardware Setup

#### Camera Setup
1. **Connect your webcam** to a USB 3.0 port
2. **Test camera** with: `cheese` or `vlc v4l2:///dev/video0`
3. **Adjust positioning**: Place camera at eye level, 2-3 feet away
4. **Check lighting**: Ensure good, even lighting on your face

#### Microphone Setup
1. **Connect microphone** to USB port
2. **Test audio input**: `arecord -l` to list devices
3. **Test recording**: `arecord -d 5 test.wav` then `aplay test.wav`
4. **Adjust levels**: Use `alsamixer` to set appropriate input levels

#### Speaker Setup
1. **Connect speakers** or headphones
2. **Test audio output**: `speaker-test -t wav -c 2`
3. **Adjust volume**: Set to comfortable level for conversation

---

## 4. Quick Start Guide

### 4.1 First Time Setup (5 minutes)

#### Step 1: Launch the Robot

```bash
# Open terminal and run
roslaunch robot_controller emotional_robot.launch
```

**What you should see:**
```
[INFO] [1690123456.789]: Face Detector Node initialized
[INFO] [1690123456.890]: Emotion Classifier Node initialized  
[INFO] [1690123456.945]: Speech Recognizer Node initialized
[INFO] [1690123457.012]: TTS Synthesizer Node initialized
[INFO] [1690123457.089]: Emotional Brain Node initialized
[INFO] [1690123457.123]: Robot ready for interaction!
```

#### Step 2: Position Yourself

1. **Sit 2-3 feet** from your camera
2. **Look directly** at the camera
3. **Ensure good lighting** on your face
4. **Check audio levels** - speak at normal volume

#### Step 3: First Interaction

1. **Wait for greeting**: Robot will say "Hello! I'm your emotional support robot."
2. **Let robot detect emotions**: Stay still for 7 seconds while robot analyzes
3. **Respond to questions**: Robot will ask about your feelings
4. **Follow suggestions**: Try the wellness activities when offered

### 4.2 Basic Commands

**Starting the Robot:**
```bash
# Full system launch
roslaunch robot_controller emotional_robot.launch

# Debug mode (with visual feedback)
roslaunch robot_controller emotional_robot.launch debug:=true

# Quiet mode (minimal logging)
roslaunch robot_controller emotional_robot.launch quiet:=true
```

**Stopping the Robot:**
```bash
# Graceful shutdown
rosnode kill -a

# Or press Ctrl+C in the terminal
```

**Checking Robot Status:**
```bash
# See active nodes
rosnode list

# Check topics
rostopic list

# Monitor emotion detection
rostopic echo /emotion/current_emotion
```

---

## 5. Using Your Robot

### 5.1 How Interaction Works

#### The Robot's Process:
1. **Activation** (3 seconds): Robot starts up and initializes
2. **Detection** (7 seconds): Camera watches for your face and emotions
3. **Analysis** (2 seconds): AI determines your dominant emotion
4. **Response** (3 seconds): Robot speaks appropriate response
5. **Listening** (10 seconds): Robot waits for your voice response
6. **Support** (5 seconds): Robot provides helpful advice
7. **Activity** (5-15 minutes): Optional wellness exercises
8. **Loop**: Process repeats or ends based on your choice

#### Your Role:
- **Be visible**: Stay in camera view during emotion detection
- **Speak clearly**: Use normal speaking voice and pace
- **Be patient**: Allow robot time to process and respond
- **Engage honestly**: Share your feelings for better support

### 5.2 Emotion Detection

#### What Emotions Are Recognized?

| Emotion | What It Looks Like | Robot Response |
|---------|-------------------|----------------|
| 😊 **Happy** | Smiling, bright eyes | Celebrates with you, suggests gratitude exercises |
| 😢 **Sad** | Frowning, downcast eyes | Offers comfort, suggests breathing exercises |
| 😠 **Angry** | Furrowed brow, tense jaw | Provides calming support, suggests relaxation |
| 😨 **Fear** | Wide eyes, tense expression | Offers reassurance, suggests grounding exercises |
| 😲 **Surprise** | Raised eyebrows, open mouth | Shows curiosity, helps process the surprise |
| 🤢 **Disgust** | Wrinkled nose, negative expression | Offers understanding, suggests cleansing activities |
| 😐 **Neutral** | Relaxed, no strong expression | Continues monitoring, casual conversation |

#### Tips for Better Detection:
- **Good lighting**: Avoid shadows on your face
- **Clear view**: Remove glasses or hats if they obstruct your face
- **Stay still**: Minimize movement during 7-second detection period
- **Express naturally**: Don't force expressions - be authentic
- **Multiple emotions**: Robot handles mixed emotions intelligently

### 5.3 Voice Interaction

#### Speaking to Your Robot:

**Best Practices:**
- **Speak clearly** at normal pace
- **Use normal volume** - don't shout or whisper
- **Pause between sentences** for better recognition
- **Avoid background noise** when possible
- **Be patient** - robot will indicate when it's listening

**Sample Conversations:**

**Robot:** "I can see you're feeling sad. What made you feel this way today?"

**You:** "I had a really difficult day at work. My boss was criticizing everything I did."

**Robot:** "That sounds really frustrating and hurtful. Work stress can be especially challenging when you feel like your efforts aren't being recognized. It's completely understandable that you'd feel sad about this."

**Common Voice Commands:**
- "Yes" / "No" - Respond to robot questions
- "I don't want to talk about it" - Skip topic
- "Can you repeat that?" - Ask for clarification
- "Stop" - End current activity
- "Continue" - Proceed with suggestion
- "Help" - Get usage instructions

### 5.4 Understanding Robot Behavior

#### Robot Personality:
Your robot has a consistent, caring personality:
- **Empathetic**: Acknowledges and validates your feelings
- **Patient**: Never rushes you or pressures you
- **Supportive**: Offers practical help and encouragement
- **Respectful**: Honors your boundaries and choices
- **Consistent**: Maintains the same caring approach

#### How Robot Makes Decisions:

**Emotion Conflicts:**
If robot detects multiple emotions equally:
1. **Removes "neutral"** if other emotions are present
2. **Randomly selects** from remaining tied emotions
3. **Explains its choice**: "I'm detecting mixed emotions - anger and sadness. I'll focus on the anger I'm sensing."

**Response Selection:**
Robot chooses responses based on:
- **Your current emotion** (primary factor)
- **Conversation history** (what you've discussed)
- **Your preferences** (learned over time)
- **Appropriate timing** (when to suggest activities)

---

## 6. Understanding Robot Responses

### 6.1 Types of Responses

#### 6.1.1 Empathetic Acknowledgment
**Purpose**: Validate your emotions and show understanding

**Examples:**
- "I can see you're feeling sad. That's completely understandable."
- "Your anger about this situation makes perfect sense."
- "It's natural to feel surprised when unexpected things happen."

#### 6.1.2 Supportive Guidance
**Purpose**: Offer perspective and coping strategies

**Examples:**
- "Remember that difficult feelings are temporary and will pass."
- "You're stronger than you realize, and you've handled challenges before."
- "It's okay to feel this way - your emotions are valid."

#### 6.1.3 Clarifying Questions
**Purpose**: Understand your situation better

**Examples:**
- "Can you tell me more about what happened?"
- "How long have you been feeling this way?"
- "What aspect of this situation bothers you most?"

#### 6.1.4 Activity Suggestions
**Purpose**: Offer practical wellness activities

**Examples:**
- "Would you like to try a breathing exercise to help calm your mind?"
- "I can guide you through a short meditation that might help."
- "How about we do some gentle stretching to release tension?"

### 6.2 Conversation Flow

#### Typical Interaction Pattern:

1. **Greeting**: "Hello! I'm here to support you today."
2. **Emotion Check**: "I can see you're feeling [emotion]. That's completely understandable."
3. **Inquiry**: "What made you feel this way?"
4. **Listening**: Robot processes your response
5. **Support**: "That sounds really [challenging/exciting/etc]. [Supportive statement]"
6. **Suggestion**: "Would you like to try [activity] to help with this?"
7. **Activity or Continuation**: Either guides activity or continues conversation
8. **Check-in**: "How are you feeling now?"
9. **Closure**: "I'm here whenever you need support."

### 6.3 Adaptive Responses

#### How Robot Learns About You:

**Conversation Memory:**
- Remembers topics you've discussed in current session
- Refers back to things you've mentioned
- Avoids repeating the same suggestions

**Preference Learning:**
- Notes which activities you prefer
- Remembers your typical response patterns
- Adapts suggestion timing based on your preferences

**Emotional Pattern Recognition:**
- Recognizes your emotional patterns over time
- Adjusts responses based on your typical emotional intensity
- Provides more personalized support

---

## 7. Wellness Activities

### 7.1 Available Activities

#### 7.1.1 Breathing Exercises

**4-4-6 Breathing (Recommended for Stress/Anxiety)**
- **Duration**: 3-5 minutes
- **Instructions**: Robot guides you through inhaling for 4 counts, holding for 4, exhaling for 6
- **Benefits**: Reduces stress, calms nervous system, improves focus
- **When Used**: For anger, fear, or high stress emotions

**Deep Belly Breathing (For Sadness/Depression)**
- **Duration**: 5-7 minutes
- **Instructions**: Focus on deep diaphragmatic breathing
- **Benefits**: Increases oxygen flow, promotes relaxation
- **When Used**: When feeling sad, tired, or emotionally drained

#### 7.1.2 Meditation Practices

**Mindfulness Meditation**
- **Duration**: 5-10 minutes
- **Instructions**: Robot guides attention to breath and present moment
- **Benefits**: Reduces anxiety, improves emotional regulation
- **When Used**: For general stress, overwhelm, or when feeling scattered

**Body Scan Meditation**
- **Duration**: 10-15 minutes
- **Instructions**: Progressive relaxation of each body part
- **Benefits**: Releases physical tension, promotes deep relaxation
- **When Used**: For physical stress, anger, or before sleep

#### 7.1.3 Grounding Techniques

**5-4-3-2-1 Technique**
- **Duration**: 3-5 minutes
- **Instructions**: 
  - Name 5 things you can see
  - Name 4 things you can touch
  - Name 3 things you can hear
  - Name 2 things you can smell
  - Name 1 thing you can taste
- **Benefits**: Reduces anxiety, brings you to present moment
- **When Used**: For panic, overwhelm, or dissociation

**Progressive Muscle Relaxation**
- **Duration**: 8-12 minutes
- **Instructions**: Tense and release muscle groups systematically
- **Benefits**: Releases physical tension, reduces stress
- **When Used**: For anger, physical stress, or before sleep

#### 7.1.4 Gentle Movement

**Neck and Shoulder Stretches**
- **Duration**: 3-5 minutes
- **Instructions**: Robot guides through gentle neck rotations and shoulder rolls
- **Benefits**: Releases tension, improves circulation
- **When Used**: For stress, anger, or long periods of sitting

**Seated Spinal Twist**
- **Duration**: 2-3 minutes
- **Instructions**: Gentle twisting stretches while seated
- **Benefits**: Improves flexibility, releases lower back tension
- **When Used**: For physical discomfort, stress, or stiffness

### 7.2 Activity Guidance

#### How Activities Are Selected:

**Emotion-Based Selection:**
- **Sad**: Gentle breathing, uplifting meditation
- **Angry**: Calming breathing, muscle relaxation
- **Anxious**: Grounding techniques, mindfulness
- **Stressed**: Progressive relaxation, gentle movement
- **Happy**: Gratitude exercises, energizing activities

**Intensity Matching:**
- **High intensity emotions**: Longer, more involved activities
- **Low intensity emotions**: Shorter, gentler activities
- **Mixed emotions**: Balanced approach with options

#### During Activities:

**Robot Guidance:**
- **Clear instructions**: Step-by-step guidance
- **Timing cues**: "Breathe in for 4... 3... 2... 1..."
- **Encouragement**: "You're doing great", "Take your time"
- **Flexibility**: "If that's uncomfortable, try this instead"
- **Progress tracking**: "We're halfway through", "Just two more rounds"

**Your Role:**
- **Follow at your pace**: Don't rush or strain
- **Communicate**: Say "stop" if you need to pause
- **Be present**: Focus on the activity rather than other thoughts
- **Practice regularly**: Benefits increase with repetition

### 7.3 Customizing Activities

#### Personal Preferences:

**Activity Duration:**
- **Short** (2-5 minutes): For busy schedules or low energy
- **Medium** (5-10 minutes): Standard duration for most activities
- **Long** (10-20 minutes): For deep relaxation or when you have time

**Activity Type:**
- **Active**: Gentle movement, stretching
- **Passive**: Meditation, breathing exercises
- **Mixed**: Combination of movement and stillness

**Intensity Level:**
- **Gentle**: Slow, easy movements
- **Moderate**: Standard activity level
- **Energizing**: More dynamic activities for low energy

#### Modifying Activities:

**Physical Limitations:**
- Tell robot about any physical limitations
- Activities can be adapted for seated or limited mobility
- Alternative instructions provided for different abilities

**Time Constraints:**
- Request shorter versions: "Can we do a 3-minute version?"
- Robot can suggest quick alternatives
- Activities can be paused and resumed later

**Environmental Factors:**
- Adapt for quiet environments (silent activities)
- Modify for small spaces
- Adjust for different lighting conditions

---

## 8. Troubleshooting

### 8.1 Common Issues and Solutions

#### 8.1.1 Robot Not Starting

**Problem**: Robot doesn't launch or shows errors

**Symptoms:**
- Terminal shows error messages
- No robot greeting
- Camera or microphone not working

**Solutions:**

**Check System Status:**
```bash
# Verify ROS installation
roscore &
# Should start without errors

# Check if robot workspace is sourced
echo $ROS_PACKAGE_PATH
# Should include your robot workspace path
```

**Fix Common Issues:**
```bash
# Restart ROS master
killall -9 roscore rosmaster
roscore &

# Rebuild workspace
cd ~/emotional_robot_ws
catkin_clean
catkin_make

# Reset environment
source devel/setup.bash
```

**Hardware Check:**
```bash
# Test camera
ls /dev/video*
# Should show /dev/video0 or similar

# Test microphone
arecord -l
# Should list your microphone

# Test speakers
speaker-test -t wav -c 2
```

#### 8.1.2 Emotion Detection Issues

**Problem**: Robot can't detect emotions or gives wrong emotions

**Symptoms:**
- Robot says "I don't detect any emotions"
- Consistently wrong emotion detection
- Robot only detects "neutral"

**Solutions:**

**Improve Lighting:**
- Sit facing a window or lamp
- Avoid backlighting (light behind you)
- Use soft, even lighting
- Avoid harsh shadows on your face

**Check Camera Position:**
- Position camera at eye level
- Sit 2-3 feet from camera
- Ensure full face is visible
- Remove glasses or hats if blocking face

**Calibrate System:**
```bash
# Test emotion detection
rostopic echo /emotion/current_emotion

# View camera feed
rosrun image_view image_view image:=/camera/image_raw

# Check face detection
rostopic echo /emotion/faces_detected
```

#### 8.1.3 Speech Recognition Problems

**Problem**: Robot can't understand your speech

**Symptoms:**
- Robot says "I didn't catch that"
- Consistently misunderstands words
- Long delays in speech processing

**Solutions:**

**Improve Audio Quality:**
- Speak clearly and at normal pace
- Reduce background noise
- Move closer to microphone
- Check microphone levels

**Audio Settings:**
```bash
# Test microphone
arecord -d 5 test.wav
aplay test.wav

# Adjust input levels
alsamixer
# Use arrow keys to adjust microphone input

# Check audio devices
pacmd list-sources
```

**Network Issues:**
- Check internet connection
- Restart router if needed
- Try offline mode if available

#### 8.1.4 Robot Speech Issues

**Problem**: Robot voice is unclear or doesn't work

**Symptoms:**
- No robot voice output
- Distorted or robotic speech
- Very slow or fast speech

**Solutions:**

**Audio Output:**
```bash
# Test speakers
speaker-test -t wav -c 2

# Check audio output
pacmd list-sinks

# Adjust volume
alsamixer
```

**Speech Settings:**
- Adjust speech rate in robot settings
- Try different voice options
- Check internet connection for TTS service

### 8.2 Performance Issues

#### 8.2.1 Slow Response Times

**Problem**: Robot takes too long to respond

**Expected Performance:**
- Emotion detection: 2-3 seconds
- Speech recognition: 3-5 seconds
- Robot response: 1-2 seconds
- Total interaction: 5-10 seconds

**Solutions:**

**System Optimization:**
```bash
# Close unnecessary programs
# Check CPU usage
htop

# Free up memory
sudo apt autoremove
sudo apt autoclean

# Restart robot with optimized settings
roslaunch robot_controller emotional_robot.launch performance:=optimized
```

**Hardware Upgrades:**
- Add more RAM (16GB recommended)
- Use faster storage (SSD)
- Upgrade graphics card for faster processing

#### 8.2.2 High CPU Usage

**Problem**: Computer becomes slow during robot operation

**Monitoring:**
```bash
# Check resource usage
htop
# Look for high CPU processes

# Monitor ROS nodes
rosnode list
rostopic hz /camera/image_raw
```

**Optimization:**
```bash
# Reduce camera resolution
roslaunch robot_controller emotional_robot.launch camera_resolution:=720p

# Reduce processing frequency
roslaunch robot_controller emotional_robot.launch processing_rate:=low
```

### 8.3 Error Messages and Solutions

#### Common Error Messages:

**"No module named 'cv2'"**
```bash
# Install OpenCV
pip3 install opencv-python
```

**"Failed to connect to camera"**
```bash
# Check camera connection
ls /dev/video*
# Restart camera node
rosnode kill /camera_node
```

**"Speech recognition service unavailable"**
```bash
# Check internet connection
ping google.com
# Restart speech node
rosnode kill /speech_recognizer_node
```

**"Emotion model not found"**
```bash
# Download emotion model
cd ~/emotional_robot_ws/src/emotional-robot-assistant/models/
wget https://github.com/emotional-robot/models/releases/download/v1.0/emotion_model.h5
```

---

## 9. Maintenance & Care

### 9.1 Regular Maintenance

#### 9.1.1 Daily Care

**Before Each Use:**
- **Clean camera lens** with soft cloth
- **Check microphone** for dust or obstructions
- **Verify good lighting** in your interaction area
- **Test audio levels** before starting

**After Each Use:**
- **Close robot properly** using Ctrl+C or shutdown command
- **Review session** if you want to track progress
- **Clean up workspace** for next session

#### 9.1.2 Weekly Maintenance

**System Health Check:**
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Check disk space
df -h
# Ensure at least 10GB free space

# Check for log file buildup
ls -la ~/.ros/log/
# Delete old log files if needed
```

**Performance Monitoring:**
```bash
# Check robot performance
rosrun robot_controller system_diagnostics.py

# Monitor resource usage
htop
# Look for memory leaks or high CPU usage
```

#### 9.1.3 Monthly Maintenance

**Deep System Cleanup:**
```bash
# Clean ROS logs
rm -rf ~/.ros/log/*

# Clean temporary files
sudo apt autoclean
sudo apt autoremove

# Rebuild workspace
cd ~/emotional_robot_ws
catkin_clean
catkin_make
```

**Hardware Inspection:**
- **Check all connections** (USB devices, power cables)
- **Clean computer vents** to prevent overheating
- **Update camera and microphone drivers** if needed
- **Test backup hardware** if available

### 9.2 Software Updates

#### 9.2.1 Robot Software Updates

**Check for Updates:**
```bash
# Navigate to robot directory
cd ~/emotional_robot_ws/src/emotional-robot-assistant

# Check for updates
git fetch
git status
# Shows if updates are available
```

**Update Robot Software:**
```bash
# Backup current configuration
cp -r config config_backup_$(date +%Y%m%d)

# Pull latest updates
git pull origin main

# Rebuild workspace
cd ~/emotional_robot_ws
catkin_make

# Test updated system
roslaunch robot_controller emotional_robot.launch test:=true
```

#### 9.2.2 Model Updates

**Emotion Model Updates:**
```bash
# Check for new emotion models
curl -s https://api.github.com/repos/emotional-robot/models/releases/latest | grep tag_name

# Download new model if available
cd ~/emotional_robot_ws/src/emotional-robot-assistant/models/
wget https://github.com/emotional-robot/models/releases/download/v1.1/emotion_model.h5
```

### 9.3 Data Management

#### 9.3.1 Session Data

**Understanding Data Storage:**
- **Emotion history**: Stored temporarily during sessions
- **User preferences**: Saved locally for personalization
- **Performance metrics**: Used for system optimization
- **No personal data**: Conversations are not permanently stored

**Data Locations:**
```bash
# User preferences
~/.config/emotional_robot/user_preferences.json

# Performance logs
~/.ros/log/latest/

# Temporary session data
/tmp/emotional_robot_session/
```

#### 9.3.2 Privacy Protection

**Data Cleanup:**
```bash
# Clear temporary session data
rm -rf /tmp/emotional_robot_session/*

# Reset user preferences (if desired)
rm ~/.config/emotional_robot/user_preferences.json

# Clear interaction logs
rm -rf ~/.ros/log/*
```

**Privacy Settings:**
- **Local processing**: All emotion detection happens on your computer
- **Cloud services**: Only used for speech recognition (can be disabled)
- **No data sharing**: Your conversations are never transmitted or stored remotely
