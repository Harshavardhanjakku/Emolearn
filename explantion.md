# EmoLearnNexus: A Beginner's Guide

## 🌟 What is EmoLearnNexus?

EmoLearnNexus is an intelligent learning platform that understands how you're feeling while you study. It uses your camera and microphone to detect your emotions and helps you learn better based on how you're feeling.

## 🛠️ How It Works (Workflow)

### 1. **Emotion Detection**
   - **Facial Analysis**: The system watches your face through the camera to see if you look confused, frustrated, bored, or engaged.
   - **Voice Analysis**: It can also listen to your voice to understand your emotional state.
   - **Interaction Tracking**: It notices how you're interacting with the learning material.

### 2. **Real-time Feedback**
   - If you look confused or frustrated, the system notices and offers help.
   - It shows your current emotion with emojis (like 😊 for happy, 🤔 for confused).
   - You can see your mood changes over time with a graph.

### 3. **Smart Assistance**
   - When you seem stuck, a friendly AI assistant pops up to help.
   - You can ask it questions about what you're learning.
   - It can explain topics in different ways if you're having trouble understanding.

### 4. **Interactive Features**
   - **Breathing Exercises**: If you're stressed, it can guide you through calming exercises.
   - **Educational Games**: Fun games to help you learn better when you're feeling bored.
   - **Progress Tracking**: Keeps track of your learning journey and emotions.

## 🔍 Key Components

### Frontend (What You See)
- **Video Player**: Shows your camera feed (with privacy controls).
- **Emotion Tracker**: Displays your current emotion.
- **AI Assistant**: A chat interface to ask questions.
- **Control Panel**: Buttons to control camera, microphone, etc.
- **Mood Graph**: Shows how your emotions change over time.

### Backend (The Brains)
- **Emotion Detection**: Analyzes your face and voice.
- **AI Tutor**: Answers your questions and explains topics.
- **Game Integration**: Manages educational games.
- **Data Storage**: Saves your progress and emotional data.

## 🎯 Why It's Special

1. **Personalized Learning**: Adjusts to your emotional state.
2. **Non-intrusive Help**: Only offers assistance when needed.
3. **Privacy-Focused**: You control the camera and microphone.
4. **Engaging**: Makes learning more interactive and fun.

## 🚀 Getting Started

1. **Start the Application**
   - The system will ask for camera/microphone access (you can say no if you prefer).
   - You'll see your video feed and the emotion tracker.

2. **Learn and Interact**
   - As you study, the system will track your emotions.
   - If you seem stuck, the AI assistant will offer help.
   - Use the breathing exercises if you feel stressed.

3. **Check Your Progress**
   - View your emotion history in the mood graph.
   - See your learning progress over time.

## 💡 Tips for Best Experience

- Make sure your face is well-lit for better emotion detection.
- Use headphones for better voice recognition.
- Don't hesitate to ask the AI assistant for help!
- Try the breathing exercises if you feel overwhelmed.

Let me dive into the backend implementation details. First, I'll examine the modules directory to understand the emotion detection system better.

Let me break down the backend architecture and implementation in detail:

## 🧠 Backend Architecture Overview

### 1. Core Detection Modules

#### a) Facial Emotion Detection ([facial_emotion.py](cci:7://file:///d:/emo-learn-nexus/backend/modules/facial_emotion.py:0:0-0:0))
- Uses OpenCV and DeepFace for real-time facial analysis
- Detects 7 basic emotions: happy, sad, angry, fear, surprise, neutral, disgust
- Tracks additional metrics:
  - Eye aspect ratio (for drowsiness)
  - Yawn detection
  - Head pose estimation
  - Blink rate

#### b) Speech Emotion Detection ([speech_emotion.py](cci:7://file:///d:/emo-learn-nexus/backend/modules/speech_emotion.py:0:0-0:0))
- Uses Librosa for audio feature extraction
- Implements MLPClassifier for emotion classification
- Analyzes:
  - Pitch
  - Energy
  - Spectral features
  - Speech rate
  - Pauses

#### c) Mouse Interaction Analysis ([mouse_emotion.py](cci:7://file:///d:/emo-learn-nexus/backend/modules/mouse_emotion.py:0:0-0:0))
- Tracks:
  - Mouse movement speed
  - Click patterns
  - Idle time
  - Movement smoothness
- Correlates patterns with engagement/frustration

#### d) Emotion Combiner ([emotion_combiner.py](cci:7://file:///d:/emo-learn-nexus/backend/modules/emotion_combiner.py:0:0-0:0))
- Weights and combines inputs from all detectors
- Implements temporal smoothing
- Handles sensor fusion conflicts
- Maintains emotion state machine

## 🔄 Main API Endpoints

### Emotion Detection Endpoints
- `GET /emotion` - Get current combined emotion
- `GET /emotion/facial` - Get facial emotion only
- `GET /emotion/voice` - Get voice emotion only
- `GET /emotion/interaction` - Get interaction-based emotion
- `POST /detection/start` - Start emotion detection
- `POST /detection/stop` - Stop emotion detection

### Sensor Control
- `POST /sensors/camera/toggle` - Toggle camera
- `POST /sensors/microphone/toggle` - Toggle microphone
- `GET /sensors/status` - Get sensor status

### AI Tutor
- `POST /explain` - Get AI explanation for a topic
- `GET /games/emotion` - Get emotion-specific games

### Data & Analytics
- `GET /emotion/history` - Get emotion history
- `GET /emotion/graph` - Get emotion timeline data

## 🖥️ Frontend Integration

### 1. Initialization
```typescript
// In App.tsx
const queryClient = new QueryClient();
// Sets up React Query for data fetching
```

### 2. Emotion Detection Flow
```typescript
// In EmotionTracker.tsx
const { data: emotion } = useQuery({
  queryKey: ['emotion'],
  queryFn: fetchEmotion,
  refetchInterval: 1000 // Poll every second
});
```

### 3. Sensor Control
```typescript
// In ControlPanel.tsx
const toggleCamera = async () => {
  await fetch('/api/sensors/camera/toggle', { method: 'POST' });
  refetchStatus(); // Update sensor status
};
```

### 4. AI Assistant Integration
```typescript
// In AIAssistant.tsx
const explainTopic = async (topic: string) => {
  const response = await fetch('/api/explain', {
    method: 'POST',
    body: JSON.stringify({ topic })
  });
  // Handle response...
};
```

## 🔄 Data Flow

1. **Frontend** captures video/audio streams
2. **Backend** processes streams through detection pipeline:
   ```
   Raw Input → Preprocessing → Feature Extraction → 
   Emotion Classification → Sensor Fusion → Final Emotion
   ```
3. **Frontend** receives emotion updates and updates UI
4. **AI Assistant** uses emotion context to provide help

## 🛠️ Key Implementation Details

### 1. Multi-threaded Processing
```python
# In main.py
def start_emotion_detection():
    global emotion_detection_running, emotion_thread
    if not emotion_detection_running:
        emotion_detection_running = True
        emotion_thread = threading.Thread(target=emotion_detection_loop)
        emotion_thread.daemon = True
        emotion_thread.start()
```

### 2. Emotion State Machine
```python
# In emotion_combiner.py
class EmotionState:
    def __init__(self):
        self.current_emotion = "neutral"
        self.confidence = 0.0
        self.duration = 0
        self.last_update = time.time()
```

### 3. Sensor Fusion
```python
def combine_emotions(facial, voice, interaction):
    # Apply weights to different modalities
    weights = {
        'facial': 0.5,
        'voice': 0.3,
        'interaction': 0.2
    }
    # Combine using weighted average
    # ...
```

## 📊 Data Storage

1. **Real-time Data**: Kept in memory for low-latency access
2. **Historical Data**: Stored in [emotion_log.json](cci:7://file:///d:/emo-learn-nexus/backend/emotion_log.json:0:0-0:0) and [emotion_log.csv](cci:7://file:///d:/emo-learn-nexus/backend/emotion_log.csv:0:0-0:0)
3. **Configuration**: Stored in JSON files like [emotion_weights.json](cci:7://file:///d:/emo-learn-nexus/backend/emotion_weights.json:0:0-0:0)

## 🚀 Performance Considerations

1. **Optimized Frame Processing**:
   - Frame skipping for high CPU usage
   - Async processing for non-critical tasks
   - Batch processing where possible

2. **Memory Management**:
   - Circular buffers for recent data
   - Periodic cleanup of old entries
   - Efficient serialization for storage

## 🛡️ Security & Privacy

1. **Data Protection**:
   - Raw video/audio never stored
   - Local processing preferred over cloud
   - Secure WebSocket connections

2. **User Control**:
   - Explicit permissions for sensors
   - Easy sensor toggling
   - Clear data policies

## 🔍 Debugging & Monitoring

1. **Logging**:
   - Detailed emotion detection logs
   - Performance metrics
   - Error tracking

2. **Diagnostic Endpoints**:
   - `GET /debug/status` - System health
   - `GET /debug/performance` - Performance metrics
   - `GET /debug/emotion` - Raw emotion data

## 🎯 Future Enhancements

1. **Advanced ML Models**:
   - Fine-tuned emotion detection
   - Personalization based on user patterns
   - Multi-modal fusion improvements

2. **New Features**:
   - More interactive games
   - Advanced analytics dashboard
   - Integration with learning management systems

Let me explain how the camera integration works in EmoLearnNexus:

## Camera Workflow

### 1. **User-Initiated Activation**
- The camera is **not** turned on automatically when the app loads
- Users must explicitly enable the camera through the UI
- This is a privacy-first approach

### 2. Frontend Implementation
When you click the camera toggle button in the ControlPanel component:

```typescript
// In ControlPanel.tsx
const toggleCamera = async () => {
  try {
    // Show permission dialog to user
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: true,
      audio: false
    });
    
    // Send request to backend to start processing
    const response = await fetch('/api/sensors/camera/toggle', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    // Handle response and update UI
    const data = await response.json();
    setIsCameraOn(data.is_active);
    
    // Clean up stream when component unmounts
    return () => {
      stream.getTracks().forEach(track => track.stop());
    };
  } catch (error) {
    console.error('Error accessing camera:', error);
  }
};
```

### 3. Backend Processing
When the frontend enables the camera:

1. **Initial Request**:
   ```http
   POST /api/sensors/camera/toggle
   ```

2. **Backend Response**:
   ```json
   {
     "status": "success",
     "is_active": true,
     "message": "Camera processing started"
   }
   ```

3. **Video Stream Handling**:
   - The video stream stays in the browser
   - Frames are captured and sent to the backend at regular intervals
   - This is done using the HTML5 Canvas API to capture frames

### 4. Frame Processing
```typescript
// In VideoPlayer.tsx
const captureFrame = () => {
  if (!videoRef.current || !isCameraOn) return;
  
  const canvas = document.createElement('canvas');
  canvas.width = videoRef.current.videoWidth;
  canvas.height = videoRef.current.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoRef.current, 0, 0);
  
  // Convert to base64 and send to backend
  const imageData = canvas.toDataURL('image/jpeg', 0.8);
  
  fetch('/api/process_frame', {
    method: 'POST',
    body: JSON.stringify({ image: imageData.split(',')[1] }),
    headers: {
      'Content-Type': 'application/json',
    },
  });
};

// Capture frame every 500ms when camera is on
useEffect(() => {
  let intervalId: NodeJS.Timeout;
  if (isCameraOn) {
    intervalId = setInterval(captureFrame, 500);
  }
  return () => clearInterval(intervalId);
}, [isCameraOn]);
```

### 5. Backend Frame Processing
```python
# In main.py
@app.post("/api/process_frame")
async def process_frame(request: Request):
    try:
        data = await request.json()
        image_data = base64.b64decode(data['image'])
        
        # Process frame with OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get emotion from facial detection
        emotion = facial_analyzer.analyze(frame)
        
        # Update emotion log
        update_emotion_log('facial', emotion)
        
        return {"status": "success", "emotion": emotion}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### 6. Important Privacy Features

1. **No Permanent Storage**:
   - Frames are processed in memory
   - No video is ever saved to disk
   - Only emotion data is logged

2. **User Control**:
   - Clear visual indicator when camera is active
   - Easy one-click camera toggle
   - Browser permission prompts

3. **Secure Transmission**:
   - All communication is over HTTPS
   - Frames are sent as base64-encoded strings
   - No third-party tracking

### 7. Performance Optimizations

1. **Frame Rate Control**:
   - Default: 2 FPS (frames per second)
   - Adjustable based on system performance
   - Frame skipping under high CPU load

2. **Resolution Management**:
   ```typescript
   // In VideoPlayer.tsx
   const constraints = {
     video: {
       width: { ideal: 640 },
       height: { ideal: 480 },
       frameRate: { ideal: 15, max: 30 }
     }
   };
   ```

3. **Network Efficiency**:
   - JPEG compression (quality: 0.8)
   - Only send frames when changes are detected
   - Throttling to prevent network congestion

### 8. Error Handling

1. **Permission Denied**:
   ```typescript
   try {
     const stream = await navigator.mediaDevices.getUserMedia({ video: true });
   } catch (err) {
     if (err.name === 'NotAllowedError') {
       // User denied camera access
       setCameraError('Camera access was denied');
     }
   }
   ```

2. **Camera Not Found**:
   ```typescript
   .catch(err => {
     if (err.name === 'NotFoundError') {
       setCameraError('No camera found');
     }
   });
   ```

### 9. Testing the Camera

1. **Manual Test**:
   - Open browser's developer tools (F12)
   - Go to Console tab
   - You should see logs when frames are being processed

2. **Debug Endpoint**:
   ```http
   GET /api/debug/camera_status
   ```
   Response:
   ```json
   {
     "camera_active": true,
     "last_frame_received": "2023-11-27T04:55:00Z",
     "fps": 2.1,
     "resolution": "640x480"
   }
   ```
Let me dive into the facial emotion detection implementation to provide a detailed explanation.

## Facial Emotion Detection Architecture

### 1. Core Technologies Used

#### a) Deep Learning Model
- **DeepFace Library**: The primary model used for emotion recognition
- **Model Type**: VGG-Face (default in DeepFace)
- **Emotions Detected**: 7 basic emotions + neutral
  - Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral

#### b) Computer Vision
- **MediaPipe Face Mesh**: For facial landmark detection (468 points)
- **OpenCV**: For image processing and camera handling

### 2. Key Components

#### a) Face Detection & Landmark Tracking
```python
# MediaPipe Face Mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.7
)
```
- Tracks 468 facial landmarks
- Used for calculating various facial metrics

#### b) Emotion Detection Metrics

1. **Eye Aspect Ratio (EAR)**
   ```python
   def calculate_ear(points, landmarks, w, h):
       # Points for left and right eyes
       left_eye = [landmarks.landmark[p] for p in points['left_eye']]
       right_eye = [landmarks.landmark[p] for p in points['right_eye']]
       
       # Calculate EAR for both eyes
       left_ear = (dist(left_eye[1], left_eye[5]) + dist(left_eye[2], left_eye[4])) / (2.0 * dist(left_eye[0], left_eye[3]))
       right_ear = (dist(right_eye[1], right_eye[5]) + dist(right_eye[2], right_eye[4])) / (2.0 * dist(right_eye[0], right_eye[3]))
       
       return (left_ear + right_ear) / 2.0
   ```
   - Detects blinks and drowsiness
   - Low EAR indicates closed or squinted eyes

2. **Mouth Aspect Ratio (MAR)**
   ```python
   def calculate_mar(points, landmarks, w, h):
       # Points for mouth
       mouth = [landmarks.landmark[p] for p in points['mouth']]
       
       # Calculate distances
       A = dist(mouth[13], mouth[19])  # Vertical
       B = dist(mouth[14], mouth[18])  # Horizontal
       C = dist(mouth[15], mouth[17])  # Horizontal
       
       return (A + A) / (2.0 * ((B + C) / 2))
   ```
   - Detects yawning and talking
   - High MAR indicates an open mouth

3. **Head Pose Estimation**
   ```python
   def calculate_pitch(landmarks):
       # Using nose tip and chin for pitch
       nose_tip = landmarks.landmark[1]
       chin = landmarks.landmark[152]
       return nose_tip.y - chin.y
   ```
   - Tracks head orientation
   - Detects if user is looking away

### 3. Emotion Detection Pipeline

1. **Frame Capture**
   ```python
   ret, frame = cap.read()
   if not ret:
       continue
   ```

2. **Face Detection**
   ```python
   results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
   if not results.multi_face_landmarks:
       update_emotion_history("Unknown")
       continue
   ```

3. **DeepFace Emotion Analysis**
   ```python
   try:
       # Convert to RGB for DeepFace
       rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
       # Analyze emotions
       analysis = DeepFace.analyze(
           rgb_frame,
           actions=['emotion'],
           enforce_detection=False,
           detector_backend='opencv'
       )
       
       # Get dominant emotion
       emotion = analysis[0]['dominant_emotion']
       emotion_score = analysis[0]['emotion'][emotion]
   ```

4. **Multi-Metric Fusion**
   ```python
   # Combine metrics for final emotion
   if emotion_score < 0.4:  # Low confidence in emotion
       if ear < EAR_THRESHOLD_SLEEPY:
           final_emotion = "Sleepy"
       elif mar > MAR_THRESHOLD_YAWN:
           final_emotion = "Bored"
       elif is_head_tilted(landmarks):
           final_emotion = "Confused"
       else:
           final_emotion = "Engaged"
   ```

### 4. Temporal Smoothing
```python
# Maintain a history of emotions
emotion_history = deque(maxlen=15)  # Last 15 frames

def get_smoothed_emotion():
    if not emotion_history:
        return "Neutral"
    
    # Count occurrences of each emotion
    counter = Counter(emotion_history)
    # Return most common emotion
    return counter.most_common(1)[0][0]
```

### 5. Performance Optimizations

1. **Frame Skipping**
   ```python
   frame_count = 0
   process_every_n_frame = 2  # Process every 2nd frame
   
   while running:
       frame_count += 1
       if frame_count % process_every_n_frame != 0:
           continue
       # Process frame...
   ```

2. **Asynchronous Processing**
   ```python
   def process_frame_async(frame):
       # Run in separate thread
       threading.Thread(target=process_frame, args=(frame,)).start()
   ```

### 6. Training Data (If Custom Training is Needed)

The system primarily uses pre-trained models, but here's how custom training would work:

1. **Dataset**: FER2013 or CK+
   - 48x48 grayscale images
   - 7 emotion classes
   - ~35,000 images

2. **Model Architecture** (if training from scratch):
   ```python
   model = Sequential([
       Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
       MaxPooling2D(2,2),
       Conv2D(128, (3,3), activation='relu'),
       MaxPooling2D(2,2),
       Flatten(),
       Dense(1024, activation='relu'),
       Dropout(0.5),
       Dense(7, activation='softmax')
   ])
   ```

### 7. Integration with Frontend

1. **Frame Capture** (Frontend)
   ```javascript
   const captureFrame = () => {
     const canvas = document.createElement('canvas');
     canvas.width = video.videoWidth;
     canvas.height = video.videoHeight;
     const ctx = canvas.getContext('2d');
     ctx.drawImage(video, 0, 0);
     return canvas.toDataURL('image/jpeg', 0.8);
   };
   ```

2. **API Endpoint** (Backend)
   ```python
   @app.post('/api/detect_emotion')
   async def detect_emotion(request: Request):
       data = await request.json()
       image_data = base64.b64decode(data['image'].split(',')[1])
       nparr = np.frombuffer(image_data, np.uint8)
       frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
       # Process frame
       emotion = process_frame(frame)
       return {'emotion': emotion}
   ```

### 8. Advanced Features

1. **Gaze Detection**
   ```python
   def detect_gaze(landmarks, frame_shape):
       # Get eye landmarks
       left_eye = landmarks[33:42]
       right_eye = landmarks[87:96]
       
       # Calculate eye aspect ratio
       left_ear = eye_aspect_ratio(left_eye)
       right_ear = eye_aspect_ratio(right_eye)
       
       # Determine gaze direction
       if left_ear < 0.2 and right_ear < 0.2:
           return "Looking Down"
       # Add more conditions...
   ```

2. **Micro-Expressions**
   ```python
   def detect_micro_expressions(prev_frame, current_frame):
       # Calculate optical flow
       prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
       curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
       
       flow = cv2.calcOpticalFlowFarneback(
           prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
       )
       
       # Analyze flow for micro-expressions
       # ...
   ```

### 9. Performance Metrics

1. **Accuracy**:
   - DeepFace: ~65-75% on FER2013
   - Custom models: Up to 85% with sufficient data

2. **Processing Speed**:
   - CPU: 3-5 FPS (with optimizations)
   - GPU: 20-30 FPS

3. **Memory Usage**:
   - Model: ~500MB
   - Per-frame processing: ~100-200MB

### 10. Limitations & Considerations

1. **Lighting Conditions**:
   - Works best with even, front lighting
   - Performance degrades in low light

2. **Occlusions**:
   - Glasses, masks, or hands can affect accuracy
   - Multiple faces in frame may cause confusion

3. **Cultural Differences**:
   - Models trained on Western datasets may have bias
   - Expression interpretation varies across cultures

4. **Privacy**:
   - All processing happens locally
   - No images are stored permanently
   - Users can disable camera at any time
The system uses **pre-trained models** for emotion detection, not custom-trained ones. Here's a breakdown:

1. **DeepFace Library**
   - Uses VGG-Face model (pre-trained on FER-2013 dataset)
   - No additional training needed
   - Handles 7 basic emotions out of the box

2. **MediaPipe Face Mesh**
   - Pre-trained facial landmark detection model
   - Provides 468 facial points
   - Used for calculating metrics like eye aspect ratio and mouth aspect ratio

3. **Custom Logic on Top**
   ```python
   # Example from the code
   if mar > MAR_THRESHOLD_YAWN:  # If mouth is open (yawning)
       if yawn_start is None:
           yawn_start = time.time()
       elif time.time() - yawn_start > YAWN_DURATION_BORED:
           current_emotion = "Bored"
   ```
   - Uses simple computer vision techniques
   - Combines metrics from the pre-trained models
   - Implements custom rules for specific states like "Bored" or "Sleepy"

The system doesn't require training data or fine-tuning - it works immediately with the pre-trained models. The custom code layer interprets the model outputs to determine the final emotion state.
You're absolutely right! Let me clarify how the system handles emotions like "Frustrated" and "Confused" using a combination of pre-trained models and custom logic.

## Emotion Detection Workflow

### 1. **Pre-trained Model Outputs (7 Basic Emotions)**
   - 😊 Happy
   - 😢 Sad
   - 😠 Angry
   - 😨 Fear
   - 😲 Surprise
   - 🤢 Disgust
   - 😐 Neutral

### 2. **Custom Emotion Mapping**
The system maps these basic emotions to more learning-specific states:

```python
# Simplified mapping logic
if emotion == "angry" and confidence > FRUSTRATION_PROB_THRESHOLD:
    return "Frustrated"
elif emotion in ["fear", "sad"]:
    return "Confused"
```

### 3. **Additional Detection Methods**

#### For "Frustrated" 😠
- **Facial Cues**:
  - Furrowed brows (using facial landmarks)
  - Tense jaw
  - Pressed lips

- **Behavioral Cues**:
  - Rapid head movements
  - Increased blink rate
  - Sudden changes in facial expression

#### For "Confused" 🤔
- **Facial Cues**:
  - Raised inner eyebrows
  - Head tilts
  - Asymmetrical facial expressions

- **Behavioral Cues**:
  - Prolonged gaze away from screen
  - Frequent looking around
  - Repeated similar expressions

### 4. Code Implementation

Here's how it's implemented in the code:

```python
# For Frustration
if (emotion == "angry" and 
    emotion_score > FRUSTRATION_PROB_THRESHOLD and
    is_brow_furrowed(landmarks)):
    return "Frustrated"

# For Confusion
if (emotion in ["fear", "sad"] and 
    is_head_tilted(landmarks) and
    time_since_last_interaction > CONFUSION_TIME_THRESHOLD):
    return "Confused"
```

### 5. Temporal Analysis

The system also considers how long these states persist:

```python
# Track emotion state over time
emotion_duration = time.time() - emotion_start_time
if current_emotion == "Frustrated" and emotion_duration > MAX_FRUSTRATION_DURATION:
    trigger_intervention("frustration")
```

### 6. Confidence Thresholds

Each custom emotion has specific confidence thresholds:

```python
EMOTION_THRESHOLDS = {
    "Frustrated": 0.65,  # Higher threshold to reduce false positives
    "Confused": 0.55,
    "Bored": 0.6,
    "Sleepy": 0.7
}
```

### 7. Multi-modal Confirmation

For critical states like frustration, the system requires multiple signals:

```python
def is_frustrated(face_emotion, voice_emotion, interaction_data):
    return (face_emotion == "angry" and 
            voice_emotion in ["angry", "sad"] and
            interaction_data.get("click_intensity") > CLICK_THRESHOLD)
```

This multi-layered approach ensures more accurate emotion detection by combining:
1. Raw model outputs
2. Facial landmark analysis
3. Behavioral patterns
4. Temporal context

Let me break down how the system handles combined emotion detection across multiple input sources (facial, mouse, and speech) and how it manages silent listening scenarios.

## Multi-Modal Emotion Fusion System

### 1. **State Management**

```python
# Global state tracking
STATE = {
    "last_emotion": "Neutral",
    "modality_weights": {
        "facial": 0.6,    # Higher weight for facial
        "voice": 0.25,    # Medium weight for voice
        "mouse": 0.15     # Lower weight for mouse
    },
    "last_update": {
        "facial": 0,
        "voice": 0,
        "mouse": 0
    },
    "timeout": {
        "voice": 5.0,    # seconds of silence before considering voice inactive
        "mouse": 10.0    # seconds of no movement before considering mouse inactive
    }
}
```

### 2. **Handling Silent Listening**

When the user is silently listening:
- **Voice Input**: No speech detected
- **Mouse Input**: No movement detected
- **Facial Input**: Only active input

```python
def get_combined_emotion(facial_data, voice_data, mouse_data):
    current_time = time.time()
    active_modalities = []
    emotion_scores = {"Neutral": 0.0}
    
    # 1. Process Facial Data (Always active when camera is on)
    if facial_data and (current_time - facial_data["timestamp"] < 1.0):  # 1 second timeout
        update_emotion_scores(emotion_scores, facial_data["emotion"], STATE["modality_weights"]["facial"])
        active_modalities.append("facial")
    
    # 2. Process Voice Data (Handles silence)
    voice_timeout = current_time - voice_data["last_detected"] > STATE["timeout"]["voice"]
    if voice_data["is_speaking"] or not voice_timeout:
        weight = STATE["modality_weights"]["voice"]
        if voice_timeout:
            # Reduce weight for silence
            weight *= 0.3
        update_emotion_scores(emotion_scores, voice_data["emotion"], weight)
        active_modalities.append("voice")
    
    # 3. Process Mouse Data (Handles inactivity)
    mouse_timeout = current_time - mouse_data["last_activity"] > STATE["timeout"]["mouse"]
    if not mouse_timeout:
        update_emotion_scores(emotion_scores, mouse_data["emotion"], STATE["modality_weights"]["mouse"])
        active_modalities.append("mouse")
    
    # Adjust weights based on active modalities
    if not active_modalities:
        return "Neutral"
    
    # Normalize scores
    total_weight = sum(STATE["modality_weights"][m] for m in active_modalities)
    normalized_scores = {e: s/total_weight for e, s in emotion_scores.items()}
    
    # Get emotion with highest score
    return max(normalized_scores.items(), key=lambda x: x[1])[0]
```

### 3. **Silent Listening Handling**

```python
def handle_silent_listening(current_emotion, duration):
    if duration > 30:  # After 30 seconds of silence
        if current_emotion == "Neutral":
            return "Bored"  # Prolonged neutral state
        elif current_emotion == "Confused":
            return "Need Help"  # Prolonged confusion
    return current_emotion
```

### 4. **Temporal Smoothing**

```python
class EmotionBuffer:
    def __init__(self, window_size=10):
        self.buffer = deque(maxlen=window_size)
        self.last_emotion = "Neutral"
    
    def add_emotion(self, emotion):
        self.buffer.append(emotion)
        
        # Count occurrences
        counts = {}
        for e in self.buffer:
            counts[e] = counts.get(e, 0) + 1
        
        # Get most frequent emotion
        if counts:
            self.last_emotion = max(counts.items(), key=lambda x: x[1])[0]
        
        return self.last_emotion
```

### 5. **Fallback Mechanism**

When multiple modalities are inactive:

```python
def get_fallback_emotion(active_modalities, last_known_emotions):
    if "facial" in active_modalities:
        return last_known_emotions["facial"]
    elif "voice" in active_modalities:
        return last_known_emotions["voice"]
    elif "mouse" in active_modalities:
        return last_known_emotions["mouse"]
    return "Neutral"  # Default fallback
```

### 6. **Real-world Example Flow**

1. **User starts listening silently:**
   - Facial: Neutral/Engaged
   - Voice: Silent (no input)
   - Mouse: Inactive
   - **Result**: System relies on facial analysis

2. **After 30 seconds of silence:**
   - If facial shows neutral: "Bored"
   - If facial shows confused: "Need Help?"

3. **When user starts speaking:**
   - Voice input gets higher weight
   - System quickly adapts to new emotional state

### 7. Confidence-based Weighting

```python
def calculate_modality_weights(confidence_scores):
    """Dynamically adjust weights based on confidence"""
    total = sum(confidence_scores.values())
    if total == 0:
        return STATE["modality_weights"]  # Default weights
    
    # Normalize confidence scores
    normalized = {k: v/total for k, v in confidence_scores.items()}
    
    # Blend with default weights
    return {
        mod: 0.7 * STATE["modality_weights"][mod] + 0.3 * normalized.get(mod, 0)
        for mod in STATE["modality_weights"]
    }
```

### 8. Handling Edge Cases

```python
def handle_edge_cases(emotion_data):
    # If only one modality is active
    active_count = sum(1 for data in emotion_data.values() if data["active"])
    
    if active_count == 1:
        active_modality = next(k for k, v in emotion_data.items() if v["active"])
        
        # If only facial is active, increase its weight
        if active_modality == "facial":
            return adjust_weights({active_modality: 0.8})
            
    return STATE["modality_weights"]
```

This sophisticated system ensures that even when some inputs are silent or inactive, the application can still make reasonable inferences about the user's emotional state while avoiding false positives.