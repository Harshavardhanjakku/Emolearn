from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import json
import csv
import os
import threading
import time
from modules.emotion_combiner import (
    get_combined_emotion,
    update_detectors,
    start_detector,
    detectors_running,
    merge_multimodal_emotion,
    snapshot_combined_emotion,
)
from modules.facial_emotion import (
    get_facial_emotion as live_facial_emotion,
    get_facial_emotion_record as live_facial_record,
    set_camera_session_allowed,
    list_available_cameras,
    set_camera_index,
    get_camera_index,
)
from modules.speech_emotion import get_speech_emotion as live_speech_emotion
from modules.mouse_emotion import get_mouse_emotion as live_mouse_emotion, has_mouse_hook_input
from modules.emotion_log_store import load_entries as _load_emotion_log_list

app = FastAPI()

# Allow frontend on localhost:5173 to access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Optional: replace with ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global flags to track sensor states and emotion detection
emotion_detection_running = False
emotion_thread = None
camera_enabled = True
microphone_enabled = False

# Pydantic model for AI Tutor requests
class TopicRequest(BaseModel):
    topic: str


class SensorToggleRequest(BaseModel):
    enabled: bool | None = None


class CameraSelectRequest(BaseModel):
    index: int

def get_latest_emotion_data():
    """Get the latest emotion data from the JSON log file"""
    data = _load_emotion_log_list()
    return data[-1] if data else None


def _live_facial_safe():
    if not camera_enabled or not detectors_running.get("facial"):
        return None
    try:
        return live_facial_emotion()
    except Exception as e:
        print(f"Live facial read failed: {e}")
        return None


def _live_voice_safe():
    if not microphone_enabled or not detectors_running.get("speech"):
        return None
    try:
        return live_speech_emotion()
    except Exception as e:
        print(f"Live voice read failed: {e}")
        return None


def _live_interaction_safe():
    if not detectors_running.get("mouse"):
        return None
    # Browser app does not call update_mouse_activity; ignore fake-idle "Sleepy".
    if not has_mouse_hook_input():
        return None
    try:
        return live_mouse_emotion()
    except Exception as e:
        print(f"Live interaction read failed: {e}")
        return None


def _live_combined_fallback():
    face = _live_facial_safe() or "Unknown"
    voice = _live_voice_safe() or "Unknown"
    mouse = _live_interaction_safe() or "Unknown"
    return merge_multimodal_emotion(face, voice, mouse)

def start_emotion_detection():
    """Start continuous emotion detection in a separate thread"""
    global emotion_detection_running, emotion_thread
    
    if not emotion_detection_running:
        emotion_detection_running = True
        
        def emotion_detection_loop():
            # Sensor POST handlers already call update_detectors.
            update_detectors(camera_enabled, microphone_enabled)
            while emotion_detection_running:
                try:
                    if camera_enabled or microphone_enabled:
                        get_combined_emotion()
                    time.sleep(1.0)
                except Exception as e:
                    print(f"Error in emotion detection loop: {e}")
                    time.sleep(1.0)
        
        emotion_thread = threading.Thread(target=emotion_detection_loop, daemon=True)
        emotion_thread.start()
        print("Emotion detection started in background")

def stop_emotion_detection():
    """Stop continuous emotion detection"""
    global emotion_detection_running
    emotion_detection_running = False
    print("Emotion detection stopped")

@app.on_event("startup")
async def startup_event():
    """Turn the camera on at boot and start facial detection."""
    global camera_enabled
    camera_enabled = True
    set_camera_session_allowed(True)
    update_detectors(True, microphone_enabled)
    start_emotion_detection()
    print("Server started - camera on, emotion detection running")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop emotion detection when the server shuts down"""
    stop_emotion_detection()

@app.get("/api/emotion")
def get_emotion():
    """Get the latest combined emotion (triggers detection if needed)"""
    emotion = get_combined_emotion()
    return {"emotion": emotion}

@app.get("/api/emotion/latest")
def get_latest_emotion():
    """Get the latest emotion data with all fields"""
    snap = snapshot_combined_emotion()
    rec = live_facial_record() if camera_enabled else {"time": ""}
    if rec.get("time"):
        snap["time"] = rec["time"]
    elif not snap.get("time"):
        latest_data = get_latest_emotion_data()
        if latest_data:
            return latest_data
    return snap

@app.get("/api/emotion/interaction")
def get_interaction_emotion():
    """Get emotion based on interaction only"""
    live = _live_interaction_safe()
    if live:
        return {"emotion": live}
    latest_data = get_latest_emotion_data()
    if latest_data:
        return {"emotion": latest_data.get("interaction_emotion", "Unknown")}
    return {"emotion": "Unknown"}

@app.get("/api/emotion/facial")
def get_facial_emotion():
    """Get emotion based on facial detection only"""
    live = _live_facial_safe()
    if live:
        return {"emotion": live}
    latest_data = get_latest_emotion_data()
    if latest_data:
        return {"emotion": latest_data.get("facial_emotion", "Unknown")}
    return {"emotion": "Unknown"}

@app.get("/api/emotion/voice")
def get_voice_emotion():
    """Get emotion based on voice detection only"""
    live = _live_voice_safe()
    if live:
        return {"emotion": live}
    latest_data = get_latest_emotion_data()
    if latest_data:
        return {"emotion": latest_data.get("voice_emotion", "Unknown")}
    return {"emotion": "Unknown"}

@app.get("/api/emotion/combined")
def get_combined_emotion_endpoint():
    """Live combined emotion (same merge as get_combined_emotion, without writing the log)."""
    if camera_enabled or microphone_enabled:
        snap = snapshot_combined_emotion()
        return {"emotion": snap["final_emotion"]}
    latest_data = get_latest_emotion_data()
    if latest_data:
        return {"emotion": latest_data.get("final_emotion", "Unknown")}
    return {"emotion": "Unknown"}

@app.post("/api/sensors/camera")
def toggle_camera(req: SensorToggleRequest = SensorToggleRequest()):
    """Set camera on/off from the requested enabled flag (does not flip blindly)."""
    global camera_enabled, emotion_detection_running
    if req is not None and req.enabled is not None:
        camera_enabled = bool(req.enabled)
    else:
        camera_enabled = not camera_enabled
    
    print(f"🔄 Camera set: {camera_enabled}")
    print(f"📊 Current state - Camera: {camera_enabled}, Mic: {microphone_enabled}, Detection: {emotion_detection_running}")
    
    update_detectors(camera_enabled, microphone_enabled)
    
    if (camera_enabled or microphone_enabled) and not emotion_detection_running:
        print("🚀 Starting emotion detection...")
        start_emotion_detection()
    elif not camera_enabled and not microphone_enabled and emotion_detection_running:
        print("🛑 Stopping emotion detection...")
        stop_emotion_detection()
    
    print(f"✅ Final state - Camera: {camera_enabled}, Mic: {microphone_enabled}, Detection: {emotion_detection_running}")
    
    return {
        "camera_enabled": camera_enabled,
        "microphone_enabled": microphone_enabled,
        "detection_running": emotion_detection_running,
        "camera_index": get_camera_index(),
    }


@app.get("/api/sensors/cameras")
def get_cameras():
    """List built-in and external webcams the backend can open."""
    return {
        "cameras": list_available_cameras(),
        "selected_index": get_camera_index(),
        "camera_enabled": camera_enabled,
    }


@app.post("/api/sensors/camera/select")
def select_camera(req: CameraSelectRequest):
    """Use a specific webcam index (0 = default/built-in, 1+ = often USB/external)."""
    global camera_enabled, emotion_detection_running
    chosen = set_camera_index(req.index)
    if not camera_enabled:
        camera_enabled = True
        set_camera_session_allowed(True)
        update_detectors(True, microphone_enabled)
        if not emotion_detection_running:
            start_emotion_detection()
    return {
        "camera_index": chosen,
        "camera_enabled": camera_enabled,
        "cameras": list_available_cameras(),
    }

@app.post("/api/sensors/microphone")
def toggle_microphone():
    """Toggle microphone state"""
    global microphone_enabled, emotion_detection_running
    microphone_enabled = not microphone_enabled
    
    print(f"Microphone toggled: {microphone_enabled}")
    
    # Update detectors based on new microphone state
    update_detectors(camera_enabled, microphone_enabled)
    
    # Start emotion detection if any sensor is enabled
    if (camera_enabled or microphone_enabled) and not emotion_detection_running:
        start_emotion_detection()
    elif not camera_enabled and not microphone_enabled and emotion_detection_running:
        stop_emotion_detection()
    
    return {
        "camera_enabled": camera_enabled,
        "microphone_enabled": microphone_enabled,
        "detection_running": emotion_detection_running
    }

@app.get("/api/sensors/status")
def get_sensor_status():
    """Get current sensor and detection status"""
    return {
        "camera_enabled": camera_enabled,
        "microphone_enabled": microphone_enabled,
        "detection_running": emotion_detection_running,
        "camera_index": get_camera_index(),
    }

@app.post("/api/emotion/start")
def start_detection():
    """Manually start emotion detection"""
    start_emotion_detection()
    return {"message": "Emotion detection started"}

@app.post("/api/emotion/stop")
def stop_detection():
    """Manually stop emotion detection"""
    stop_emotion_detection()
    return {"message": "Emotion detection stopped"}

@app.get("/api/emotion/status")
def get_detection_status():
    """Get the status of emotion detection"""
    return {"running": emotion_detection_running}

@app.get("/api/emotion/all")
def get_all_emotions():
    """Get all emotion data from the JSON file"""
    return _load_emotion_log_list()


@app.get("/api/emotion/history")
def get_emotion_history():
    """Return normalized JSON and CSV history with exact duplicate events merged."""
    records = []
    json_path = os.path.join(os.path.dirname(__file__), "emotion_log.json")
    csv_path = os.path.join(os.path.dirname(__file__), "emotion_log.csv")

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            json_entries = json.load(file)
    except (OSError, json.JSONDecodeError):
        json_entries = []

    for entry in json_entries if isinstance(json_entries, list) else []:
        emotion = entry.get("final_emotion") or "Unknown"
        timestamp = entry.get("time")
        if timestamp:
            records.append({
                "time": timestamp,
                "emotion": emotion,
                "value": entry.get("value"),
                "source": "json",
                "facial_emotion": entry.get("facial_emotion"),
                "voice_emotion": entry.get("voice_emotion"),
                "interaction_emotion": entry.get("interaction_emotion"),
            })

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as file:
            for entry in csv.DictReader(file):
                if entry.get("timestamp") and entry.get("emotion"):
                    records.append({
                        "time": entry["timestamp"],
                        "emotion": entry["emotion"] or "Unknown",
                        "value": None,
                        "source": f"csv:{entry.get('source') or 'unknown'}",
                        "facial_emotion": None,
                        "voice_emotion": None,
                        "interaction_emotion": None,
                    })
    except OSError:
        pass

    # JSON and CSV can record the same detector event. Use normalized timestamp
    # plus final emotion as the event identity and retain both source labels.
    merged = {}
    for record in records:
        timestamp = str(record["time"]).replace("T", " ")[:19]
        key = (timestamp, record["emotion"])
        if key not in merged:
            merged[key] = {**record, "source": [record["source"]]}
            continue
        current = merged[key]
        if record["source"] not in current["source"]:
            current["source"].append(record["source"])
        if current.get("value") is None and record.get("value") is not None:
            current["value"] = record["value"]

    return sorted(merged.values(), key=lambda item: item["time"])

@app.get("/games/{emotion}/{game_name}")
def serve_game(emotion: str, game_name: str):
    """Serve simple HTML games or a Coming Soon page."""
    games = {
        "emoji-doodle": {
            "title": "Emoji Doodle 🎨",
            "html": '''
                <h2>Emoji Doodle 🎨</h2>
                <p>Draw with your mouse and add emojis!</p>
                <canvas id="canvas" width="400" height="200" style="border:1px solid #ccc;"></canvas><br>
                <button onclick="addEmoji('😀')">😀</button>
                <button onclick="addEmoji('🌟')">🌟</button>
                <button onclick="addEmoji('🎈')">🎈</button>
                <button onclick="clearCanvas()">Clear</button>
                <script>
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                let drawing = false;
                canvas.onmousedown = () => drawing = true;
                canvas.onmouseup = () => drawing = false;
                canvas.onmousemove = e => {
                  if (!drawing) return;
                  const rect = canvas.getBoundingClientRect();
                  ctx.fillRect(e.clientX-rect.left, e.clientY-rect.top, 2, 2);
                };
                function addEmoji(emoji) {
                  ctx.font = '2rem serif';
                  ctx.fillText(emoji, Math.random()*350, Math.random()*150+30);
                }
                function clearCanvas() { ctx.clearRect(0,0,400,200); }
                </script>
            '''
        },
        "line-shuffle": {
            "title": "Line Shuffle 🔀",
            "html": '''
                <h2>Line Shuffle 🔀</h2>
                <p>Arrange the code lines in the correct order!</p>
                <div id="lines"></div>
                <button onclick="checkOrder()">Check Order</button>
                <script>
                const correct = [
                  'function greet(name) {',
                  '  return Hello, ${name}!;',
                  '}',
                  'console.log(greet(\'World\'));'
                ];
                let lines = [...correct].sort(()=>Math.random()-0.5);
                function render() {
                  document.getElementById('lines').innerHTML = lines.map((l,i)=><div style='padding:4px;border:1px solid #ccc;margin:2px;cursor:pointer' onclick='move(${i})'>${l}</div>).join('');
                }
                window.move = i => { if(i>0){ [lines[i],lines[i-1]]=[lines[i-1],lines[i]]; render(); } };
                window.checkOrder = () => alert(JSON.stringify(lines)===JSON.stringify(correct)?'Correct!':'Try again!');
                render();
                </script>
            '''
        },
        "quick-match": {
            "title": "Quick Match ⚡",
            "html": '''
                <h2>Quick Match ⚡</h2>
                <p>Click matching pairs!</p>
                <div id="grid"></div>
                <script>
                const symbols = ['🍎','🍌','🍇','🍊','🍉','🍒'];
                let cards = [...symbols,...symbols].sort(()=>Math.random()-0.5);
                let flipped = [], matched = [];
                function render() {
                  document.getElementById('grid').innerHTML = cards.map((s,i)=><button style='width:40px;height:40px;font-size:2rem;margin:2px' onclick='flip(${i})' ${matched.includes(i)?'disabled':''}>${flipped.includes(i)||matched.includes(i)?s:'❓'}</button>).join('');
                }
                window.flip = i => {
                  if(flipped.length<2 && !flipped.includes(i) && !matched.includes(i)){
                    flipped.push(i); render();
                    if(flipped.length===2){
                      setTimeout(()=>{
                        if(cards[flipped[0]]===cards[flipped[1]]) matched.push(...flipped);
                        flipped=[]; render();
                      },700);
                    }
                  }
                };
                render();
                </script>
            '''
        },
        "smash-the-bug": {
            "title": "Smash The Bug",
            "html": '''
                <h2>Smash The Bug 🐛</h2>
                <p>Click the bug as fast as you can!</p>
                <div id="area" style="width:300px;height:200px;position:relative;background:#eee;"></div>
                <div>Score: <span id="score">0</span></div>
                <button onclick="startGame()">Start</button>
                <script>
                let score=0, bug, running=false;
                function startGame(){
                  score=0; document.getElementById('score').textContent=score; running=true; spawn();
                }
                function spawn(){
                  if(!running) return;
                  const area=document.getElementById('area');
                  area.innerHTML='';
                  bug=document.createElement('div');
                  bug.textContent='🐛';
                  bug.style.position='absolute';
                  bug.style.left=Math.random()*260+'px';
                  bug.style.top=Math.random()*160+'px';
                  bug.style.fontSize='2rem';
                  bug.style.cursor='pointer';
                  bug.onclick=()=>{ score++; document.getElementById('score').textContent=score; spawn(); };
                  area.appendChild(bug);
                  setTimeout(()=>{ if(area.contains(bug)){ running=false; area.innerHTML='<b>Game Over!</b>'; } },1200);
                }
                </script>
            '''
        }
    }
    game = games.get(game_name, None)
    if not game:
        html = f"""
        <h2>Coming Soon</h2>
        <p>The game <b>{game_name}</b> is not available yet.</p>
        <a href="javascript:history.back()">&larr; Back to Games</a>
        """
        return HTMLResponse(content=html)
    html = f"""
    <html><head><title>{game['title']}</title></head>
    <body style='font-family:sans-serif;background:#fafbfc;padding:2em;'>
    <a href="javascript:history.back()" style="position:fixed;top:10px;left:10px;text-decoration:none;font-size:1.2em">&larr; Back</a>
    {game['html']}
    </body></html>
    """
    return HTMLResponse(content=html)

@app.post("/explain")
def explain_topic(request: TopicRequest):
    """Generate AI explanation for a given topic"""
    topic = request.topic.lower()
    
    # Pre-defined explanations for common topics
    explanations = {
        "javascript": {
            "title": "JavaScript Fundamentals",
            "overview": "JavaScript is a versatile programming language that powers the modern web. It's used for creating interactive websites, web applications, and even server-side applications with Node.js.",
            "keyConcepts": [
                "Variables and Data Types (let, const, var)",
                "Functions and Scope",
                "Objects and Arrays",
                "DOM Manipulation",
                "Asynchronous Programming (Promises, async/await)",
                "ES6+ Features (Arrow functions, destructuring, modules)"
            ],
            "examples": [
                {
                    "title": "Basic Function",
                    "code": "function greet(name) {\n  return Hello, ${name}!;\n}\n\nconsole.log(greet('World'));"
                },
                {
                    "title": "Arrow Function",
                    "code": "const multiply = (a, b) => a * b;\n\nconsole.log(multiply(5, 3)); // 15"
                },
                {
                    "title": "Array Methods",
                    "code": "const numbers = [1, 2, 3, 4, 5];\nconst doubled = numbers.map(n => n * 2);\nconst sum = numbers.reduce((acc, n) => acc + n, 0);"
                }
            ],
            "learningPath": [
                "Learn basic syntax and data types",
                "Understand functions and scope",
                "Master object-oriented programming",
                "Learn DOM manipulation",
                "Study asynchronous programming",
                "Explore modern ES6+ features",
                "Practice with real projects"
            ]
        },
        "react": {
            "title": "React Framework",
            "overview": "React is a popular JavaScript library for building user interfaces. It uses a component-based architecture and virtual DOM for efficient rendering.",
            "keyConcepts": [
                "Components and Props",
                "State and Lifecycle",
                "Hooks (useState, useEffect, useContext)",
                "Virtual DOM",
                "JSX Syntax",
                "Component Composition",
                "Event Handling"
            ],
            "examples": [
                {
                    "title": "Functional Component",
                    "code": "import React, { useState } from 'react';\n\nfunction Counter() {\n  const [count, setCount] = useState(0);\n  \n  return (\n    <div>\n      <p>Count: {count}</p>\n      <button onClick={() => setCount(count + 1)}>\n        Increment\n      </button>\n    </div>\n  );\n}"
                },
                {
                    "title": "useEffect Hook",
                    "code": "useEffect(() => {\n  document.title = Count: ${count};\n  \n  return () => {\n    // Cleanup function\n    document.title = 'React App';\n  };\n}, [count]);"
                }
            ],
            "learningPath": [
                "Learn JSX syntax",
                "Understand components and props",
                "Master state management",
                "Learn React hooks",
                "Study component lifecycle",
                "Practice with small projects",
                "Build larger applications"
            ]
        },
        "python": {
            "title": "Python Programming",
            "overview": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation.",
            "keyConcepts": [
                "Variables and Data Types",
                "Control Structures (if, loops)",
                "Functions and Modules",
                "Object-Oriented Programming",
                "File Handling",
                "Exception Handling",
                "List Comprehensions"
            ],
            "examples": [
                {
                    "title": "Basic Function",
                    "code": "def greet(name):\n    return f\"Hello, {name}!\"\n\nprint(greet(\"World\"))"
                },
                {
                    "title": "List Comprehension",
                    "code": "numbers = [1, 2, 3, 4, 5]\nsquares = [n**2 for n in numbers]\neven_numbers = [n for n in numbers if n % 2 == 0]"
                },
                {
                    "title": "Class Example",
                    "code": "class Person:\n    def _init_(self, name, age):\n        self.name = name\n        self.age = age\n    \n    def introduce(self):\n        return f\"Hi, I'm {self.name} and I'm {self.age} years old.\""
                }
            ],
            "learningPath": [
                "Learn basic syntax and data types",
                "Understand control structures",
                "Master functions and modules",
                "Learn object-oriented programming",
                "Study file handling and exceptions",
                "Explore advanced features",
                "Practice with real projects"
            ]
        },
        "machine learning": {
            "title": "Machine Learning Fundamentals",
            "overview": "Machine Learning is a subset of artificial intelligence that enables computers to learn and make predictions from data without being explicitly programmed.",
            "keyConcepts": [
                "Supervised vs Unsupervised Learning",
                "Training and Testing Data",
                "Feature Engineering",
                "Model Evaluation Metrics",
                "Overfitting and Underfitting",
                "Cross-Validation",
                "Popular Algorithms (Linear Regression, Decision Trees, Neural Networks)"
            ],
            "examples": [
                {
                    "title": "Linear Regression with Python",
                    "code": "import numpy as np\nfrom sklearn.linear_model import LinearRegression\n\n# Sample data\nX = np.array([[1], [2], [3], [4], [5]])\ny = np.array([2, 4, 5, 4, 5])\n\n# Create and train model\nmodel = LinearRegression()\nmodel.fit(X, y)\n\n# Make prediction\nprediction = model.predict([[6]])"
                },
                {
                    "title": "Data Preprocessing",
                    "code": "from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\n\n# Split data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n\n# Scale features\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)"
                }
            ],
            "learningPath": [
                "Learn Python and data manipulation",
                "Understand statistics and probability",
                "Study linear algebra and calculus",
                "Learn data preprocessing techniques",
                "Master basic ML algorithms",
                "Practice with real datasets",
                "Explore advanced topics (Deep Learning, NLP)"
            ]
        }
    }
    
    # Check if we have a pre-defined explanation
    if topic in explanations:
        return explanations[topic]
    
    # For unknown topics, generate a generic response
    return {
        "title": f"{request.topic} - Learning Guide",
        "overview": f"{request.topic} is an interesting topic to learn about. This comprehensive guide will help you understand the fundamentals and advanced concepts.",
        "keyConcepts": [
            "Basic principles and fundamentals",
            "Core concepts and terminology",
            "Practical applications",
            "Best practices and methodologies",
            "Common challenges and solutions"
        ],
        "examples": [
            {
                "title": "Basic Example",
                "code": f"// Basic {request.topic} example\n// Add your code here\nconsole.log('Hello, {request.topic}!');"
            }
        ],
        "learningPath": [
            "Start with fundamentals",
            "Learn core concepts",
            "Practice with examples",
            "Build small projects",
            "Explore advanced topics",
            "Contribute to real-world applications"
        ]
    }

@app.get("/api/sensors/camera/status")
def get_camera_status():
    """Get detailed camera status including if it's being accessed"""
    try:
        from modules.facial_emotion import check_camera_availability
        # Never probe USB webcam when facial detection is disabled (otherwise LED flickers).
        if not camera_enabled:
            return {
                "camera_enabled": False,
                "camera_available": False,
                "detection_running": emotion_detection_running,
                "status": "Camera off — backend not opening the webcam",
            }
        camera_available = check_camera_availability()
        return {
            "camera_enabled": camera_enabled,
            "camera_available": camera_available,
            "detection_running": emotion_detection_running,
            "status": "Camera is accessible" if camera_available else "Camera not accessible"
        }
    except Exception as e:
        return {
            "camera_enabled": camera_enabled,
            "camera_available": False,
            "detection_running": emotion_detection_running,
            "status": f"Error checking camera: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
