from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import json
import os
import threading
import time
from modules.emotion_combiner import get_combined_emotion, update_detectors

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
camera_enabled = False
microphone_enabled = False

# Pydantic model for AI Tutor requests
class TopicRequest(BaseModel):
    topic: str

def get_latest_emotion_data():
    """Get the latest emotion data from the JSON log file"""
    try:
        json_file = "emotion_log.json"
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
                if data:
                    return data[-1]  # Return the latest entry
        return None
    except Exception as e:
        print(f"Error reading emotion log: {e}")
        return None

def start_emotion_detection():
    """Start continuous emotion detection in a separate thread"""
    global emotion_detection_running, emotion_thread
    
    if not emotion_detection_running:
        emotion_detection_running = True
        
        def emotion_detection_loop():
            while emotion_detection_running:
                try:
                    # Update detectors based on current sensor states
                    update_detectors(camera_enabled, microphone_enabled)
                    
                    # Only run detection if at least one sensor is enabled
                    if camera_enabled or microphone_enabled:
                        # Get combined emotion (this updates the JSON file)
                        emotion = get_combined_emotion()
                        print(f"Detected emotion: {emotion} (Camera: {camera_enabled}, Mic: {microphone_enabled})")
                    else:
                        # If no sensors are enabled, just sleep
                        print("No sensors enabled, skipping emotion detection")
                    
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    print(f"Error in emotion detection loop: {e}")
                    time.sleep(5)
        
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
    """Start emotion detection when the server starts"""
    # Don't start detection automatically - wait for frontend to enable sensors
    print("Server started - waiting for frontend to enable sensors")

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
    latest_data = get_latest_emotion_data()
    if latest_data:
        return latest_data
    return {
        "time": "",
        "facial_emotion": "Unknown",
        "voice_emotion": "Unknown", 
        "interaction_emotion": "Unknown",
        "final_emotion": "Unknown",
        "value": 0.0
    }

@app.get("/api/emotion/interaction")
def get_interaction_emotion():
    """Get emotion based on interaction only"""
    latest_data = get_latest_emotion_data()
    if latest_data:
        return {"emotion": latest_data.get("interaction_emotion", "Unknown")}
    return {"emotion": "Unknown"}

@app.get("/api/emotion/facial")
def get_facial_emotion():
    """Get emotion based on facial detection only"""
    latest_data = get_latest_emotion_data()
    if latest_data:
        return {"emotion": latest_data.get("facial_emotion", "Unknown")}
    return {"emotion": "Unknown"}

@app.get("/api/emotion/voice")
def get_voice_emotion():
    """Get emotion based on voice detection only"""
    latest_data = get_latest_emotion_data()
    if latest_data:
        return {"emotion": latest_data.get("voice_emotion", "Unknown")}
    return {"emotion": "Unknown"}

@app.get("/api/emotion/combined")
def get_combined_emotion_endpoint():
    """Get the combined/final emotion"""
    latest_data = get_latest_emotion_data()
    if latest_data:
        return {"emotion": latest_data.get("final_emotion", "Unknown")}
    return {"emotion": "Unknown"}

@app.post("/api/sensors/camera")
def toggle_camera():
    """Toggle camera state"""
    global camera_enabled, emotion_detection_running
    camera_enabled = not camera_enabled
    
    print(f"🔄 Camera toggled: {camera_enabled}")
    print(f"📊 Current state - Camera: {camera_enabled}, Mic: {microphone_enabled}, Detection: {emotion_detection_running}")
    
    # Update detectors based on new camera state
    update_detectors(camera_enabled, microphone_enabled)
    
    # Start emotion detection if any sensor is enabled
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
        "detection_running": emotion_detection_running
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
        "detection_running": emotion_detection_running
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
    try:
        json_file = "emotion_log.json"
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
                return data
        return []
    except Exception as e:
        print(f"Error reading all emotion data: {e}")
        return []

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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

