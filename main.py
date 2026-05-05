from flask import Flask, render_template, Response, request
import json
import os
from retrieve import generate_frames

app = Flask(__name__)

def load_cameras() -> dict:
    try:
        with open('traffic_cameras.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@app.route('/')
def index() -> str:
    cameras = load_cameras()
    sorted_cameras = dict(sorted(cameras.items(), key=lambda item: item[1]))
    return render_template('index.html', cameras=sorted_cameras)

@app.route('/video_feed')
def video_feed() -> Response:
    uuid = request.args.get('uuid')
    if not uuid:
        return "No UUID provided", 400
    
    return Response(generate_frames(uuid),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)