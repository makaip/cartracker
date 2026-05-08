import json
import os
import uuid as uuid_lib
import shutil

from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename

from retrieve import generate_frames
import database


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

database.init_db()

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
    vehicles = database.get_vehicles()
    return render_template('index.html', cameras=sorted_cameras, vehicles=vehicles)

@app.route('/video_feed')
def video_feed() -> Response:
    uuid = request.args.get('uuid')
    if not uuid:
        return "No UUID provided", 400
    
    return Response(generate_frames(uuid),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/embedding_feed')
def embedding_feed() -> Response:
    # Placeholder for embedding feed route


    return Response()

@app.route('/add_vehicle', methods=['POST'])
def add_vehicle():
    files = request.files.getlist('pictures')
    if not files or all(f.filename == '' for f in files):
        return "No pictures uploaded", 400
    
    vehicle_uuid = str(uuid_lib.uuid4())
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], vehicle_uuid)
    os.makedirs(upload_path, exist_ok=True)
    
    for f in files:
        if f.filename:
            filename = secure_filename(f.filename)
            f.save(os.path.join(upload_path, filename))
            
    database.add_vehicle(vehicle_uuid)
    
    return redirect(url_for('index'))

@app.route('/delete_vehicle', methods=['POST'])
def delete_vehicle():
    vehicle_uuid = request.form.get('uuid')
    if not vehicle_uuid:
        return "No UUID provided", 400
        
    database.delete_vehicle(vehicle_uuid)
    
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], vehicle_uuid)
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path)
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)