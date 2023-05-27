from flask import Flask, render_template, request, jsonify, redirect, Response
import os
import uuid

# -----
import cv2
import warnings
warnings.filterwarnings('ignore')

def generate_frames(filepath):
    # OpenCV video capture object
    cap = cv2.VideoCapture('uploads/' + filepath)

    # Trained XML classifiers describe features of the object to be detected (cars)
    car_cascade = cv2.CascadeClassifier('model/cars.xml')

    car_count = 0

    while True:
        # Read frames from the video
        ret, frames = cap.read()

        if not ret:
            break

        # Convert the frame to gray scale
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        # Detect cars of different sizes in the frame
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)

        # Draw rectangles around the detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
            car_count += 1

        # Encode the frame as JPEG format
        ret, buffer = cv2.imencode('.jpg', frames)

        if not ret:
            break

        # Yield the buffer as a byte string to be sent to the client
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n');

    cap.release()
# ------

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file from the request
    uploaded_file = request.files['video']

    # Generate a unique ID for the file
    unique_id = str(uuid.uuid4())

    # Get the file extension
    file_extension = os.path.splitext(uploaded_file.filename)[1]

    # Create a new filename with the unique ID and original extension
    new_filename = unique_id + file_extension

    # Save the uploaded file to a specific folder with the new filename
    uploaded_file.save('uploads/' + new_filename)

    # Prepare the JSON response
    response = {
        'status': 'success',
        'filename': new_filename
    }

    redirection_path = 'http://127.0.0.1:5000/process/' + new_filename

    # Return the JSON response
    # return redirect(redirection_path, code=200)
    return render_template('video_stream.html', server_url=redirection_path)

@app.route('/process/<filename>')
def process_video(filename):
    return Response(generate_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)