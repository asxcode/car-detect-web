from flask import Flask, render_template, request, jsonify, redirect, Response
import os
import uuid
from PIL import Image as im
import base64
import io
import cv2
import warnings
warnings.filterwarnings('ignore')


def pil_image_to_base64(image):
    # Create an in-memory byte stream
    byte_stream = io.BytesIO()
    
    # Save the PIL Image to the byte stream in JPEG format
    image.save(byte_stream, format='JPEG')
    
    # Rewind the stream to the beginning
    byte_stream.seek(0)
    
    # Convert the image to a Base64 encoded string
    base64_string = base64.b64encode(byte_stream.read()).decode('utf-8')
    
    return base64_string


def is_video(filename):
    video_extensions = ['.mp4', '.mov', '.avi']  # Add more video extensions if needed
    file_extension = os.path.splitext(filename)[1]
    return file_extension.lower() in video_extensions


def generate_frames(filepath):
    cap = cv2.VideoCapture('uploads/' + filepath)

    # Trained XML classifiers describe features of the object to be detected (cars)
    car_cascade = cv2.CascadeClassifier('model/cars.xml')


    while True:
        car_count = 0
        
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
        
        print("Car count: ", car_count, end='\r')

        # Yield the buffer as a byte string to be sent to the client
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n');

    cap.release()


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['upfile']
    
    # Generate a unique ID for the file
    unique_id = str(uuid.uuid4())
    
    file_extension = os.path.splitext(uploaded_file.filename)[1]
    
    # Create a new filename with the unique ID and original extension
    new_filename = unique_id + file_extension

    # Save the uploaded file to a specific folder with the new filename
    uploaded_file.save('uploads/' + new_filename)

    response = {
        'status': 'success',
        'filename': new_filename
    }

    if(is_video(new_filename)):
        redirection_path = 'http://127.0.0.1:5000/process-video/' + new_filename
        return render_template('video_stream.html', server_url=redirection_path)
    else:
        print('image file')
        redirection_path = 'http://127.0.0.1:5000/process-image/' + new_filename
        return redirect(redirection_path)


@app.route('/process-video/<filename>')
def process_video(filename):
    return Response(generate_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process-image/<filename>')
def process_image(filename):
    # Read the image file
    frame = cv2.imread('uploads/' + filename)

    # Trained XML classifiers describe features of the object to be detected (cars)
    car_cascade = cv2.CascadeClassifier('model/cars.xml')

    car_count = 0

    # Convert the frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars of different sizes in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Draw rectangles around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        car_count += 1

    # Encode the frame as JPEG format
    # Prepare the data for HTML rendering
    image_data = pil_image_to_base64(im.fromarray(frame))

    return render_template('image_process.html', image_data=image_data, car_count=car_count)


if __name__ == '__main__':
    app.run(debug=True)