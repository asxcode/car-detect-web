import torch
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import cv2
import uuid
import warnings
warnings.filterwarnings('ignore')

print("Loading YOLOv5s6 model...")
# Load the YOLOv5s6 model
model = torch.hub.load('model/ultralytics-yolov5-5eb7f7d/', 'custom', path='./model/yolov5s6.pt', source='local')

# Set the device to 'cuda' if available, otherwise use 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# Creating the flask app
app = Flask(__name__)

# Creating an API object
api = Api(app)

# Define the class labels for COCO dataset
class_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]    


class VehicleCount(Resource):
    def get(self):
        vehicle_counts = {
            'bicycle': 0,
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0,
        }

        try:
            # Get the base64 encoded image from the request
            image_data = request.json['image']

            # Decode the base64 image
            image_bytes = base64.b64decode(image_data)

            # open the image using PIL
            image = Image.open(BytesIO(image_bytes))
        except:
            response_obj = {
                'error': '"image" key in payload: please provide base64 formatted value.',
            }
            return response_obj, 415

        # Perform object detection on the image
        results = model(image)

        # Get the indices of all vehicle detections
        vehicle_indices = np.isin(results.pred[0][:, -1].detach().cpu().numpy(), [1, 2, 3, 4, 5, 6, 7])

        # Filter out the vehicle detections
        vehicle_results = results.pred[0][vehicle_indices]

        # Map class_index with labels, and fill vehicle_counts
        for vehicle in vehicle_results:
            class_index = int(vehicle[-1].detach().cpu().numpy())
            label = class_labels[class_index]
            vehicle_counts[label] += 1

        response = {
            'data': vehicle_counts
        }

        return response, 200


@app.route('/')
def index():
    return render_template('index.html')


# Define API route
api.add_resource(VehicleCount, '/api/vehicle-count')


@app.route('/api-reference')
def api_reference():
    return render_template('api_docs.html')


@app.route('/vehicle-count', methods=['POST'])
def vehicle_count_page():
        vehicle_counts = {
        'bicycle': 0,
        'car': 0,
        'motorcycle': 0,
        'bus': 0,
        'truck': 0,
        }

        #read image file string data
        filestr = request.files['image'].read()
        #convert string data to numpy array
        file_bytes = np.fromstring(filestr, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        # Perform object detection on the image
        results = model(img)

        # Get the indices of all vehicle detections
        vehicle_indices = np.isin(results.pred[0][:, -1].detach().cpu().numpy(), [1, 2, 3, 4, 5, 6, 7])

        # Filter out the vehicle detections
        vehicle_results = results.pred[0][vehicle_indices]

        # Map class_index with labels, and fill vehicle_counts
        for vehicle in vehicle_results:
            class_index = int(vehicle[-1].detach().cpu().numpy())
            label = class_labels[class_index]
            vehicle_counts[label] += 1
            bbox = vehicle[:4].detach().cpu().numpy()
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        unique_filename = str(uuid.uuid4())

        cv2.imwrite(f'static/{unique_filename}.jpg', img)

        return render_template('result.html', image=unique_filename, response=vehicle_counts)

if __name__ == '__main__':
    app.run()