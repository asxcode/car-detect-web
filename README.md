# Car Detect Web

Car Detect Web is a web application and REST API that provides vehicle detection for any given image. It uses the YOLOv5s6 pretrained model from PyTorch for efficient and accurate vehicle detection. The web application allows users to upload an image and view the localized objects along with the count of different types of vehicles detected in the image. The REST API enables programmatic access to the vehicle detection system by sending images and receiving JSON responses with vehicle counts.

## Table of Contents
- [Demo](#demo)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
    - [Web Application](#web-application)
    - [REST API](#rest-api)
- [Acknowledgements](#acknowledgements)

## Demo
![](https://github.com/arv1nd-s/car-detect-web/blob/web-app/docs/webapp-demo.png)

## Getting Started

### Prerequisites

* Python 3.6 or higher
* Other dependencies as specified in requirements.txt

### Installation

1. Clone the repository:

```bash
git clone https://github.com/arv1nd-s/car-detect-web.git
cd car-detect-web
```
2. Install the required dependencies:
`pip install -r requirements.txt`

## Usage

### Web Application
To run the web application, execute the following command:
`python3 app.py`

Once the application is running, open your web browser and go to `http://localhost:5000` to access the web interface. Upload an image, and the application will display the localized obects with vehicle counts.

### REST API
To programmatically access the vehicle detection functionality, you can use the REST API. API documentation is available at `http://localhost:5000/api-reference`, where you can find details on the API endpoint, expected input format, and the JSON response format.

## Acknowledgements
https://pytorch.org/hub/ultralytics_yolov5/

https://docs.opencv.org/4.x/d9/df8/tutorial_root.html

https://flask.palletsprojects.com/en/2.3.x/

https://flask-restful.readthedocs.io/en/latest/quickstart.html#a-minimal-api

https://getbootstrap.com/docs/5.3/getting-started/introduction/
