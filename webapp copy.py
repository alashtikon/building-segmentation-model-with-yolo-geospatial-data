from flask import Flask, render_template, request, send_from_directory, Response
import os
import cv2
import argparse
import time
from ultralytics import YOLO

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            if f.filename is not None:
                basepath = os.path.dirname(__file__)
                filepath = os.path.join(basepath, 'uploads', f.filename)
                print("Upload folder is ", filepath)
                f.save(filepath)
                
                file_extension = f.filename.rsplit('.', 1)[1].lower()

                if file_extension in ['jpg', 'tiff', 'png']:
                    img = cv2.imread(filepath)

                    # Perform the detection
                    model = YOLO('best.pt')  # Load model for processing space images
                    detections = model(img, save=True, show =True)
                    return display(f.filename)
                elif file_extension == 'mp4':
                    video_path = filepath
                    cap = cv2.VideoCapture(video_path)

                    # Get video dimensions
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Define the codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
                    
                    # Initialize the YOLOv9 model
                    model = YOLO('yolov9c.pt')
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break                                                      

                        results = model(frame, save=True)
                        print(results)
                        cv2.waitKey(1)

                        res_plotted = results[0].plot()
                        cv2.imshow("result", res_plotted)
                        
                        # Write the frame to the output video
                        out.write(res_plotted)

                        if cv2.waitKey(1) == ord('q'):
                            break

                    return video_feed()            

    return "Filename is None"

@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    if subfolders:
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
        directory = os.path.join(folder_path, latest_subfolder)    
        print("Printing directory:", directory) 
        files = os.listdir(directory)
        latest_file = files[0]
        filename = os.path.join(directory, latest_file)

        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension in ['jpg', 'tiff', 'png']:     
            return send_from_directory(directory, latest_file, request.environ)  # Show the result in a separate tab

    return "Invalid file format"

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)  # Detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image) 
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  # Control the frame rate to display one frame every 100 milliseconds

@app.route("/video_feed")
def video_feed():
    print("Function called")

    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv9 models")
    parser.add_argument("--port", default=5000, type=int, help="Port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
