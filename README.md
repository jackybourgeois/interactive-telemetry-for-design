![python-version](https://img.shields.io/badge/python-v3.12.8-blue)
![license](https://img.shields.io/badge/license-GPLv3-blue)
[![download](https://img.shields.io/badge/download-.zip-brightgreen)](https://github.com/Interactive-Telemetry-for-Design/interactive-telemetry-for-design/archive/refs/heads/main.zip)

# Interactive Telemetry for Design
This project aims to enable designers to engage in data-driven design through an active learning interface. IMU telemetry data (accelerometer and gyroscope) that is collected from the usage of the designer's prototype can be utilised to detect anomalies. This data can then give insights to designers to refine the prototype.

The model is built on a Long Short-Term Memory (LSTM) architecture, which excels at interpreting long-term dependencies in consumer usage patterns and identifying anomalies. The prototype is an invariant for this model: in theory any object, tool or appliance is compatible.

## Guidelines for Use
- **Quantitative Analysis**: Use this for tasks like stability testing and detecting unusual user behaviour or anomalies.
- **Qualitative Analysis**: Avoid using this for tasks where you need subjective feedback from prototype testers, such as asking, "Do you prefer this prototype over this product?" or "Would you use this product in your daily life and why or why not?"

## Getting Started
### Dependencies
- [Python 3.12.8](https://www.python.org/downloads/release/python-3128/)
- [TensorFlow NVIDIA software requirements](https://www.tensorflow.org/install/pip#software_requirements). For GPU acceleration on Linux, optional but recommended for better performance
- [Brave browser](https://brave.com/). From our testing, video playback works best in the Brave browser

### Installation
Clone the repository
```
git clone https://github.com/Interactive-Telemetry-for-Design/interactive-telemetry-for-design.git
cd interactive-telemetry-for-design
```
or download as a ZIP file.

#### Windows (CPU-based)
1. Create a virtual environment

    Using the Python launcher:
    ```
    py -3.12 -m venv .venv
    ```
    Using Python directly (ensure Python 3.12.8 is installed and available in your PATH):
    ```
    python -m venv .venv
    ```

1. Activate the environment

    Using Git Bash:
    ```
    source .venv/Scripts/activate
    ```

    Using CMD:
    ```
    .venv\Scripts\activate.bat
    ```

1. Install the dependencies

    ```
    pip install -r requirements_win32.txt
    ```

#### Linux (GPU acceleration)

1. Create a virtual environment
    ```
    python3 -m venv .venv
    ```

1. Activate the environment and install the dependencies
    ```
    source .venv/bin/activate
    pip install -r requirements_linux.txt
    ```

### Starting the Flask server
Copy the `.env-example` file and rename the copy to `.env`. This contains the environment variable settings for the Flask server. Optionally set a secret key.

To start the Flask server, ensure your virtual environment is activated and execute
```
flask run
```
and go to [http://localhost:5000](http://localhost:5000).

## Using The Interface
For now, only a resolution of 1920x1080 is officially supported.

### Input
You can provide data in two ways:
- Video file with IMU streams (e.g. GoPro .MP4 file containing video and gpmf stream)
- Video file + (pre-extracted) IMU telemetry data CSV file

### Supported File Formats
Using [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser), the following file formats are supported for IMU stream parsing and extraction:
- GoPro (HERO 5 and later)
- Sony (a1, a7c, a7r V, a7 IV, a7s III, a9 II, a9 III, FX3, FX6, FX9, RX0 II, RX100 VII, ZV1, ZV-E10, ZV-E10 II, ZV-E1, a6700)
- Insta360 (OneR, OneRS, SMO 4k, Go, GO2, GO3, GO3S, Caddx Peanut, Ace, Ace Pro)
- DJI (Avata, Avata 2, O3 Air Unit, Action 2/4/5, Neo)
- Blackmagic RAW (*.braw)
- RED RAW (V-Raptor, KOMODO) (*.r3d)
- Freefly (Ember)
- Betaflight blackbox (*.bfl, *.bbl, *.csv)
- ArduPilot logs (*.bin, *.log)
- Gyroflow [.gcsv log](https://docs.gyroflow.xyz/app/technical-details/gcsv-format)
- iOS apps: [`Sensor Logger`](https://apps.apple.com/us/app/sensor-logger/id1531582925), [`G-Field Recorder`](https://apps.apple.com/at/app/g-field-recorder/id1154585693), [`Gyro`](https://apps.apple.com/us/app/gyro-record-device-motion-data/id1161532981), [`GyroCam`](https://apps.apple.com/us/app/gyrocam-professional-camera/id1614296781)
- Android apps: [`Sensor Logger`](https://play.google.com/store/apps/details?id=com.kelvin.sensorapp&hl=de_AT&gl=US), [`Sensor Record`](https://play.google.com/store/apps/details?id=de.martingolpashin.sensor_record), [`OpenCamera Sensors`](https://github.com/MobileRoboticsSkoltech/OpenCamera-Sensors), [`MotionCam Pro`](https://play.google.com/store/apps/details?id=com.motioncam.pro)
- Runcam CSV (Runcam 5 Orange, iFlight GOCam GR, Runcam Thumb, Mobius Maxi 4K)
- Hawkeye Firefly X Lite CSV
- XTU (S2Pro, S3Pro)
- WitMotion (WT901SDCL binary and *.txt)
- Vuze (VuzeXR)
- KanDao (Obisidian Pro, Qoocam EGO)
- [CAMM format](https://developers.google.com/streetview/publish/camm-spec)

Keep in mind that loading large video files may take a few seconds.

### Interface Features
- **Customisation**: Since the distribution of the IMU telemetry data is specific to the prototype, advanced users can tweak the hyperparameters to improve the model's performance on a particular object.

- **Timelines**:
    - **GT (Ground Truth)**: User-annotated labels.
    - **AI**: Model predictions.
    - **Ci (Confidence Score)**: Visual representation of confidence scores (0 = red, 1 = green).

    Each pixel in the timelines represents approximately 5 frames, with each frame containing around 3 IMU telemetry points in a 60 fps video sampled at 200 Hz.

- **PCA Plot**: Located at the bottom of the page. Clicking a data point seeks to the corresponding timestamp in the video. To refresh the PCA plot, manually request a new axis (e.g., after the model updates it's predictions).

- **Labeling**:
    - Add labels by clicking the 'Add label' block, which creates a continuous block on the ground truth timeline at the playhead (red bar).
    - Ensure at least one labeled block is added before pressing the 'Predict and Train' button.

- **Training and Prediction**:
    - Pressing the 'Predict and Train' button sends the labels to the backend for (re)training the model and generating predictions.
    - During processing, you can continue labeling as needed.
    - After predictions are received, the AI and Ci timelines are updated:
        - **AI Timeline**: Displays blocks of continuous predictions of consecutive frames with the same label.
        - **Ci Timeline**: Shows the confidence score's associated colour for each pixel based on the first frame's score within the pixel's represented frames (around 5).

- **Adopting Predictions**:
    - Click on a block in the AI timeline and then on 'adopt' to apply the prediction to the ground truth timeline.
    - This is the active learning part: based on the confidence the model has (visualised by red-green gradient on Ci timeline), users are adviced to annotate the most uncertain parts first and can decide when to stop annotating.
    - **Note**: This action overrides any overlapping user-annotated blocks, so use it cautiously.

- **Model Training Constraints**:
    - Do not press the 'Train and Predict' again before the model has updated it's predictions.
    - Additional labels can be added at any time, but this will automatically retrain the model from scratch, which may take some time.
    - For better model performance, it's best to label each action equally.
    - The more epochs, the more the model (over)fits to the ground truth timeline, but will never reach 100% accuracy due to neuron dropout.

### Finalizing and Exporting
Once annotation and labeling are complete, click the 'Finish' button to navigate to the export page, where you can:

- Download the trained model.
- Retrain the model on another video.
- Use the model to infer predictions on new, unseen data.
    
### Anomaly Detection
On the predict page a slider allows you to set the anomaly threshold (manually refresh the PCA plot). Data points with confidence scores below the threshold are labeled as anomalies in the PCA plot.

## License
This project is [licensed](https://github.com/interactive-Telemetry-for-Design/interactive-telemetry-for-design/blob/main/LICENSE) under the terms of the GNU General Public License v3.0 (GPLv3).
