# Real Time Violence Detection in Video

Detection of violent events in surveillance systems plays a crucial role in law enforcement and public safety. The effectiveness of such violence detection systems is typically measured by response speed, detection accuracy, and generalizability across different video sources and formats. Previous studies in this domain have often focused on either speed or accuracy, or a combination of both, but rarely addressed the generalizability across diverse video inputs.

In this work, we propose a real-time violence detection framework based on deep learning techniques. The model integrates a Convolutional Neural Network (CNN) for spatial feature extraction with an LSTM network for modeling temporal relationships, aiming to optimize three key factors: generalizability, accuracy, and fast response time. Experimental results demonstrate that the proposed system achieves 98% accuracy while processing 131 frames per second, outperforming previous approaches. Comparative analysis indicates that our model provides the highest accuracy and fastest processing speed among existing violence detection methods, making it highly suitable for real-time surveillance applications.

## Project Structure

- `mamonfight22.py`: The core file containing the video reader function and the pre-trained deep learning models (`mamon_videoFightModel` and `mamon_videoFightModel2`) to predict violence. It integrates a VGG19 base model with a TimeDistributed CNN and an LSTM network.
- `web-fight22.py`: A Flask web API server script that exposes an endpoint (`/api/fight/`) for posting a video and retrieving a prediction on whether violence was detected.
- `client.py`: A client script showing how to send a video to the Flask web API server and print out the server's response (prediction and processing time).
- `genral-training-example.ipynb`, `localfile-testing.ipynb`, `predicting-localvideo-and-saving-violence-video.ipynb`: Jupyter notebooks that show examples of training, local testing, and prediction flows.
- `mamon98777.keras`, `mamon-videofight100.hdf5`, `mamonbest947oscombo-drive.h5` (if applicable): Pre-trained model weights.

## Prerequisites

To run this project, make sure you have the following installed:
- Python 3.x
- TensorFlow (Note: The project contains references to both TensorFlow 1.x `tf.contrib.keras` and TensorFlow 2.x `tf.keras`. An upgrade to a pure TensorFlow 2.x implementation may be required.)
- NumPy
- OpenCV (`cv2`)
- scikit-image (`skimage`)
- Flask
- Requests
- Pillow (`PIL`)

## Usage

### 1. Running the API Server
Start the Flask API to listen for incoming video files:
```bash
python web-fight22.py
```
This will start the server at `http://0.0.0.0:3091`.

### 2. Testing with Client
Use the client script to send a local video file to the API for prediction:
```bash
python client.py
```
Make sure you have a valid video file (e.g., `hdfight.mp4`) in the same directory, or modify `client.py` to point to a valid file.

### 3. Local Prediction Testing
Alternatively, you can test video prediction locally using the provided functions in `mamonfight22.py` or by opening the Jupyter Notebook `localfile-testing.ipynb`.

## Known Issues & Notes

1. **TensorFlow 1.x vs 2.x Compatibility**:
   - `mamonfight22.py` has a function `mamon_videoFightModel` that uses `tf.contrib.keras`, which is deprecated and removed in TensorFlow 2.x.
   - You might need to adjust imports to standard `tf.keras` for compatibility with modern TensorFlow versions.

2. **`np.float` Deprecation**:
   - Some arrays are initialized using `np.float`, which is deprecated in newer versions of NumPy. It is recommended to replace `np.float` with `np.float64` or Python's built-in `float`.

3. **Missing Model Weights**:
   - `web-fight22.py` and `mamonfight22.py` require weight files (e.g. `mamonbest947oscombo-drive.h5`) to be present in the directory.
