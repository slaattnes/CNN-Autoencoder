from tensorflow.python.keras.models import Model, load_model
import pickle
from sklearn.cluster import KMeans
import numpy as np

print ("Loading autoencoder...")
autoencoder_latest = load_model('./ae-latest-dense.h5')
windowed_features = None

kmeans = None
scaler = None

def initialize(actions=8):
   global kmeans
   global scaler
   print ("Initializing clusters...")
   kmeans = KMeans(n_clusters=actions)
   windowed_features = pickle.load(open("./cluster_data.p", "rb"))
   kmeans.fit(windowed_features)
   scaler = pickle.load(open("./scaler.p", "rb"))
   
window = []
def add_frame(frame):
    global kmeans
    global scaler
    frame = np.array(frame)
    if (len(frame.shape) is not 1 or frame.shape[0] is not 4):
        print ("Wrong frame shape! Must be 4 elements")
        return
        
    frame = np.array(frame).reshape(1, 4)
    frame = scaler.transform(frame)
    frame = frame.reshape(4)
        
    window.append(frame)
    
    if (len(window) > 999):
        window.pop(0)
    
    if (len(window) == 999):
        f = autoencoder_latest.predict(np.array(window).reshape(1, 999, 4))
        return kmeans.predict(f)[0]
    else:
        return -1
