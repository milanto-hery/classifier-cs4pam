import json
from MultipleRun import run_cs_prediction
import os

def load_json_config(file_name):
    try:
        with open(file_name, 'r') as file:
            config = json.load(file)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{file_name}': {e}")
        return None

def main():
    species_folder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(species_folder)
    compression_rate = 0.15
    print(f"Compression ratio: {compression_rate}")
    threads= 24
    batch_size = 24
    solver = 'lasso'
    weights_folder = './Weights_CS/Weights_15'
    saved_prediction = 'Prediction_cs_15'
    saved_segments = 'Segments'
    extension = 'audio_wav'
    run_cs_prediction(species_folder, extension, weights_folder, saved_segments, saved_prediction, compression_rate, solver, threads, batch_size)
if __name__ == "__main__":
    main()
