import warnings
warnings.filterwarnings('ignore')

import json

from PredictionHelper import *
from PredictionHelper_CS import *

def load_json_config(file_name):
    try:
        with open(file_name, 'r') as file:
            config = json.load(file)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{file_name}': {e}")
        return None
    
def run_baseline_prediction(species_folder, audio_folder, extension, weights_folder, saved_prediction):
    # Load configuration from JSON
    config = load_json_config('params.json')

    species_name = 'hainan_gibbon' # just specify the name of your species here, the parameters already define in the json
    species_params = config[species_name]  # this for thyolo alethe
    segment_duration = species_params['segment_duration']
    positive_class = species_params['positive_class']
    negative_class = species_params['negative_class']
    audio_extension = species_params[extension] #
    n_fft = species_params['n_fft']
    hop_length = species_params['hop_length']
    n_mels = species_params['n_mels']
    f_min = species_params['f_min']
    f_max = species_params['f_max']

    predict = PredictionHelper(species_folder, audio_folder, segment_duration,
            positive_class, negative_class, 
            n_fft, hop_length, n_mels, f_min, f_max, audio_extension, weights_folder)
        
    predict.predict_all_test_files(weights_folder, saved_prediction)
            
    print("done.")
    print('===============================================================')

    return None


def run_cs_prediction(species_folder, extension, weights_folder, saved_prediction, compression_rate, solver, threads, batch_size):
    # Load configuration from JSON
    config = load_json_config('params.json')
    species_name = 'hainan_gibbon' # just specify the name of your species here, the parameters already define in the json
    species_params = config[species_name]
    segment_duration = species_params['segment_duration']
    positive_class = species_params['positive_class']
    negative_class = species_params['negative_class']
    audio_extension = species_params[extension] #
    n_fft = species_params['n_fft']
    hop_length = species_params['hop_length']
    n_mels = species_params['n_mels']
    f_min = species_params['f_min']
    f_max = species_params['f_max']


    predict = PredictionHelper_CS(species_folder, segment_duration,
                positive_class, negative_class,
                n_fft, hop_length, n_mels, f_min, f_max, weights_folder,compression_rate, solver)
        
    predict.predict_all_test_files(True, weights_folder, saved_prediction,threads, batch_size)
            
    print("done.")
    print('===============================================================')

    return None