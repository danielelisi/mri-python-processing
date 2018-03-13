from brain_img_processor import BrainData, normalize_255
import numpy as np

class BrainProcessor:

    def __init__(self):

        self.brain_data = None
        self.top_view_index = 0
        self.front_view_index = 0
        self.side_view_index = 0
        
        self.pre_process_output = None
        self.marker = None
        self.local_gradient = None
        self.watershed = None

    def load_mri(self, file_path):

        '''
        Loads an mri image from a file path

        returns false if it had problem loading the file
        '''

        try:
            self.brain_data = BrainData(file_path)
        except Exception:
            print(f'Error opening file {file_path}')
            return False

        return True

    def get_data(self):
        normalized = normalize_255(self.brain_data.data)

        return {
            'data': normalized,
            'dim':  self.brain_data.dimensions
        }