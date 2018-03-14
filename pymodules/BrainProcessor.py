from brain_img_processor import *
import numpy as np

class BrainProcessor:

    def __init__(self):

        self.brain_data = None
        self.pre_process_output = None
        self.marker = None
        self.local_gradient = None 
        self.watershed = None
        self.current_view = None
        self.current_index = -1
        self.filters = {
            'bilateral': bilateral,
            'equalize': equalize,
            'median': median
        }

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

    def set_view(self, view):
        '''
        view = (top | side | front)
        '''
        if view == 'top':
            profile = BrainData.TOP_PROF
        if view == 'side':
            profile = BrainData.SIDE_PROF
        if view == 'front':
            profile = BrainData.FRONT_PROF

        self.current_view = profile
        
        return
    def init_pre_process_output(self):
        slice = self.brain_data.get_slice(self.current_view, self.current_index)
        self.pre_process_output = normalize_255(slice)
        return

    def set_view_index(self, index):
        
        self.current_index = index
        return

    def apply_filter(self, filter):
        
        if filter in self.filters:
            print('applying',filter)
            f = self.filters[filter]
            self.pre_process_output = f(self.pre_process_output)
            self.pre_process_output = normalize_255(self.pre_process_output)
        
        return

    def get_original_view(self):
        slice = self.brain_data.get_slice(self.current_view, self.current_index)
        result = normalize_255(slice)
        return result
        
    def get_pre_process_output(self):
        return self.pre_process_output

    def reset_pre_process(self):

        self.set_view_index(self.current_index)

        return
    
    def isolate_brain(self):

        self.pre_process_output = isolate_brain(self.pre_process_output)['data']
        return

    def apply_watershed(self):
        return
