from brain_img_processor import *
from random import randint
import numpy as np

class BrainProcessor:

    def __init__(self):

        self.brain_data = None
        self.pre_process_output = None
        self.normalized = None
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
        self.watershed_output = None
        self.tumor_info = None

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
        # normalized = normalize_255(self.brain_data.data)

        return {
            'data': self.brain_data.data,
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
        self.pre_process_output = self.brain_data.get_slice(self.current_view, self.current_index)
        # self.pre_process_output = normalize_255(slice)
        self.normalized = self.pre_process_output

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
        result = self.brain_data.get_slice(self.current_view, self.current_index)
        # result = normalize_255(slice)
        return result
        
    def get_pre_process_output(self):
        return self.pre_process_output

    def reset_pre_process(self):

        self.set_view_index(self.current_index)

        return
    
    def isolate_brain(self):

        self.pre_process_output = isolate_brain(self.pre_process_output)['data']
        self.normalized = self.pre_process_output
        return

    def apply_watershed(self):

        self.watershed_output = watershed_segment(self.pre_process_output)
        self.watershed = self.watershed_output['watershed']

        unique_values = np.unique(self.watershed_output['watershed'])

        color_key = {}
        for value in unique_values:
            color_key[value] = _generate_rand_color()

        temp = self.watershed_output['watershed']
        x, y = temp.shape

        temp_watershed = []
        for i in range(x):
            new_row = []
            for j in range(y):
                new_row.append(color_key[temp[i,j]])
            temp_watershed.append(new_row)

        self.watershed_output['watershed'] = temp_watershed

        return


    def get_watershed_output(self):
        return self.watershed_output

    def detect_tumor(self, threshold):

        watershed = self.watershed

        self.tumor_info = detect_tumor(watershed, self.normalized, threshold)

        return self.tumor_info

def _generate_rand_color():

    red = randint(0,255)
    green = randint(0,255)
    blue = randint(0,255)

    return {
        'red': red,
        'green': green,
        'blue': blue
    }
