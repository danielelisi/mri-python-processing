from brain_img_processor import BrainData, normalize_255


class BrainProcessor:

    def __init__(self):

        self.brain_data = None

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