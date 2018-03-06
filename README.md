# mri-python-processing
Digital Images Processing and Segmentation for Brain Tumor MRI


## Setting up Development Environment

### Prerequisites
* Python 3.6
* For Windows: Gitbash or WSL
   * run the commands using these shells


### Steps
1. Set up virtual environment   
* Windows   
```python -m venv <nameofenv>```
* Linux / Mac OSx  
```python3 -m venv <nameofenv>```

2. Activate the virtual environment  
__Note: you need to do this everytime you modify and run the project__  
* Windows  
```source <nameofenv>/Scripts/activate```
* Linux / Mac OSx  
```source <nameofenv>/bin/activate```

3. Install the packages  
```pip install -r requirements.txt```

## Running The Server

In the root directory of the project, run  
```. ./runserver.sh```