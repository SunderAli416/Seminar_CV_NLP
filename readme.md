# Image Captioning

## Setup
### AMBER Dataset
 - Download AMBER Dataset [Dataset](https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view?usp=sharing)
 - Create a folder 'data' in root directory 
 - Extract dataset in 'data' (data/image/AMBER_1.jpg).
### Python Environment 
 - Create and activate virtual environment.
	`python -m venv env`
	`\env\Scripts\Activate`
	
 - Install requirements
  `pip install requirements.txt`
  
### RAM++
 - Create a folder "pretrained" in root directory 
 - Download [checkpoint file](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) in this directory
 - Install RAM++ as package
 `pip install git+https://github.com/xinyu1205/recognize-anything.git`
 
