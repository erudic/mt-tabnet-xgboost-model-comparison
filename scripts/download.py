from kaggle import api as kaggle_api

dataset = "sobhanmoosavi/us-accidents"
outpath = "data/"
kaggle_api.dataset_download_files(dataset,outpath,unzip=True)
