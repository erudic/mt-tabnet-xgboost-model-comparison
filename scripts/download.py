from kaggle import api as kaggle_api

dataset = "sobhanmoosavi/us-accidents"
outpath = "../data/us-accidents.zip"
kaggle_api.dataset_download_files(dataset,outpath)
