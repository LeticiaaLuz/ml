import lps_ml.core as ml_core

loader = ml_core.AudioFileLoader.iara("/data/IARA")
print(loader.file_dict)
