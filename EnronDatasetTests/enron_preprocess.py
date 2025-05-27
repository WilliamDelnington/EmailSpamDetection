import sys
import pandas as pd

sys.path.append("../Handlers")

from preprocessing import EnronPreprocess

enron1 = pd.read_csv("./csv/enron1.csv")
enron2 = pd.read_csv("./csv/enron2.csv")
enron3 = pd.read_csv("./csv/enron3.csv")
enron4 = pd.read_csv("./csv/enron4.csv")
enron5 = pd.read_csv("./csv/enron5.csv")
enron6 = pd.read_csv("./csv/enron6.csv")

enron1_preprocess = EnronPreprocess(enron1, "enron1")
enron1_preprocessed_X, enron1_preprocessed_y = enron1_preprocess.preprocess_data(save=True)

enron2_preprocess = EnronPreprocess(enron2, "enron2")
enron2_preprocessed_X, enron2_preprocessed_y = enron2_preprocess.preprocess_data(save=True)

enron3_preprocess = EnronPreprocess(enron3, "enron3")
enron3_preprocessed_X, enron3_preprocessed_y = enron3_preprocess.preprocess_data(save=True)

enron4_preprocess = EnronPreprocess(enron4, "enron4")
enron4_preprocessed_X, enron4_preprocessed_y = enron4_preprocess.preprocess_data(save=True)

enron5_preprocess = EnronPreprocess(enron5, "enron5")
enron5_preprocessed_X, enron5_preprocessed_y = enron5_preprocess.preprocess_data(save=True)

enron6_preprocess = EnronPreprocess(enron6, "enron6")
enron6_preprocessed_X, enron6_preprocessed_y = enron6_preprocess.preprocess_data(save=True)

merged_enron = pd.concat([
    enron1, enron2, enron3, 
    enron4, enron5, enron6
], ignore_index=True)

# Since it takes a long time to extract features from the entire dataset,
# we will get the merged data in a separate function.
def get_merged_enron_data():
    merged_enron_preprocess = EnronPreprocess(merged_enron, "merged_enron")
    merged_enron_X, merged_enron_y = merged_enron_preprocess.preprocess_data(save=True)
    return merged_enron_preprocess, merged_enron_X, merged_enron_y

get_merged_enron_data()