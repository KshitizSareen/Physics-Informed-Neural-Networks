from model_train import load_model,data
from main_plot import fetchData
import numpy as np
import pandas as pd
from collections import defaultdict


def extract_model_data(t,YData,XData,dataDict):
    Z = fetchData(t,YData,XData)
    for y in range(len(YData)):
        for x in range(len(XData)):
            dataDict[f"{XData[x]},{YData[y]}"].append(Z[y][x])
    

if __name__ == "__main__":

    XData = [0, 1, 2.5, 5,10, 15]
    YData = [0, 1, 5, 15]

    maxTimestamp = 200000

    dataDict= defaultdict(list)
    extract_model_data(1,YData,XData,dataDict)
    for i in range(10,maxTimestamp+1,10):
        extract_model_data(i,YData,XData,dataDict)
    data_from_model = pd.DataFrame()
    data_from_model["Timetamp"] = np.array([1]+list(range(10,200001,10)))
    data_from_model["Power"] = [data["Power"][1] for i in range(len(data_from_model["Timetamp"]))]
    for key in dataDict:
        data_from_model[key] = np.array(dataDict[key])
    data_from_model.to_csv('data_from_model.csv', index=False)
    