import pandas as pd
import numpy as np


data = pd.read_csv("./data/data.csv")
datadf = pd.DataFrame(data)

print(datadf)