import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import gc
import numpy as np


a=pd.read_csv("last/1028.csv")
print a.head()

b=pd.read_csv("last/103_.csv")
print b.head()

df=a
df["proba"]=a["proba"]*0.7+b["proba"]*0.3
#
print df.head()







# f = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
# df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission_0607_last.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)