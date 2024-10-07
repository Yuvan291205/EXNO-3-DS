## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/7c45e653-3d86-4fc5-b7dd-29d4fb5e2998)
# ORDINAL ENCODER:
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/35088ce4-4a1b-45b7-b657-42b50aff26ce)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/352d268d-75a9-48f0-a5d9-c2f647760535)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
```
![image](https://github.com/user-attachments/assets/5c237bc2-9359-4da3-b839-886efad815e8)
```
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/4ed857bc-111e-4f29-8a70-04ae766b7e39)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/59afabc4-deb2-48c8-a7d5-1f26ab51177d)
```
df2=pd.concat([df,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/543f1998-72cc-464b-b0ec-b0325743471f)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/d9fc84dc-0ef0-4728-945b-aa6f17a616fc)
```
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/e47b8bdf-d8ba-4519-a55b-d95ad29778c4)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/1a5458a5-b022-4742-90ee-09ecb4c299dd)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/0b7c389e-5879-4a90-bfcd-947a830b1423)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/296809e6-1bdd-4cc8-95f6-270ee17ded53)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/d676d571-5a0c-45d8-b13b-620d453f8997)
```

df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/e57f50bb-5499-4ac5-a161-03a3a5cf19d9)
```

df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/5a8c6629-7e0c-4d2d-8bbe-45372ec10d4f)
```

df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/94b3eed1-b6cd-4881-8148-d305579ea5e3)
```

df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/16200960-3294-47cc-b0f2-87cc936502b4)
```

df["Highly Positive Skew"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/3de1078f-efab-49f3-aa0a-85b74575edf6)
```

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3f924257-1dda-4271-b8e7-984ac2ff2928)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/969faf94-9887-4781-918b-fa75267ad1a0)
```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f0540762-0401-402d-aca8-d5ca91b574e2)

# RESULT:
       Thus,the given data are read and Feature Encoding and Transformation process are performed and the data is saved to the file.

       
