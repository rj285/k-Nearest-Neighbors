from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
fn = iris.feature_names
# print(fn)
'''
['sepal length (cm)', 'sepal width (cm)', 
'petal length (cm)', 'petal width (cm)']
'''
tn = iris.target_names
# print(tn) 
#['setosa' 'versicolor' 'virginica']

df = pd.DataFrame(iris.data, columns=fn)
# print(df.head())
'''
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2
'''

df['target'] = iris.target
# print(df.head())
'''
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0
'''

df_tar_1 = df[df.target == 1].head()
# print(df_tar_1)
'''
sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
50                7.0               3.2                4.7               1.4       1
51                6.4               3.2                4.5               1.5       1
52                6.9               3.1                4.9               1.5       1
53                5.5               2.3                4.0               1.3       1
54                6.5               2.8                4.6               1.5       1
'''
df['flower_name'] = df.target.apply(lambda x:iris.target_names[x])
'''
adds a new column named 'flower_name' to a DataFrame 'df', 
where each value corresponds to the 'target_names' from the 'iris' dataset, 
indexed by the values in the 'target' column of 'df'.
'''
# print(df)
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target flower_name
0                  5.1               3.5                1.4               0.2       0      setosa
1                  4.9               3.0                1.4               0.2       0      setosa
2                  4.7               3.2                1.3               0.2       0      setosa
3                  4.6               3.1                1.5               0.2       0      setosa
4                  5.0               3.6                1.4               0.2       0      setosa
..                 ...               ...                ...               ...     ...         ...
145                6.7               3.0                5.2               2.3       2   virginica
146                6.3               2.5                5.0               1.9       2   virginica
147                6.5               3.0                5.2               2.0       2   virginica
148                6.2               3.4                5.4               2.3       2   virginica
149                5.9               3.0                5.1               1.8       2   virginica
'''
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]
'''
'df0' containing the first 50 rows, 
'df1' containing rows 50 to 99,
'df2' containing rows 100 and onwards.
'''
# print(df0.head())
'''
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target flower_name
0                5.1               3.5                1.4               0.2       0      setosa
1                4.9               3.0                1.4               0.2       0      setosa
2                4.7               3.2                1.3               0.2       0      setosa
3                4.6               3.1                1.5               0.2       0      setosa
4                5.0               3.6                1.4               0.2       0      setosa
'''
# print(df1.head())
'''
    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target flower_name
50                7.0               3.2                4.7               1.4       1  versicolor
51                6.4               3.2                4.5               1.5       1  versicolor
52                6.9               3.1                4.9               1.5       1  versicolor
53                5.5               2.3                4.0               1.3       1  versicolor
54                6.5               2.8                4.6               1.5       1  versicolor
'''
# print(df2.head())
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target flower_name
100                6.3               3.3                6.0               2.5       2   virginica
101                5.8               2.7                5.1               1.9       2   virginica
102                7.1               3.0                5.9               2.1       2   virginica
103                6.3               2.9                5.6               1.8       2   virginica
104                6.5               3.0                5.8               2.2       2   virginica
'''
import matplotlib.pyplot as plt
plt.xlabel('SEPAL LENGTH')
plt.ylabel('SEPAL WIDTH')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'], color='green', marker='+', label='df0-setosa')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'], color='blue', marker='.', label='df1-versicolor')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'], color='red', marker='*', label='df2-virginica')
plt.legend()
plt.savefig('iris data separation (sepal length,sepal width).png')
# plt.show()

from sklearn.model_selection import train_test_split
x = df.drop(['target','flower_name'], axis='columns')
y = df.target
# print(x)
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
..                 ...               ...                ...               ...
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8
'''
# print(y)
'''
0      0
1      0
2      0
      ..
148    2
149    2
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
# print(x_train)
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
147                6.5               3.0                5.2               2.0
110                6.5               3.2                5.1               2.0
82                 5.8               2.7                3.9               1.2
120                6.9               3.2                5.7               2.3
66                 5.6               3.0                4.5               1.5
..                 ...               ...                ...               ...
22                 4.6               3.6                1.0               0.2
7                  5.0               3.4                1.5               0.2
89                 5.5               2.5                4.0               1.3
87                 6.3               2.3                4.4               1.3
34                 4.9               3.1                1.5               0.2
'''
# print(x_test)
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
73                 6.1               2.8                4.7               1.2
106                4.9               2.5                4.5               1.7
142                5.8               2.7                5.1               1.9
128                6.4               2.8                5.6               2.1
..                 ...               ...  
50                 7.0               3.2                4.7               1.4
102                7.1               3.0                5.9               2.1
15                 5.7               4.4                1.5               0.4
97                 6.2               2.9                4.3               1.3
'''
# print(y_train)
'''
76     1
107    2
122    2
53     1
3      0
      ..
69     1
67     1
131    2
80     1
56     1
'''
# print(y_test)
'''
110    2
145    2
42     0
40     0
      ..
28     0
62     1
57     1
138    2
'''
# print(len(x_train)) #100
# print(len(x_test)) #50

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
KNeighborsClassifier(n_neighbors=3)
knn_score = knn.score(x_test, y_test)
# print(knn_score) #0.96 or 1.0

from sklearn.metrics import confusion_matrix

x_pred = knn.predict(x_test)
# print(x_pred)
'''
[0 1 1 2 1 2 2 1 2 0 2 1 1 1 1 2 1 0 2 0 1 1 1 1 1 2 1 2 0 2 0 1 0 2 1 1 0 2 0 2 2 2 1 2 1 1 2 1 1 0]
'''
cm = confusion_matrix(y_test, x_pred)
# print(cm)
'''
[[21  0  0]
 [ 0 15  1]
 [ 0  0 13]]
'''

import seaborn as sn

plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel = ('PREDICTED')
plt.ylabel = ('TRUTH')
plt.savefig('truth vs prediction(using seaborn).png')
plt.show()