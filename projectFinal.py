import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.linear_model import LinearRegression   
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from utils_nans1 import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('data\\test.csv')
df.head
#broj kolona i redoba
print(df.tail())
print(df.shape)
print(df.info())
# gledam nedostajuce vrednosti
print(df.isnull().sum()) #nema nedostajucih vrednosti
#statistika merenje o data
print(df.describe())
#provarevamo za nas cilj
print(df['HeartDisease'].value_counts)
#korelacija
# Izračunavanje korelacije između svih promenljivih u DataFrame-u
df_encoded = pd.get_dummies(df.drop(['HeartDisease'], axis=1), drop_first=True)
correlation_matrix = df_encoded.corr()

# Kopiranje DataFrame-a (ako je potrebno)
df_perfect_collinearity = df_encoded.copy()

plt.figure(figsize=(8, 5))  # da podesimo veličinu grafika

# Racunamo matricu korelacije
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.1)
plt.title('Matrica korelacije')
plt.show()


#upit koliko godine zavise da li ce se desiti hearth failure
print('-------------------------------------------------------------------------------------')


# Učitavanje podataka

data = pd.read_csv('data\\test.csv')

# Odabir relevantnih promenljivih
features = ['Age']
target = 'HeartDisease'

# Razdvajanje podataka na trening i test skup
X = data[features].values.reshape(-1, 1)  # X mora biti dvodimenzionalan
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicijalizacija modela
model = LogisticRegression()

# Treniranje modela
model.fit(X_train, y_train)

# Predviđanje na test skupu
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Verovatnoća pozitivnog ishoda
y_pred = np.where(y_pred_proba > 0.5, 1, 0)  # Prag odlučivanja 0.5

# Evaluacija modela
accuracy = accuracy_score(y_test, y_pred)


print(f'Accuracy: {accuracy}')
print('Classification Report:')

# Vizualizacija regresionog modela
plt.scatter(X_test, y_test, color='lightblue', label='Stvarne vrednosti')
plt.scatter(X_test, y_pred, color='red', marker='x', label='Predviđene vrednosti', alpha=0.7)  # Dodajte alpha argument
plt.xlabel('Godine')
plt.ylabel('Heart Disease (1: Yes, 0: No)')
plt.title('Zavisnost Heart Disease od Godina')
plt.legend()
plt.show()

print('-----------------------------------------------')


# Učitavanje podataka
data3 = pd.read_csv('data\\test.csv')

# Odabir relevantnih promenljivih
features = ['Cholesterol']
target = 'HeartDisease'

# Razdvajanje podataka na trening i test skup
X = data3[features].values.reshape(-1, 1)  # X mora biti dvodimenzionalan
y = data3[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicijalizacija modela
model = LogisticRegression()

# Treniranje modela
model.fit(X_train, y_train)

# Predviđanje na test skupu
y_pred = model.predict(X_test)

# Evaluacija modela
accuracy = accuracy_score(y_test, y_pred)


print(f'Accuracy: {accuracy}')
print('Classification Report:')
plt.scatter(X_test, y_test, color='lightblue', label='Stvarne vrednosti')
plt.scatter(X_test, y_pred, color='red', marker='x', label='Predviđene vrednosti', alpha=0.7)  # Dodajte alpha argument
plt.xlabel('Holesterol')
plt.ylabel('Heart Disease (1: Yes, 0: No)')
plt.title('Zavisnost Heart Disease od Holesterola')
plt.legend()
plt.show()

print('-------------------------------------------')


# Učitavanje podataka
data2 = pd.read_csv('data\\test.csv')

# Odabir relevantnih promenljivih
features = ['RestingBP']
target = 'HeartDisease'

# Razdvajanje podataka na trening i test skup
X = data2[features].values.reshape(-1, 1)  # X mora biti dvodimenzionalan
y = data2[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicijalizacija modela
model = LogisticRegression()

# Treniranje modela
model.fit(X_train, y_train)

# Predviđanje na test skupu
y_pred = model.predict(X_test)

# Evaluacija modela
accuracy = accuracy_score(y_test, y_pred)


print(f'Accuracy: {accuracy}')

plt.scatter(X_test, y_test, color='red', label='Stvarne vrednosti')
plt.scatter(X_test, y_pred, color='lightblue', marker='x', label='Predviđene vrednosti', alpha=0.7)  # Dodajte alpha argument
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Heart Disease (1: Yes, 0: No)')
plt.title('Zavisnost Heart Disease od Resting Blood Pressure')
plt.legend()
plt.show()


# Učitavanje podataka
data4 = pd.read_csv('data\\test.csv')

# Liste kategoričkih kolona koje želimo enkodirati
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Primena one-hot encoding-a na kategoričke kolone
data_encoded = pd.get_dummies(data4, columns=categorical_columns, drop_first=True)

# Prikazivanje rezultata
print(data_encoded.head())

#pretvaram u brojeve
from sklearn.preprocessing import LabelEncoder

# Kreiranje LabelEncoder objekta
label_encoder = LabelEncoder()

# Enkodiranje labele za kategoričke kolone
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Prikazivanje rezultata
print(data.head())

# Odvajanje ciljne promenljive od atributa
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Kreiranje i treniranje modela logističke regresije
model = LogisticRegression(max_iter = 1000, random_state=42)
model.fit(X, y)

# Izvlačenje značajnosti atributa (koeficijenata) iz modela
feature_importance = model.coef_[0]

# Prikazivanje značajnosti atributa
for feature, importance in zip(X.columns, feature_importance):
    print(f'{feature}: {importance}')

# Racunamo RMSE

x3 = data.drop(columns=['HeartDisease', 'ST_Slope', 'FastingBS', 'Sex', 'ExerciseAngina'])
y3 = data['HeartDisease']

# Delimo na test i train podatke
x_train3, x_val3, y_train3, y_val3 = train_test_split(x3, y3, train_size=0.8, shuffle=True, random_state=42)

# Kreiranje llr
logistic_model = LogisticRegression(max_iter = 1000, random_state=42)
logistic_model.fit(x_train3, y_train3)

# Assess assumptions (if applicable)
# Note: Logistic regression has its own assumptions, different from linear regression.
# Make sure your 'are_assumptions_satisfied' function is suitable for logistic regression.

val_predictions = logistic_model.predict(x_val3)

# RMSE
val_rmse = mean_squared_error(y_val3, val_predictions, squared=False)
print(f'Validation RMSE: {val_rmse}')

# RMSE2 drugi model
x4 = data.drop(columns=['HeartDisease', 'ST_Slope', 'FastingBS', 'Sex', 'ExerciseAngina', 'Cholesterol'])
y4 = data['HeartDisease']

# Delimo na test i train podatke
x_train4, x_val4, y_train4, y_val4 = train_test_split(x4, y4, train_size=0.8, shuffle=True, random_state=42)

# Kreiranje llr
logistic_model = LogisticRegression(max_iter = 1000, random_state=42)
logistic_model.fit(x_train4, y_train4)


val_predictions2 = logistic_model.predict(x_val4)

# RMSE
val_rmse2 = mean_squared_error(y_val4, val_predictions2, squared=False)
print(f'Validation RMSE: {val_rmse2}')

#SVM
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Učitavanje podataka
data = pd.read_csv('data\\test.csv')

# Liste kategoričkih kolona koje želimo enkodirati
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Primena one-hot encoding-a na kategoričke kolone
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Prikazivanje rezultata
print(data_encoded.head())

# Odvajanje ciljne promenljive od atributa
X = data_encoded.drop('HeartDisease', axis=1)
y = data_encoded['HeartDisease']

# Podela podataka na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizacija podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kreiranje i treniranje SVM modela
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predviđanje na test skupu
y_pred = svm_model.predict(X_test_scaled)

# Evaluacija modela
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Validation SVM: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
