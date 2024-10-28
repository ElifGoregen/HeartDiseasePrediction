#import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import warnings
warnings.filterwarnings("ignore")


#load dataset ve EDA ---------------------
df=pd.read_csv("heart_disease_uci.csv")
df =df.drop(columns=["id"])

df.info()
#object string
describe=df.describe()
#sayısal değerlerin istatiksel değerleri
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

plt.figure()
sns.pairplot(df,vars=numerical_features,hue="num")
plt.show()
#soluk renkler sağa yaklaştıkça yani yaş arttıkça hastalık görülme ihtimalia artıyor.
#kolestrol datasında iki tepecik var sebebi:sıfır verisinden kaynaklı.
#ca ve num benzerliği: ca ve num featurunda 0 1 2 3 4 değerlerini olması kategorik değerler.Nümerik demek yanlış
#oldpeak:Sıralı olması yanlış veri eksik veya yanlıl girilmiş.
#Tek nokta halinde olanlar outlierler.
plt.figure()
sns.countplot(x="num",data=df)
plt.show()
#Dört numarayı artırabilirdik.Dengesiz dağılım olduğu için.
#Dört numarayı çıkartabiliriz.Bu da bir seçenek.
#Dört numara başarıyı olumsuz etkiliyor.
#Sınıflandırma çalışması için.


#handling missing value --------------------
#kayıp verileri veri setinden çıkarabilir ya da doldurabiliriz.
df.isnull() 
#Değerler false ve true şeklinde.
#True ise kayıp veri demek.
df.isnull().sum()
#slope:309 ca:611 adet kayıp veri var.
print(df.isnull().sum())
#ca sütununu veri setinden çıkaralım.

df=df.drop(columns=["ca"])
print(df.isnull().sum())
#şu an thal:486 ve slope :309 kayıp değere sahip.

#nan değerleri trestbpsnin median değerini alabiliriz.
df["trestbps"].fillna(df["trestbps"].median(),inplace=True)
#fillna metodunu missing value doldurmak için kullandık.
#median değeri:130.
#inplace parametresi df= demek yerine direkt olarak doldurmuş oluyor.

#aynı işlemi kolestrol için de yapalım.
df["chol"].fillna(df["chol"].median(),inplace=True)
df["fbs"].fillna(df["fbs"].mode()[0],inplace=True)
#fbs featuru boolean veri tipinde olduğu için modunu alıyoruz.
#mode false mu true sayısı mı fazla bunu veriyor.False sayısı fazla olduğu için nan değerleri false ile dolduracak.
#False değerine erişim sağlayabilmek için mode un sıfırncı indeksini yazmamız gerekiyor.
#Böylece false değerine direkt erişim sağladık.
#df["fbs"].mode()[0]
#Out[29]: False
df["restecg"].fillna(df["restecg"].mode()[0],inplace=True)
df["thalch"].fillna(df["thalch"].median(),inplace=True)
df["exang"].fillna(df["exang"].mode()[0],inplace=True)
df["oldpeak"].fillna(df["oldpeak"].median(),inplace=True)
df["slope"].fillna(df["slope"].mode()[0],inplace=True)
df["thal"].fillna(df["thal"].mode()[0],inplace=True)
print(df.isnull().sum())
#Herhangi bir nan değer kalmamış oldu.

#train test split----------------------------
X = df.drop(["num"],axis=1)
y = df["num"] #target variable

X_train , X_test, y_train, y_test= train_test_split(X ,y ,test_size=0.25,random_state=42)
#920 adet verinin 690 train 230 test verisi olarak ayrıldı.
#Verinin içindeki kategorik ve numerik değerleri ayıralım

categorical_features=["sex","dataset","cp","restecg","exang","slope","thal"]
numerical_features=["age","trestbps","chol","fbs","thalch","oldpeak"]

X_train_num=X_train[numerical_features]
X_test_num = X_test[numerical_features]

scaler = StandardScaler()
X_train_num_scaled=scaler.fit_transform(X_train_num)
X_test_num_scaled=scaler.transform(X_test_num)

#Kategorik değerleri encoder işlemi ile kodlamamız gerekiyor.
#sparse desteklemiyor onu sparse_output ile değiştirdik.
encoder=OneHotEncoder(sparse_output=False, drop="first")
X_train_cat =X_train[categorical_features]
X_test_cat = X_test[categorical_features]

X_train_cat_encoded=encoder.fit_transform(X_train_cat)
X_test_cat_encoded=encoder.transform(X_test_cat)

#Örneğin female sütununu kadın:1 erkek:0 olacak şekilde kodladık.

#Kategorik ve nümerik değerleri ayrı ayrı scaled,encoded ettik tek bir veri setinde birleştirelim.
X_train_transformed = np.hstack((X_train_num_scaled,X_train_cat_encoded)) #horizontal stack yatay birleştirme
X_test_transformed = np.hstack((X_test_num_scaled,X_test_cat_encoded))


#modelling:Random Forest,KNN,Voting Classifier,train ve test-----------------

rf=RandomForestClassifier(n_estimators=100,random_state=42)
knn=KNeighborsClassifier()

voting_clf = VotingClassifier(estimators=[
      ("rf",rf),
      ("knn",knn)],voting="soft")
      #"knn",knn],voting="hard")  
#Farklı makine öğrenmesi yöntemlerinin sonuçlarını alır ve oylama yöntemi ile sonucu verir.
#voting parametresi oylamayı nasıl yapacağını belirler.
#soft da çıkan sonuçların olasılık ortalamasını alır.
#hard da her bir ml sonuçlarına bakıp çoğunluk ortalamasına bakıyor.
#MODEL EĞİTİMİ

voting_clf.fit(X_train_transformed,y_train)

#test verisi ile tahmin yap

y_pred = voting_clf.predict(X_test_transformed)

print("Accuracy:",accuracy_score(y_test,y_pred))
#y test gerçek değerler y pred tahmin değerleri


#confusion matris-------------------------

print("Confusion Matrix:")
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("Classificatioon Report")
print(classification_report(y_test,y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

#Target değeri 4 olan verilerin hepsini yanlış sınıflandırıyor.
# 4 ile 3 birleştirilebilir:Çözüm














