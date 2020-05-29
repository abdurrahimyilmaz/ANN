#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:24:07 2020

@author: abdurrahim
"""
#***** TÜM KODLARDAKİ NB_EPOCH İFADESİ GÖRÜRSEN O DEĞİŞTİ EPOCHS OLACAK BUNDAN
#DOLAYI ACCURACY DÜŞÜK ÇIKIYOR *******

#part 1 - data preproccessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values #matrix #13 dahil değil
y = dataset.iloc[:, 13].values #vector

#encoding categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

label_encoder_country = LabelEncoder()
x[:,1] = label_encoder_country.fit_transform(x[:,1])

label_encoder_sex = LabelEncoder()
x[:,2] = label_encoder_sex.fit_transform(x[:,2])

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x),dtype=np.float)

x = x[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling - lr otomatik yapmıyor manuel yapmamız gerekiyor
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#part 2 -build ann
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#initiliasing
classifier = Sequential()

#adding input and hidden layer
#rectification function for activation, 11 independent parameters yani input bunu ara katmanlarda vermemize gerek yok çünkü aradaki 
#katman önceki katmandan ne geldiğini biliyor sadece başta vermemiz gerekiyor, units independent yarısı tekse üste yuvarla, kernel 
#random olsun diye uniform default hali daha uygun ama
classifier.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer ='uniform'))
classifier.add(Dropout(p = 0.1)) #0.5 ve üzeri underfitting olur 0.1 iyidir

#adding second hidden layer
classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer ='uniform'))

#adding output layer
#ikiden fazla kategori olursa softmax, tahmin yaptığımız ve 2 kategori olduğu için sigmoid
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer ='uniform')) 

#compiling the ann
#optimizer = optimal weight nasıl bulacaz burada stochastic gradient descent = adam
#loss = loss function = optimize etmek ve optimal weight için burada adam ve stochastic gradient descent tabanlı ve logaritmik olmasından dolayı
#binary_crossentropy 2den fazla categori için categorical_crossentropy
#stochastic gradient descent matematiksel anlamda ayrıntısını incelersek onun loss functionı bi tür logistic regresyondur sigmoid ile aktive
#olan bir perceptronda da logistic regresyon elde etmiş oluruz bu stochatic gradient descent tabanlı perceptron temeline de inersek
#lossu lineer regresyon gibi değil logaritmik bir fonksiyon görürüz yani logarithmic bir lossumuz olur 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ann to training set
#batch_size = weight update edildikten sonra gözlem sayısıdır
#number_of_epoch = kaç defa bütün veri ann dolaşıp eğitimden geçecek
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

#part 3 - predictions and evaluating the model
#predict test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

#making the confusion matrix = biz modelin predictive powerını görmüş olduk
from sklearn.metrics import confusion_matrix #sol üst sağ alt doğru tahminler sağ üst sol alt yanlış tahminler
cm = confusion_matrix(y_test, y_pred)

#homework
# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
y_pred_hw = classifier.predict(sc_x.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
y_pred_hw = (y_pred_hw > 0.5)

# Part 4 - Evaluating, Improving and Tuning the ANN
#evaluating - k fold cross validation kullanılır
#veri setleri k parçaya ayrılır ve her bir parça tekrar parçalara ayrılır her batchte farklı
#bir parça üzerinden hesap yapılarak accuracyler incelenir ve bir yakınsama elde edilmeye çalışılır
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer ='uniform'))
    classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer ='uniform'))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer ='uniform')) 
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#fit kısmını fonksiyonla yapıyoruz
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
#tüm cpuları kullan 10 kere cross val yap
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std() #low bias low variance high mean istiyoruz

#improving the ann
#dropout regularization to reduce overfitting if needed
#dropoutta her iterasyonda rastgele nöranlar kapatılır böylece her birinin birbirlerine aşırı bağlı olması önlenir
#öğrenme sırasında korelasyon olduğundan böylece birçok verideki bağımsız korelasyon yok edilmiş olur
#öğrenme işleminin bağımsız olması sağlanır ve overfitting önlenmiş olur
#input katmanından sonra dropout katmanı eklenir - classifier.add(Dropout(p = 0.1))

#tuning the ann - burada sürekli deneme ile farklı parametrelerde nasıl sonuç alınıyor
#onun taramasını yaparız ve en iyi parametreleri seçeriz
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer ='uniform'))
    classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer ='uniform'))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer ='uniform')) 
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#fit kısmını fonksiyonla yapıyoruz
classifier = KerasClassifier(build_fn = build_classifier)
#modeldeki hyper parametreler yazılır ve hangi değerleri denemek istiyorsak yanlarına onlar yazılır
parameters = {'batch_size': [25, 32], #bu bir sözlük
              'epochs': [100, 500], #buraya ne yazıyorsak hepsini kombinasyonlarıyla deniyor onun için buraya istediğimiz gibi yazmak çok uzun sürer
              'optimizer': ['adam', 'rmsprop']} #bu 6 değeri denemesi kombine edip denemesi sekiz(8) saatten fazla sürdü
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10) #cv = number of fold
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_ #en iyi parametreleri
best_accuracy = grid_search.best_score_
#bu 8 saatlik tuning sonucunda elde edilen best accuracy = 0.84725
#bu best accuracy için gereken best parameter = batch_size = 32, epochs = 500, optimizer = adam