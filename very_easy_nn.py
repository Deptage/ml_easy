import tensorflow as tf
from tensorflow import keras
import pandas as pd
import random as rd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

def func(x):
    return 10*x*x*x+42*math.sin(x*x)+28*math.sin(x)*math.sin(x)-69

X=[]
y=[]

for _ in range(0,1000000):
    temp=rd.random()
    temp=temp*math.pi*4
    temp=temp-2*math.pi
    X.append(temp)
    y.append(func(temp)+10*rd.random())

df=pd.DataFrame(np.column_stack([X, y]),columns=["X","y"])
print(df)

plt.plot(X,y,'o')
plt.show()


model=keras.Sequential([
    keras.layers.Dense(50,activation='relu',input_shape=(1,)),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(10,activation='relu'),
    keras.layers.Dense(10,activation='relu'),
    keras.layers.Dense(10,activation='relu'),
    keras.layers.Dense(10,activation='relu'),
    keras.layers.Dense(10,activation='relu'),
    keras.layers.Dense(1),
])
model.summary()

X=np.array(X)
y=np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,random_state=1697)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.compile(optimizer='adam',loss=keras.losses.MeanSquaredError())
history=model.fit(X_train,y_train,epochs=50,validation_split=0.3,batch_size=128,callbacks=[early_stopping_monitor])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(model.predict(X_test))
print(y_test)
model.evaluate(X_test,y_test)

# summarize history for loss
plt.plot(history.history['loss'][5:])
plt.plot(history.history['val_loss'][5:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
