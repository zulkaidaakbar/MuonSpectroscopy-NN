from tensorflow import keras
import numpy as np
import pandas as pd
from numpy import asarray
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0075,
    decay_steps=10000,
    decay_rate=0.996)

# get the model

def get_model(n_inputs, n_outputs):
        model = Sequential()
        model.add(Dense(25, input_dim=n_inputs, kernel_initializer='he_uniform', activation = 'elu'))
        #model.add(Dropout(0.2))
        model.add(Dense(25, kernel_initializer='he_uniform', activation = 'elu'))
        model.add(Dense(25, kernel_initializer='he_uniform',activation = 'elu'))
        model.add(Dense(25, kernel_initializer='he_uniform',activation = 'elu'))
        model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
        #opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='mse', optimizer=opt)
        return model


#early stopping
keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.00000001)
]

data = np.loadtxt("ML_data_1.txt")
t = data[:,[0]]
A_t = data[:,[1]]
A_t_err = data[:,[2]]

mask = (A_t > 0) & ( t < 5.0 )
n = np.size(t[mask])
X = t[mask]
y = A_t[mask]

n_inputs, n_outputs = 1, 1

yrep = y
# get model & fit
model = get_model(n_inputs, n_outputs)
#history = model.fit(X, yrep, validation_split =0.15, verbose=0, epochs=20000, callbacks = keras_callbacks, shuffle=True)
history = model.fit(X, yrep, validation_split =0.15, verbose=0, epochs=40000)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("loss.png")
plt.clf()

# make a prediction for test data
result_arr = np.array([])
for ii in range(n):
 inp_t = X[ii]
 row = [inp_t]
 newX = asarray([row])
 yhat = model.predict(newX)
 print(' %.4f %.4f' %( inp_t, yhat[0]))
 result_arr = np.append(result_arr, yhat[0])
 #print('%.4f %.4f' %(xb, yhat[0]))
print("=============")

plt.errorbar(t[mask], A_t[mask], A_t_err[mask],fmt='.', label='Data', color='green', elinewidth=2, capsize = 3, alpha = 0.3)
plt.plot(t[mask], result_arr,'-' ,label="", color = 'red')
plt.xlim(0,5)
plt.ylim(15,40)
plt.legend(loc=1,fontsize=10,handlelength=3)
plt.xlabel('t (micro-second)', fontsize = 15)
plt.ylabel('A (%)', fontsize = 15)
plt.savefig('test.png')
plt.clf()

dev = np.array([])
dev = (result_arr - A_t[mask])/A_t_err[mask]
plt.hist(dev)
plt.savefig('histo.png')
plt.clf()

print(np.mean(dev))
print(np.std(dev))
