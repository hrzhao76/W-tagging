from array import array
import h5py
import numpy as np
import matplotlib.pyplot as plt
import keras

train_file = h5py.File("/Volumes/MacOS/Research/Data/Images/train_no_pile_10000000.h5", "r")
test_file = h5py.File("/Volumes/MacOS/Research/Data/Images/test_no_pile_5000000.h5", "r")

X_train = train_file['features']
X_test = test_file['features']

y_train = train_file['targets']
y_test = test_file['targets']


# N_train = len(X_train)
# N_test = len(X_test)
N_train = N_test = 10000
train_images=np.array(X_train[0:N_train])
train_labels=np.array(y_train[0:N_train])

test_images=np.array(X_test[0:N_test])
test_labels=np.array(y_test[0:N_test])

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.LocallyConnected2D(32,(4,4),activation='tanh', input_shape=(1, 32, 32), kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)),
    keras.layers.LocallyConnected2D(32,(4,4),activation='tanh'),
    keras.layers.LocallyConnected2D(32,(4,4),activation='tanh'),
    keras.layers.LocallyConnected2D(32,(4,4),activation='tanh'),
    
    
    keras.layers.Flatten(), # 

    keras.layers.Dense(425, activation='tanh'),
    keras.layers.Dense(425, activation='tanh'),
    keras.layers.Dense(425, activation='tanh'),
    keras.layers.Dense(425, activation='tanh'),
    
    keras.layers.Dense(1, activation='sigmoid')
])

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.9, amsgrad=False)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("OK!, then fit!")
history = model.fit(train_images, train_labels, validation_split = 0.1, batch_size = 100, epochs=50, verbose=1)

plt.figure(figsize=(12, 12))
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accu', 'val_accu'], loc='upper left') 
plt.subplot(2,1,2)
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left') 

plt.savefig("performance")
plt.show()

 
model.save('my_NN-paper.h5')
