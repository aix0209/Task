import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, AveragePooling2D, Flatten,LeakyReLU
from keras import regularizers 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
import keras.utils.layer_utils 
import sys
import numpy as np
import random
#
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Convolution2D, AveragePooling2D, Flatten
#from keras.regularizers import l2, activity_l2
#from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint, EarlyStopping, History
#import keras.utils.layer_utils
#import numpy as np
date = time.strftime('%Y_%m_%d_%H_%M_%S')
DataNum=20000#樣本數
Dim1 = 20#維數
Dim2 = 20#維數
Channel=1
filterNum=3
ConvNum=6
poolintNum=0
print("資料建立中...")
InputDate=np.zeros((DataNum,Dim1,Dim2,Channel))#必須四維
OutputDate=np.zeros((DataNum,2))#輸出Dim2可自訂
Stats=np.zeros(2)
for LoopNumA in range(0,DataNum,1):
	SumNum=0
	for LoopNumB in range(0,Dim1,1):
		for LoopNumC in range(0,Dim2,1):
			ram=random.randint(-1,1)
			InputDate[LoopNumA][LoopNumB][LoopNumC][Channel-1]= ram
			SumNum=SumNum+InputDate[LoopNumA][LoopNumB][LoopNumC][Channel-1]
	if SumNum>=0:
		OutputDate[LoopNumA][0]=1
	else:
		OutputDate[LoopNumA][1]=1
print("學習參數設定中...")

model = Sequential()
for ConvLoopNum in range(0,ConvNum,1):
	model.add(Conv2D(input_shape=(Dim1,Dim2,Channel),filters=filterNum, kernel_size=(filterNum,filterNum),dilation_rate=(1, 1),\
		strides=(1,1), padding="same",\
		data_format='channels_last', activation=None, use_bias=True,\
		kernel_initializer='random_uniform', bias_initializer='zeros',\
		kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None,\
		activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

	model.add(LeakyReLU(alpha=0))
	if (ConvLoopNum+1)%3==0:
		model.add(AveragePooling2D(pool_size=(2,2)))
		poolintNum=poolintNum=+1
	if poolintNum==0:
		model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(2,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('softmax'))
 
w = model.get_weights()
#dim=np.shape(w[0])
#print(dim)
for WLoopNum1 in range(0,Channel):
	for WLoopNum2 in range(0,filterNum):
		w[0][:,:,WLoopNum1,WLoopNum2] = (-1)**(WLoopNum1+WLoopNum2)/filterNum**2
####

####
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(), metrics=['accuracy'])
              

print("學習中...")

model.fit( InputDate, OutputDate, batch_size=29, epochs=1000, verbose=1, callbacks=None,\
	validation_split=0.0, validation_data=None, shuffle=True, \
	class_weight=None, sample_weight=None, initial_epoch=0)
loss_acc = model.evaluate(InputDate,OutputDate, verbose=0)
date = time.strftime('%Y_%m_%d_%H_%M_%S')
filepath="./Learnpracitice"+date+".hdf5".format(loss_acc[0],loss_acc[1])
w2 = model.get_weights()
#model.save_weights(filepath)
print(filepath)

TestData=np.zeros((1,Dim1,Dim2,Channel))
for LoopNumA in range(0,1,1):
	SumNum=0
	for LoopNumB in range(0,Dim1,1):
		for LoopNumC in range(0,Dim2,1):
			ram=random.randint(-1,1)
			TestData[LoopNumA][LoopNumB][LoopNumC][Channel-1]= ram
			SumNum=SumNum+TestData[LoopNumA][LoopNumB][LoopNumC][Channel-1]
result = model.predict(TestData)
print(SumNum)
print(result)
print(Stats)
