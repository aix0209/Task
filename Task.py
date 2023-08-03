import FlowCal as FC
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as ks
import openpyxl


ChannelFile = openpyxl.load_workbook("EU_marker_channel_mapping.xlsx",data_only = True)
s1=ChannelFile.active
Usecolumn = s1["B"]
ChannelColumn = s1["D"]
ChannelList=[]
for NumA in range(1,len(Usecolumn)):
    if Usecolumn[NumA].value == 1:
        ChannelList.append(ChannelColumn[NumA].value)

ResuleFile = openpyxl.load_workbook("EU_label.xlsx",data_only = True)
s2=ResuleFile.active
Resultcolumn = s2["B"]
Result= []
for NumA in range(1,len(Resultcolumn)):
    Result.append(Resultcolumn[NumA].value)


FCSDir = os.listdir("raw_fcs/")
OutputDate = np.zeros((len(FCSDir)))
Size=["0"]
for NumA in range(len(FCSDir)):
    if Result[NumA]=="Healthy":
        OutputDate[NumA]= 0.0
    elif Result[NumA]=="Sick":
        OutputDate[NumA]= 1.0

for NumA in range(len(FCSDir)):
    FCSfiles = os.listdir("raw_fcs/"+FCSDir[0]+"/")

    Data = FC.io.FCSData("raw_fcs/"+FCSDir[0]+"/"+FCSfiles[0])
    if Size[0]=="0":
        Size = Data.shape
        InputDate = np.zeros((len(FCSDir),len(ChannelList),Size[0]))

    for NumB in range(len(ChannelList)):
        InputDate[NumA,NumB,:] = Data[:,ChannelList[NumB]]
#print(InputDate[0])

InputDate=InputDate/1000000.0
print(InputDate)
model = ks.Sequential([
    ks.layers.Flatten(input_shape=(len(ChannelList),Size[0])),
    ks.layers.Dense(8, activation='relu'),
    ks.layers.Dropout(0.2),
    ks.layers.Dense(2)
])
model.compile(optimizer='adam',
              loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(InputDate, OutputDate, epochs=10)
test_loss, test_acc = model.evaluate(InputDate, OutputDate, verbose=2)
print(test_loss, test_acc)
probability_model = ks.Sequential([model,ks.layers.Softmax()])
predictions = probability_model.predict(InputDate)
print(predictions)

