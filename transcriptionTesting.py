# Imports
import tensorflow as tf
print(tf.__version__)
import h5py, os
import numpy as np
import matplotlib.pyplot as plt
fileI = "train.mat" # In some cases(eg, VSC running .ipynb file) this path may need to be defined statically.
fileVal = "valid.mat"
print("Everything is more or less working.")
# Loading data
# This cell uses a LOT of RAM and battery power.
# train.mat contains 3.4GB of data, but, in some cases,
# RAM usage is much higher. This could be caused by:
"""
- Other processes
- The source running the .ipynb file(eg VSC has many extra features == extra RAM usage)
- Multiple variables storing the file; shallow copying may account for this, but it still remains a problem.
"""
 
print(os.path.exists(fileI))
f = h5py.File(fileI, "r+")
print(f.keys())
inputs = f.get("trainxdata")[()]
print(inputs.shape)
print(inputs[0])
outputs = f.get("traindata")[()]
print("Loaded train data.")
# Outputs are 919 chromatin features.
# 125 DNase features, 690 TF features, and 104 histone predictions.
# Our inputs, on the other hand,
# Are of length (1000, 4)
# Deleting the initial training array(f) aids in decreasing memory usage a little.
del f
print(inputs[0])
threshold = int(len(inputs) * 3/4)
inputVal = np.transpose(inputs, (2, 0, 1))[threshold:]
outputVal = outputs[:,threshold:].T
print(inputVal.shape, outputVal.shape)
inputs = np.transpose(inputs, (2, 0, 1))[:threshold]
outputs = outputs[:,:threshold].T
print(inputs.shape, outputs.shape)
# Here is some data to show
# This is the input; we can only show a certain
# Amount of letters before it doesn't look nice anymore,
# So here is a barcode-like representation:
 
# plt.imshow(inputs[8][:100])
# plt.show()
# Here is the output(validation set), displayed as a bar code.
def argmaxFunc(a):
    return np.array([np.argmax(i) for i in a])
# argmaxFunc = np.vectorize(myFunc)
def draw(inputs, outputs, max_len=200):
    subplots = plt.subplots(200, 2)
    for i in range(len(subplots[1])):
        subplots[1][i][0].set_axis_off()
        subplots[1][i][0].imshow(outputs[i].reshape((1, -1)), aspect="auto", cmap="binary", interpolation=None)
        subplots[1][i][1].set_axis_off()
        subplots[1][i][1].imshow(argmaxFunc(inputs[i]).reshape((1, -1)), aspect="auto", interpolation=None)
    return plt
draw(inputVal, outputVal).show()
 
# Looking at the data
infoFile = "journal.pcbi.1007616.s007.xlsx"
import xlrd
fullCellData = []
workbook = xlrd.open_workbook(infoFile)
worksheet = workbook.sheet_by_index(0)
for row in range(1, worksheet.nrows):
    fullCellData.append({
        "cell": worksheet.cell_value(row,0),
        "regulatory element": worksheet.cell_value(row,1),
        "treatment": worksheet.cell_value(row, 2)
    })
# Alright, we're going to 
# - Create a function which takes in the output array,
# - Puts it in alignment with all the other info we have,
# - And graphs the normalized probability of that occurring.
# fullCellData = np.array(fullCellData)
def visualizeOutput(output):
    # Taken in output, let's try a histogram.
    val = 0 # Where the data will appear
    # We can plot the data as a simple scatter plot, as shown here.
    plt.plot(output, color="green", marker="o", markersize=2)
    plt.show()
    # We can transform it into a pie chart, which will take a little more work....
    chart = fullCellData
    # Now we can show a chart
    for i in range(len(chart)):
        chart[i]["output"] = output[i]
    chart.sort(key=lambda val: val["output"], reverse=True)
    # print(chart[:10])
    outputPush = []
    for i in range(len(chart)):
        if i % 100 == 0:
            print("%d percent through" % int(i / len(chart)*100))
        outputPush.append(list(chart[i].values()))
    plt.table(cellText=outputPush,
    loc="center",
    colLabels=["Cell Lines", "Regulatory Element", "Treatment", "Probability"])
    plt.savefig("PDF.pdf")
    plt.show()
# np.savetxt("info.csv", outputVal[0], delimiter=",")
visualizeOutput(outputVal[0])
 
# # Now there is some more file assorting.
# # We have a file with the names of all the chromatin features we're looking for; problem is, they are in files.txt.
# fileO = []
# timesBefore = 1
# for i in open("files.txt", "r+").read().split("\n"):
#     if len(i.split("; ")) >= 6:
#         if len(fileO) > 0:
#             cellName = i.split("; ")[6][5:]
#             if cellName != fileO[-1]:
#                 fileO.append(cellName)
#             else:
#                 print("Skipped " + str(timesBefore))
#                 timesBefore += 1
#         else:
#             fileO.append(i.split("; ")[6][5:])
#     else:
#         print("SKIPPED")
# print(len(fileO), fileO)
# Testing whether we have a GPU or not
# As of TF 2, GPU support is used by default, so this only applies
# If we have TF version < 2.
if int(tf.__version__[0]) < 2:
    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU # This checks if it can use CuDNNGRU.
        print("GPU support enabled.")
    else:
        import functools
        rnn = functools.partial(
          tf.keras.layers.GRU, recurrent_activation='tanh')
        print("GPU not found, defaulting to CPU.")
else:
    rnn = tf.keras.layers.GRU
    if tf.test.is_gpu_available():
        print("GPU support enabled.")
    else:
        print("GPU will NOT be used. Make sure Cuda is in your PATH.")
 
# F1 metric
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
# Now we're going to work on the actual model.
# --------TEST MODEL----------
def Model(input_shape, output_shape, unit1=128, unit2=128):
    inputs = tf.keras.Input(input_shape[1:])
    # Simple RNN
    recurrent = rnn(unit1, return_sequences=True)(inputs)
    recurrent1 = rnn(unit2, return_sequences=False)(recurrent)
    dense1 = tf.keras.layers.Dense(output_shape[1], activation="sigmoid")(recurrent1) # return_state
    # Dense layer
    # dense = tf.keras.layers.Dense(919)(recurrentLayer) #919
    print(dense1.shape, inputs.shape)
    model = tf.keras.Model(inputs=inputs, outputs=dense1)
    return model
 
# This very simple model is merely a proof of purpose.
# Just to see whether the shapes work, whether the GRU
# Performs more or less correctly, etc.
# Actual training
# Creating the model
print(tf.__version__)
import tensorflow.keras.backend as K
print(inputs.shape, outputs.shape)
print(len(inputs[0][0]))
model = Model(inputs.shape, outputs.shape)
# We will need to reverse the shape in order for this to work
# Work properly, but first let's check whether this works.
accReadings = []
f1Readings = []
perplexity = []
class MyCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        # print(logs["loss"], tf.math.exp(logs["loss"]))
        perplexity.append(np.exp(logs["loss"]))
        accReadings.append(logs["loss"])
        print(np.exp(logs["loss"]))
        print("Batch ended with a perplexity of %2d" % np.exp(logs["loss"]))
    def on_epoch_end(self, batch, logs=None):
        draw(inputVal, model.predict(inputVal)).show()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy", f1_metric])
model.summary()
# 
# Now that the model is created, we finally
# Can try out the training.
# Proper model
# Taken from DeepSea model
def Model(input_shape, output_shape):
    # Input
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(input_shape[1:]))
    """
    "The basic layer types in out model are convolution layer, pooling layer and fully connected layer. A convolution layer computes output by one-dimensional convolution operation with a specified number of kernels . . . . In the first convolution layer, each kernel can be considered as a position weight matrix(PWM) and the convolution operation is equivilent to computing the PWM scores with a moving window with step size one on the sequence."
    """
    """Here is the code in Lua:
    
model:add(nn.SpatialConvolutionMM(nfeats, nkernels[1], 1, 8, 1, 1, 0):cuda())
model:add(nn.Threshold(0, 1e-6):cuda())
model:add(nn.SpatialMaxPooling(1,4,1,4):cuda())
model:add(nn.Dropout(0.2):cuda())
 
model:add(nn.SpatialConvolutionMM(nkernels[1], nkernels[2], 1, 8, 1, 1, 0):cuda())
model:add(nn.Threshold(0, 1e-6):cuda())
model:add(nn.SpatialMaxPooling(1,4,1,4):cuda())
model:add(nn.Dropout(0.2):cuda())
 
model:add(nn.SpatialConvolutionMM(nkernels[2], nkernels[3], 1, 8, 1, 1, 0):cuda())
model:add(nn.Threshold(0, 1e-6):cuda())
model:add(nn.Dropout(0.5):cuda())
 
nchannel = math.floor((math.floor((width-7)/4.0)-7)/4.0)-7
model:add(nn.Reshape(nkernels[3]*nchannel))
model:add(nn.Linear(nkernels[3]*nchannel, noutputs))
model:add(nn.Threshold(0, 1e-6):cuda())
model:add(nn.Linear(noutputs , noutputs):cuda())
model:add(nn.Sigmoid():cuda())   
"""
    nkernels = [4, 320,480,960]
    dropout = [0.2, 0.2, 0.5]
    # We have 3 rounds of convolution. Each one contains a convolution layer with output of kernel size nkernels[i], a threshold(which we can implement later...), and a dropout.
    model.add(tf.keras.layers.Conv1D(4, 2, 1, "valid"))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.Dropout(dropout[i]))
    model.summary()
    return model
print(inputs.shape, outputs.shape)
model = Model(inputs.shape, outputs.shape)
