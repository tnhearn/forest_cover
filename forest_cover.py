import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Import data from CSV to dataframe
data = pd.read_csv('cover_data.csv')

# Inspect data
data.head()
data.info()
data_desc = data.describe()

# Split  data into features/labels
X = data.loc[:, 'Elevation':'Soil_Type40']
y = data.loc[:, 'class']

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    shuffle = True,
    stratify = None
    )

# Select the columns to scale
cols_to_scale = ['Elevation', 'Aspect', 'Slope', 
                 'Horizontal_Distance_To_Hydrology', 
                 'Vertical_Distance_To_Hydrology', 
                 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 
                 'Hillshade_Noon', 'Hillshade_3pm', 
                 'Horizontal_Distance_To_Fire_Points']

# Create the scaler object
scaler = StandardScaler()

# Fit the scaler to the selected columns
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def design_model(X):
    # Initialize model
    model = Sequential()
    # Create input layer
    model.add(InputLayer(input_shape = (X.shape[1],)))
    # Create hidden layer
    model.add(Dense(128, activation = 'relu'))
    # Apply dropout
    model.add(Dropout(.2))
    # Create hidden layer
    model.add(Dense(64, activation = 'relu'))
    # Apply dropout
    model.add(Dropout(.2))
    # Create hidden layer
    model.add(Dense(32, activation = 'relu'))
    # Apply dropout
    model.add(Dropout(.2))
    # Create hidden layer
    model.add(Dense(16, activation = 'relu'))
    # Apply dropout
    model.add(Dropout(.2))
    # Creat output layer (8 nodes bc data has eight classes)
    model.add(Dense(8, activation = 'softmax'))
    # Create optimizer
    opt = Adam(learning_rate = 0.001)
    # Compile model
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = opt,
        metrics = ['accuracy']
        )
    return model

# Apply the model
model = design_model(X_train_scaled)

# Add EarlyStopping for effiency
es = EarlyStopping(
    monitor = 'val_accuracy', 
    mode = 'min',
    verbose = 1,
    patience = 20
    )

# Fit the model
b_size = 100
n_epochs = 50
history = model.fit(X_train_scaled, 
                    y_train, 
                    batch_size = b_size,
                    epochs = n_epochs,
                    validation_split = 0.2,
                    verbose = 1,
                    callbacks = [es]
                    )

# Create model summary
model.summary()

# Evaluate the model
loss, acc = model.evaluate(
    X_test_scaled, 
    y_test, 
    verbose = 1)

# Make prediction
y_pred = model.predict(X_test_scaled)
# Convert prediction to screte values
y_pred = np.argmax(y_pred, axis = 1)
class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine',
               'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
# Print classification report
print(classification_report(
    y_test,
    y_pred, 
    target_names = class_names)
    )

# Plot Accuracy and Validation Accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(['Train', 'Validation'], loc='upper left')

# Plot Loss and Validation Loss over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(['Train', 'Validation'], loc='upper left')

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax3 = plt.subplots(figsize=(15, 15))
heatmap = sns.heatmap(
    cm, 
    fmt = 'g', 
    cmap = 'mako_r', 
    annot = True, 
    ax = ax3)
ax3.set_xlabel('Predicted class')
ax3.set_ylabel('True class')
ax3.set_title('Confusion Matrix')
ax3.xaxis.set_ticklabels(class_names)
ax3.yaxis.set_ticklabels(class_names)








