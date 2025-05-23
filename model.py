from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import itertools

df = pd.read_csv("hand_landmarks_dataset.csv")
df.drop(columns=['label', 'path'], inplace=True)



X, y = df.drop(columns='label_id'), df['label_id']

minmax = MinMaxScaler()
X_scaled = minmax.fit_transform(X)

window_size = 37
n_samples = len(X_scaled) // window_size

y_seq = np.array(y[:n_samples * window_size]).reshape(n_samples, window_size)

y_seq = y_seq[:, 0] 

X_scaled = X_scaled.reshape(-1, 37, 63)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_seq, train_size=0.8, random_state=13)

print("Уникальные метки до преобразования:", np.unique(y_train))

y_train = to_categorical(y_train, num_classes=27)
y_test = to_categorical(y_test, num_classes=27)

print("Форма y_train после преобразования:", y_train.shape)
best_acc = 0
best_model = 0
param_grid = {
    'units': [16, 32, 50, 64, 128],
    'recurrent_dropout': [0.1, 0.2, 0.3, 0.4],
    'dropout': [0.1, 0.2, 0.3, 0.4],
    'dense_units': [8, 16, 32, 64]
}

# Генерируем все комбинации
all_combinations = list(itertools.product(
    param_grid['units'],
    param_grid['recurrent_dropout'],
    param_grid['dropout'],
    param_grid['dense_units']
))

# Перебор всех комбинаций
for i, (units, rec_drop, drop, dense_units) in enumerate(all_combinations):
    print(f"\nКомбинация {i+1}/{len(all_combinations)}: "
          f"units={units}, rec_drop={rec_drop}, drop={drop}, dense_units={dense_units}")
    model = Sequential([
        GRU(units, 
            input_shape=(37, 63), 
            kernel_regularizer=l2(0.01), 
            recurrent_regularizer=l2(0.01),                      
            recurrent_dropout=rec_drop,),
        Dropout(drop),
        Dense(dense_units, activation='relu'),
        Dense(27, activation='softmax')  
    ])

    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    
    early_stop = EarlyStopping(
        monitor='val_loss',     
        patience=10,            
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        f"model_{i}_best.h5",  
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    callbacks = [early_stop, checkpoint]

    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks  
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Тестовая точность: {test_acc}")
    model.save(f"model_{i}.h5")
    if (test_acc > best_acc):
        best_acc = test_acc
        best_model = i

print(f"best_model: {best_model}, best acc of this model: {best_acc}")

