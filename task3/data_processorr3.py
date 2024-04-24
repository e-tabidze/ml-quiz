import ipaddress
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv("Darknet.csv")

df = df.drop(["Flow ID", "Timestamp", "Label2"], axis=1)

df = df.dropna()

df['Src IP'] = df['Src IP'].apply(lambda x: int(ipaddress.ip_address(x)))
df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ipaddress.ip_address(x)))

label_encoder = LabelEncoder()
df['Label1'] = label_encoder.fit_transform(df['Label1'])

df.to_csv("processed.csv", index=False)

problematic_columns = []
for col in df.columns:
    if df[col].dtype == np.float64 or df[col].dtype == np.int64:
        max_value = df[col].max()
        min_value = df[col].min()
        if max_value == np.inf or min_value == -np.inf:
            problematic_columns.append(col)

if problematic_columns:
    print("Skip incorrect rows. remove them")
    df = df[~df[problematic_columns].isin([np.inf, -np.inf]).any(axis=1)]

features = df.drop(['Label1'], axis=1)
label = df['Label1']

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['Label1'] = label
scaled_df.to_csv("scaled3.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(scaled_features, label, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(scaled_features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

with open('accuracy3.txt', 'w') as f:
    f.write(f'Test Loss: {loss:.4f}\n')
    f.write(f'Test Accuracy: {accuracy:.4f}\n')
