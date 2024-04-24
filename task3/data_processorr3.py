import ipaddress
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split

df = pd.read_csv("../Darknet.csv")

df = df.drop(["Flow ID", "Timestamp", "Label2"], axis=1)

df = df.dropna()

df['Src IP'] = df['Src IP'].apply(lambda x: int(ipaddress.ip_address(x)))
df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ipaddress.ip_address(x)))

for col in ['Src Port', 'Dst Port', 'Protocol']:
    df[col] = df[col].astype(int)

label_encoder1 = LabelEncoder()
df['Label1'] = label_encoder1.fit_transform(df['Label1'])

df.to_csv("processed.csv", index=False)

scaler = RobustScaler()

# Handle infinite or excessively large values
def robust_scale_skip_large(X, scaler):
    try:
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    except ValueError as e:
        print(f"Skipping scaling: {str(e)}")
        return X

scaled_features = robust_scale_skip_large(df.drop('Label1', axis=1), scaler)

scaled_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])

scaled_df['Label1'] = df['Label1']

scaled_df.to_csv("scaled3.csv", index=False)

scaled_df = pd.read_csv("scaled3.csv")

features = scaled_df.drop(['Label1'], axis=1)
label = scaled_df['Label1']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Write accuracy to accuracy3.txt
with open('accuracy3.txt', 'w') as f:
    f.write(f'Test Loss: {loss:.4f}\n')
    f.write(f'Test Accuracy: {accuracy:.4f}\n')
