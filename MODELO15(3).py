
## LO QUE HAGO EN STE MODELO ES COGER TODOS LOS DIAS DE LA SECUENCIA MENOS EL ULTIMO Y TRATA DE ADIINAR LA TENDENCIA PARA EL DIA SIGUIENTE (que es el dia final de la secuencia)
#  ESTE SCRIPT PINTA BIEN PERO NO TENGO MEMEORIA FUNCIONA!!!
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Concatenate, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Definir la ruta del directorio y listar todos los archivos CSV
directory_path = "C:/Users/s0141677/OneDrive - THALES SA/Documents/TFG - ALEJANDRO ALONSO ANDA/DESCARGA DE DATOS/DESCARGA DE DATOS 3/3.DATOS HISTORICOS ACCIONES CON INDICE"
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

def split_data(df, train_size=2/3):
    """ Divide los datos en entrenamiento (1/3) y prueba (2/3) basado en el tiempo. """
    total_rows = len(df)
    train_rows = int(total_rows * train_size)
    return df.iloc[:train_rows], df.iloc[train_rows:]

# Diccionario para almacenar los DataFrames de cada compañía
company_data_dict = {}
company_mapping = {}
n_steps = 120

for file_name in csv_files:
    file_path = os.path.join(directory_path, file_name)
    df = pd.read_csv(file_path)
    
    # Convertir la columna 'Date' a datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= '2002-01-01') & (df['Date'] <= '2024-01-01')]
    
    # Asegurarse de que los datos están ordenados por fecha
    df.sort_values('Date', inplace=True)

    # Suponer que el nombre de la compañía es único y constante en todo el archivo
    company = df['Company'].iloc[0]  # Tomar el nombre de la compañía del primer registro

    for company in df['Company'].unique():
        if company not in company_mapping:
            company_mapping[company] = len(company_mapping)
        if company not in company_data_dict:
            company_data_dict[company] = df[df['Company'] == company]
        else:
            company_data_dict[company] = pd.concat([company_data_dict[company], df[df['Company'] == company]], ignore_index=True)

print(f"Total de compañías procesadas: {len(company_data_dict)}")
print(f"Índices de compañías: {company_mapping}")

# Función para crear secuencias dentro de cada compañía
def create_sequences(df, n_steps):
    features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_30', 'SMA_50', 'EMA_30', 'EMA_50', 'MACD', 'Signal_Line', 'RSI', 'Upper_Band', 'Lower_Band', 'Trend']].values.astype('float32')
    sp500_features = df[['SP500_Open', 'SP500_High', 'SP500_Low', 'SP500_Close', 'SP500_Adj Close', 'SP500_Volume', 'SP500_SMA_30', 'SP500_SMA_50', 'SP500_EMA_30', 'SP500_EMA_50', 'SP500_MACD', 'SP500_Signal_Line', 'SP500_RSI']].values.astype('float32')
    labels = df['Trend'].values.astype('float32')

    X, y, X_sp500 = [], [], []
    indices = []
    for i in range(len(features) - n_steps):
        seq_features = features[i:i + n_steps - 1]
        seq_sp500_features = sp500_features[i:i + n_steps - 1]
        seq_labels = labels[i + n_steps - 1]
        X.append(seq_features)
        X_sp500.append(seq_sp500_features)
        y.append(seq_labels)
        indices.append(np.full((n_steps - 1,), company_mapping[df['Company'].iloc[i]]))
    
    return np.array(X), np.array(X_sp500), np.array(y), np.array(indices)


all_X_train, all_X_test, all_y_train, all_y_test, all_X_sp500_train, all_X_sp500_test = [], [], [], [], [], []
all_indices_test, all_indices_train, = [], []

for company, df in company_data_dict.items():
    company_index = company_mapping[company]  # Obtener el índice asignado a la compañía
    print(f"Procesando compañía: {company}, Índice: {company_index}")

    # Utilizar la función de división
    train_df, test_df = split_data(df)
    
    # Estandarizar datos de entrenamiento y prueba
    scaler = StandardScaler()
    train_features = train_df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_30', 'SMA_50', 'EMA_30', 'EMA_50', 'MACD', 'Signal_Line', 'RSI', 'Upper_Band', 'Lower_Band', 
                               'SP500_Open', 'SP500_High', 'SP500_Low', 'SP500_Close', 'SP500_Adj Close', 'SP500_Volume', 'SP500_SMA_30', 'SP500_SMA_50', 'SP500_EMA_30', 'SP500_EMA_50', 'SP500_MACD', 'SP500_Signal_Line', 'SP500_RSI']]
    test_features = test_df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_30', 'SMA_50', 'EMA_30', 'EMA_50', 'MACD', 'Signal_Line', 'RSI', 'Upper_Band', 'Lower_Band', 
                             'SP500_Open', 'SP500_High', 'SP500_Low', 'SP500_Close', 'SP500_Adj Close', 'SP500_Volume', 'SP500_SMA_30', 'SP500_SMA_50', 'SP500_EMA_30', 'SP500_EMA_50', 'SP500_MACD', 'SP500_Signal_Line', 'SP500_RSI']]
    
    scaler.fit(train_features)
    train_features_scaled = scaler.transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Asignar las características escaladas de vuelta a los DataFrames originales
    train_df.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_30', 'SMA_50', 'EMA_30', 'EMA_50', 'MACD', 'Signal_Line', 'RSI', 'Upper_Band', 'Lower_Band', 
                     'SP500_Open', 'SP500_High', 'SP500_Low', 'SP500_Close', 'SP500_Adj Close', 'SP500_Volume', 'SP500_SMA_30', 'SP500_SMA_50', 'SP500_EMA_30', 'SP500_EMA_50', 'SP500_MACD', 'SP500_Signal_Line', 'SP500_RSI']] = train_features_scaled
    test_df.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_30', 'SMA_50', 'EMA_30', 'EMA_50', 'MACD', 'Signal_Line', 'RSI', 'Upper_Band', 'Lower_Band', 
                    'SP500_Open', 'SP500_High', 'SP500_Low', 'SP500_Close', 'SP500_Adj Close', 'SP500_Volume', 'SP500_SMA_30', 'SP500_SMA_50', 'SP500_EMA_30', 'SP500_EMA_50', 'SP500_MACD', 'SP500_Signal_Line', 'SP500_RSI']] = test_features_scaled

    # Verificar que los datos se estandarizaron correctamente
    print(f"Valores originales de entrenamiento para la compañía {company}:")
    print(train_features[:5])
    print(f"Valores estandarizados de entrenamiento para la compañía {company}:")
    print(train_df[['Open', 'Close', 'SP500_Open', 'SP500_Close']].head())

    print(f"Valores originales de prueba para la compañía {company}:")
    print(test_features[:5])
    print(f"Valores estandarizados de prueba para la compañía {company}:")
    print(test_df[['Open', 'Close', 'SP500_Open', 'SP500_Close']].head())

    # Crear secuencias
    X_train, X_sp500_train, y_train, indices_train = create_sequences(train_df, n_steps)
    X_test, X_sp500_test, y_test, indices_test = create_sequences(test_df, n_steps)

    all_X_train.extend(X_train)
    all_X_test.extend(X_test)
    all_y_train.extend(y_train)
    all_y_test.extend(y_test)
    all_X_sp500_train.extend(X_sp500_train)
    all_X_sp500_test.extend(X_sp500_test) 
    all_indices_train.extend(indices_train)
    all_indices_test.extend(indices_test) # Extender la lista de índices para cada secuencia

# Convertir listas a numpy arrays para uso posterior en modelos de machine learning
all_X_train = np.array(all_X_train)
all_X_sp500_train = np.array(all_X_sp500_train)
all_y_train = np.array(all_y_train)
all_indices_train = np.array(all_indices_train)

all_X_test = np.array(all_X_test)
all_X_sp500_test = np.array(all_X_sp500_test)
all_y_test = np.array(all_y_test)
all_indices_test = np.array(all_indices_test)

print(f"esta es la forma de all_X_train: {all_X_train.shape} ")
print(f"esta es la forma de all_X_sp500_train: {all_X_sp500_train.shape} ")
print(f"esta es la forma de all_indices_train: {all_indices_train.shape} ")
print(f"esta es la forma de all_y_train: {all_y_train.shape} ")

print(f"esta es la forma de all_X_test: {all_X_test.shape} ")
print(f"esta es la forma de all_X_sp500_test: {all_X_sp500_test.shape} ")
print(f"esta es la forma de all_indices_test: {all_indices_test.shape} ")
print(f"esta es la forma de all_y_test: {all_y_test.shape} ")

# Imprimir detalles del primer elemento para verificar
if len(all_X_train) > 0:
    print("Detalles del primer elemento de las secuencias:")
    print("X (features):", all_X_train[0])
    print("X_sp500 (features):", all_X_sp500_train[0])
    print("y (labels):", all_y_train[0])
    print("Índice de la compañía del primer elemento:", all_indices_train[0])
else:
    print("No se han generado secuencias.")

print("Datos preparados para el modelo.")

### CONSTRUCCION DEL MODELO####



from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


# Configuraciones del modelo ajustadas
embedding_size = 16
lstm_units = 254  # Reducir unidades LSTM
dense_units = 128  # Reducir unidades Dense
dropout_rate = 0.3  # Ajustar tasa de dropout
learning_rate = 0.0005
l2_reg = 0.0005  # Reducir regularización L2

num_companies = len(company_mapping)

# Entradas del modelo
input_company = Input(shape=(n_steps-1,), name='company')
input_features = Input(shape=(n_steps-1, all_X_train.shape[2]), name='features')
input_sp500 = Input(shape=(n_steps-1, all_X_sp500_train.shape[2]), name='sp500')

# Capa de Embedding para las compañías
embedding = Embedding(input_dim=num_companies, output_dim=embedding_size)(input_company)

# Concatenar embeddings con las features
concatenated_inputs = Concatenate(axis=-1)([embedding, input_features, input_sp500])

lstm_out = LSTM(units=lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), recurrent_dropout=0.2)(concatenated_inputs)
lstm_out = Dropout(rate=dropout_rate)(lstm_out)
lstm_out = BatchNormalization()(lstm_out)
lstm_out = LSTM(units=lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), recurrent_dropout=0.2)(lstm_out)
lstm_out = Dropout(rate=dropout_rate)(lstm_out)
lstm_out = BatchNormalization()(lstm_out)
lstm_out = LSTM(units=lstm_units, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), recurrent_dropout=0.2)(lstm_out)
lstm_out = Dropout(rate=dropout_rate)(lstm_out)
lstm_out = BatchNormalization()(lstm_out)

dense_out = Dense(units=dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(lstm_out)
dense_out = Dropout(rate=dropout_rate)(dense_out)
dense_out = BatchNormalization()(dense_out)
# Capa de salida
output = Dense(units=5, activation='softmax')(dense_out)

# Definir el modelo
model = Model(inputs=[input_company, input_features, input_sp500], outputs=output)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=learning_rate), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Entrenamiento del modelo con validación cruzada
history = model.fit([all_indices_train, all_X_train, all_X_sp500_train], 
                    all_y_train, 
                    validation_data=([all_indices_test, all_X_test, all_X_sp500_test], all_y_test), 
                    epochs=50, 
                    batch_size=128,
                    callbacks=[early_stopping, reduce_lr])

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate([all_indices_test, all_X_test, all_X_sp500_test], all_y_test)
print(f"Pérdida en Test: {test_loss}")
print(f"Precisión en Test: {test_accuracy}")

import os

# Ruta de salida
output_dir = 'C:/Users/s0141677/OneDrive - THALES SA/Documents/TFG - ALEJANDRO ALONSO ANDA/DATOS MODELOS/MODELO15(DATOS ADICIONALES)'
os.makedirs(output_dir, exist_ok=True)

# Guardar el modelo
model.save(os.path.join(output_dir, 'my_model15.h5'))
# Indicar la ubicación del archivo generado
print(f"Modelo guardado: {os.path.join(output_dir, 'my_model15.h5')}")

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np



# Historial de entrenamiento
history_dict = history.history

# Gráficas de Pérdida y Precisión
plt.figure(figsize=(12, 4))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(history_dict['loss'], label='Pérdida de Entrenamiento')
plt.plot(history_dict['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida durante el Entrenamiento y la Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Precisión
plt.subplot(1, 2, 2)
plt.plot(history_dict['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history_dict['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión durante el Entrenamiento y la Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_plots.png'))
plt.show()

# Generar las predicciones del modelo
y_pred = model.predict([all_indices_test, all_X_test, all_X_sp500_test])
y_pred_classes = np.argmax(y_pred, axis=1)

# Matriz de Confusión
conf_matrix = confusion_matrix(all_y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Muy Alcista', 'Alcista', 'Muy Bajista', 'Bajista', 'Lateral'], yticklabels=['Muy Alcista', 'Alcista', 'Muy Bajista', 'Bajista', 'Lateral'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.show()

# Clasificación Report
class_report = classification_report(all_y_test, y_pred_classes, target_names=['Muy Alcista', 'Alcista', 'Muy Bajista', 'Bajista', 'Lateral'])

# Backtesting
df['signal'] = y_pred_classes  # y_pred_classes es el resultado de np.argmax(y_pred, axis=1)

initial_balance = 100000  # Balance inicial en dólares
balance = initial_balance
positions = 0  # Posiciones (número de acciones compradas)

for index, row in df.iterrows():
    if row['signal'] == 0 or row['signal'] == 1:  # Señal de compra (Muy Alcista o Alcista)
        positions = balance / row['Close']  # Comprar tantas acciones como sea posible
        balance = 0  # Todo el balance se usa para comprar
    elif row['signal'] == 2 or row['signal'] == 3:  # Señal de venta (Muy Bajista o Bajista)
        balance = positions * row['Close']  # Vender todas las acciones
        positions = 0  # No quedan posiciones

# Valor final de la cartera
final_balance = balance + positions * df.iloc[-1]['Close']
profit = final_balance - initial_balance

print(f"Balance inicial: ${initial_balance:.2f}")
print(f"Balance final: ${final_balance:.2f}")
print(f"Ganancia: ${profit:.2f}")

# Guardar los resultados del backtesting en el informe
with open(os.path.join(output_dir, 'model_training_report.txt'), 'a') as report_file:
    report_file.write("\n\nBacktesting:\n")
    report_file.write(f"Balance inicial: ${initial_balance:.2f}\n")
    report_file.write(f"Balance final: ${final_balance:.2f}\n")
    report_file.write(f"Ganancia: ${profit:.2f}\n")

# Crear el informe
with open(os.path.join(output_dir, 'model_training_report.txt'), 'w') as report_file:
    report_file.write("Informe de Entrenamiento del Modelo\n")
    report_file.write("===============================\n\n")
    report_file.write("Resumen del Modelo:\n")
    model.summary(print_fn=lambda x: report_file.write(x + '\n'))
    report_file.write("\n\n")

    report_file.write("Resultados del Entrenamiento:\n")
    report_file.write(f"- Pérdida de Entrenamiento Final: {history_dict['loss'][-1]:.4f}\n")
    report_file.write(f"- Pérdida de Validación Final: {history_dict['val_loss'][-1]:.4f}\n")
    report_file.write(f"- Precisión de Entrenamiento Final: {history_dict['accuracy'][-1]:.4f}\n")
    report_file.write(f"- Precisión de Validación Final: {history_dict['val_accuracy'][-1]:.4f}\n")
    report_file.write("\n\n")

    report_file.write("Evaluación en el Conjunto de Prueba:\n")
    report_file.write(f"- Pérdida en Test: {test_loss:.4f}\n")
    report_file.write(f"- Precisión en Test: {test_accuracy:.4f}\n")
    report_file.write("\n\n")

    report_file.write("Análisis de Convergencia:\n")
    report_file.write("Las gráficas de pérdida y precisión muestran la evolución del entrenamiento y la validación del modelo.\n")
    report_file.write("La pérdida de validación parece estabilizarse, indicando una posible convergencia del modelo.\n")
    report_file.write("Sin embargo, las fluctuaciones en la precisión de validación sugieren que el modelo podría beneficiarse de más ajustes o un mayor conjunto de datos.\n")
    report_file.write("\n\n")

    report_file.write("Matriz de Confusión:\n")
    report_file.write("La matriz de confusión muestra cómo se están clasificando las diferentes clases.\n")
    report_file.write(f"{conf_matrix}\n\n")

    report_file.write("Reporte de Clasificación:\n")
    report_file.write(f"{class_report}\n\n")

    report_file.write("Visualizaciones:\n")
    report_file.write("Las visualizaciones del entrenamiento y la validación del modelo se pueden encontrar en los archivos 'training_plots.png' y 'confusion_matrix.png'.\n")

    report_file.write("Backtesting:\n")
    report_file.write(f"Balance inicial: ${initial_balance:.2f}\n")
    report_file.write(f"Balance final: ${final_balance:.2f}\n")
    report_file.write(f"Ganancia: ${profit:.2f}\n")

# Indicar la ubicación de los archivos generados
print(f"Informe generado: {os.path.join(output_dir, 'model_training_report.txt')}")
print(f"Gráficas de entrenamiento: {os.path.join(output_dir, 'training_plots.png')}")
print(f"Matriz de confusión: {os.path.join(output_dir, 'confusion_matrix.png')}")
print(f"Modelo guardado: {os.path.join(output_dir, 'my_model.h5')}")