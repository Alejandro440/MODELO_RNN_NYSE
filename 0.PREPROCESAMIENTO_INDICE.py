import os

import numpy as np
import pandas as pd

from pipeline_config import INDEX_PROCESSED_DIR, RAW_INDICES_DIR, ensure_data_directories

def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    return data

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(prices, days):
    return prices.ewm(span=days, adjust=False).mean()

def calculate_macd(data):
    ema12 = calculate_ema(data['Close'], 12)
    ema26 = calculate_ema(data['Close'], 26)
    macd = ema12 - ema26
    signal_line = calculate_ema(macd, 9)
    return macd, signal_line

def calculate_rsi(data, window=14):
    change = data['Close'].diff()
    gain = (change.where(change > 0, 0)).rolling(window=window).mean()
    loss = (-change.where(change < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(data):
    data['SMA_30'] = calculate_sma(data['Close'], 30)
    data['SMA_50'] = calculate_sma(data['Close'], 50)
    data['EMA_30'] = calculate_ema(data['Close'], 30)
    data['EMA_50'] = calculate_ema(data['Close'], 50)
    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    data['RSI'] = calculate_rsi(data)
    return data

def process_directory(directory_path, output_directory):
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory_path, filename)
            data = load_and_clean_data(filepath)
            data = add_technical_indicators(data)
            data.dropna(inplace=True)  # Eliminar filas con valores NaN
            output_filename = filename
            output_filepath = os.path.join(output_directory, output_filename)
            data.to_csv(output_filepath, index=False)
            print(f'Processed and saved as {output_filename}')
def main():
    ensure_data_directories()
    process_directory(RAW_INDICES_DIR, INDEX_PROCESSED_DIR)


if __name__ == "__main__":
    main()
