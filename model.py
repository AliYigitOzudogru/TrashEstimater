import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Veriyi yükle ve işlem yap
def load_and_preprocess_data(csv_path):
    """
    CSV dosyasını yükler, saat bazında gruplar ve veriyi ölçeklendirir.
    """
    data = pd.read_csv(csv_path)
    df = pd.DataFrame(data)
    
    # Kategorileri sayısal değerlere çevir
    kategori_map = {"Az": 1, "Orta": 2, "Cok": 3}
    df["Kategori_Sayisal"] = df["Kategori"].map(kategori_map)
    
    # Zamanı, UNIX timestamp formatından datetime'a çevir (saniye cinsinden)
    df["Zaman"] = pd.to_datetime(df["Zaman"], unit='s', errors="coerce")
    df["Saat"] = df["Zaman"].dt.hour
    
    # Saat bazında ortalama doluluk hesapla
    ortalama_doluluk = df.groupby("Saat")["Kategori_Sayisal"].mean()
    
    # Veriyi ölçeklendir (0-1 aralığına getir)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(ortalama_doluluk.values.reshape(-1, 1))
    
    return scaled_data, scaler, ortalama_doluluk.index

# LSTM model için dizileri oluştur
def create_sequences(data, look_back=3):
    """
    PyTorch modeli için giriş (X) ve çıkış (Y) dizilerini oluşturur.
    """
    X, Y = [], []
    
    if len(data) <= look_back:
        print("Veri uzunluğu 'look_back' değerinden küçük. Lütfen daha fazla veri kullanın.")
        return X, Y

    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
        Y.append(data[i + look_back, 0])

    X, Y = np.array(X), np.array(Y)
    
    if X.shape[0] == 0:
        print("Veri dizisi oluşturulamadı. Lütfen veri uzunluğunu kontrol edin.")
    return X.reshape(X.shape[0], X.shape[1], 1), Y

# LSTM modelini oluştur
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Son zaman adımındaki çıkışı al
        return out

# Modeli eğit
def train_model(model, X_train, Y_train, epochs=50, batch_size=1, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)  # Y'yi uygun boyuta getir
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X_train_tensor)
        
        # Loss hesapla
        loss = criterion(output, Y_train_tensor)
        
        # Backward pass ve optimizasyon
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model

# Bir sonraki saati tahmin et
def predict_next_hour(model, last_hours, scaler):
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(last_hours.reshape(1, len(last_hours), 1), dtype=torch.float32))
        return scaler.inverse_transform(prediction.numpy().reshape(-1, 1))[0][0]

# Ana çalışma alanı
if __name__ == "__main__":
    # Çalışma alanınızdaki data klasöründeki dosya yolunu kullanıyoruz.
    csv_path = "/home/ali/Desktop/time_series_analyze_with_trash/TrashEstimater/data/datason.csv"
    
    # Veriyi hazırla
    scaled_data, scaler, saat_index = load_and_preprocess_data(csv_path)
    
    # LSTM için giriş ve çıkış verilerini hazırla
    look_back = 3
    X, Y = create_sequences(scaled_data, look_back)
    
    # Eğer veri yoksa, hatayı önlemek için çık
    if len(X) == 0:
        print("Yetersiz veri, işlem sonlandırıldı.")
        exit()

    # Eğitim ve test verilerine ayır
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    
    # PyTorch modelini oluştur
    model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
    
    # Modeli eğit
    model = train_model(model, X_train, Y_train, epochs=50, batch_size=1)
    
    # Son 3 saatlik veriye göre tahmin yap
    predicted_value = predict_next_hour(model, scaled_data[-look_back:], scaler)
    print(f"Tahmin edilen doluluk seviyesi: {predicted_value}")
    
    # Saat indeksini NumPy dizisine dönüştür
    saat_index = np.array(saat_index)

    # Grafik çiz
    plt.plot(saat_index, scaler.inverse_transform(scaled_data), marker="o", linestyle="-", color="b", label="Gerçek")
    plt.scatter(saat_index[-1] + 1, predicted_value, color="r", label="Tahmin")
    plt.xlabel("Saat")
    plt.ylabel("Ortalama Doluluk")
    plt.title("Saat Bazında Doluluk Tahmini")
    plt.legend()
    plt.grid(True)
    plt.show()
