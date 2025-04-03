import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("./data/izmir_data.csv")
df = pd.DataFrame(data)
kategori_map = {"Az": 1, "Orta": 2, "Çok": 3}
df["Kategori_Sayisal"] = df["Kategori"].map(kategori_map)
# Saat bazında doluluk seviyelerini gruplama
df["Zaman"] = pd.to_datetime(df["Zaman"], unit="s")  # Unix timestamp'i gerçek tarihe çevir
df["Saat"] = df["Zaman"].dt.hour  # Saat bilgisini al

ortalama_doluluk = df.groupby("Saat")["Kategori_Sayisal"].mean()


"""
# Grafik çizme
plt.figure(figsize=(10, 5))
plt.plot(ortalama_doluluk.index, ortalama_doluluk.values, marker="o", linestyle="-", color="b")
plt.xlabel("Saat")
plt.ylabel("Ortalama Doluluk Seviyesi")
plt.title("Saat Bazlı Ortalama Doluluk Seviyesi")
plt.grid(True)
plt.show()
"""
#print(df["Zaman"].head(10))  # İlk 10 değeri yazdıralım
print(df["Saat"].unique())


import matplotlib.pyplot as plt

# NumPy array'lere dönüştürerek çizdir
plt.plot(ortalama_doluluk.index.to_numpy(), ortalama_doluluk.values, marker="o", linestyle="-", color="b")

# Grafik etiketleri
plt.xlabel("Saat")
plt.ylabel("Ortalama Doluluk")
plt.title("Saat Bazında Ortalama Doluluk")
plt.grid(True)

plt.show()
print(ortalama_doluluk)


