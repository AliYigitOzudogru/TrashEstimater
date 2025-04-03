import folium
from folium.plugins import HeatMap

import pandas as pd

data = pd.read_csv("./data/izmir_data.csv")

df = pd.DataFrame(data)
# Haritanın merkezini hesaplama (ortalama enlem ve boylam)

df["Enlem"] = pd.to_numeric(df["Enlem"], errors="coerce")
df["Boylam"] = pd.to_numeric(df["Boylam"], errors="coerce")

df = df.dropna(subset=["Enlem", "Boylam"])
heat_data = df[["Enlem", "Boylam"]].values.tolist()


center_lat, center_lon = df["Enlem"].mean(), df["Boylam"].mean()


# Folium haritası oluşturma
m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

# Isı haritası ekleme
for index, row in df.iterrows():
    folium.Marker(
        location=[row["Enlem"], row["Boylam"]],
        popup=f"Doluluk: {row['Kategori']}",
        icon=folium.Icon(color="blue" if row["Kategori"] == "az" else "orange" if row["Kategori"] == "orta" else "red")
    ).add_to(m)

HeatMap(heat_data).add_to(m)

# Haritayı kaydetme
heatmap_path = "./heatmap.html"
m.save(heatmap_path)