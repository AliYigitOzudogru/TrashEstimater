import pandas as pd
import numpy as np


data = pd.read_csv("./data/data.csv")
df = pd.DataFrame(data)

print(df)
import folium
from folium.plugins import HeatMap

# Haritanın merkezini hesaplama (ortalama enlem ve boylam)
center_lat, center_lon = df["Enlem"].mean(), df["Boylam"].mean()

# Folium haritası oluşturma
m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

# Isı haritası ekleme
heat_data = list(zip(df["Enlem"], df["Boylam"], df["Doluluk Seviyesi"]))
HeatMap(heat_data).add_to(m)

# Haritayı kaydetme
heatmap_path = "/mnt/data/heatmap.html"
m.save(heatmap_path)
