import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

dataset = pd.read_csv('dataset.csv') 

dane = pd.DataFrame(dataset)
print(dane.head(30))
print(dane['track_genre'][:20])
print(dane.loc[10:20])

##odchylenie standardowe

#odstand = dana.column
print(dane.columns)

tempo = dane['tempo']
print(tempo)
print('średnia',tempo.mean())
print('odchylenie standardowe',tempo.std())
print(dane['tempo'])

#sprawdzam czy energiczność utworu koreluje z tanecznością

print(dane[['energy','danceability']])
taneczność = pd.Series(dane['danceability'])
energiczność = pd.Series(dane['energy'])
korealcja = st.pearsonr(taneczność, energiczność)
#print(dane['track_genre'].unique())
print('korelacja wynosi:',korealcja)

#spodziewałam się, że korelacja będzie wysoka, ale nie :(
# Istotna statystycznie, ale o słabej sile

# badanie pojedynczych zmiennych

egzemplarze_kategorii = dane['track_genre'].value_counts()
kategorie = dane['track_genre'].unique()
plt.bar(dane['track_genre'].unique(), egzemplarze_kategorii)
plt.show()

# for i in range(len(kategorie)):
#     for j in range(len(egzemplarze_kategorii)):
#         print(kategorie[i],': ',egzemplarze_kategorii[j])
# print(egzemplarze_kategorii)
# print(kategorie)
# print(dane['track_genre'].mode())
print(dane['artists'].unique())
print('czesc')
