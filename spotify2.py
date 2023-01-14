import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

dataset = pd.read_csv('dataset.csv') 

dane = pd.DataFrame(dataset)

#sprawdzam czy energiczność utworu koreluje z tanecznością

print(dane[['energy','danceability']])
taneczność = pd.Series(dane['danceability'])
energiczność = pd.Series(dane['energy'])
korealcja = st.pearsonr(taneczność, energiczność)
print('korelacja wynosi:',korealcja)

#spodziewałam się, że korelacja będzie wysoka, ale nie :(
# Istotna statystycznie, ale o słabej sile

#analiza statystyczna pojedynczej zmiennej - loudness
print('Statystyki głośności:')
print('średnia arytmetyczna: ', dane['loudness'].mean())
print('odchylenie standardowe: ', dane['loudness'].std())
print('mediana: ', dane['loudness'].median())
print('dominanta: ', dane['loudness'].mode())

plt.hist(dane['loudness'], bins=20)
plt.show()

