import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

dataset = pd.read_csv('dataset.csv') 

dane = pd.DataFrame(dataset)

# analiza zmiennej: track_genre
dane['track_genre'] = dane['track_genre'].astype('category')
print('kategorie: ', dane['track_genre'].unique())
print('Liczba kategorii: ', dane['track_genre'].count())
print('dominanta: ', dane['track_genre'].mode()) #tyle samo egzemplarzy w każdej kategorii

#analiza zmiennej: artist
dane['artists'] = dane['artists'].astype('category')
print('liczba wartości zmiennej: ', dane['artists'].count())
print("tyle razy w zestawieniu pojawia się artysta: \n", dane['artists'].value_counts())
print('dominanta: ', dane['artists'].mode())

#analiza zmiennej: explicit
dane['explicit'] = dane['explicit'].astype('category')
print('Liczba wartości zmiennej: ', dane['explicit'].count())
print('Liczba wyników dla każdej kategorii: ',dane['explicit'].value_counts())
print('dominanta: ', dane['explicit'].mode())

plt.pie([104253,9747], labels = ['wyraźny tekst', 'niewyraźny tekst'], autopct = '%1.1f%%')
plt.title('Czy teksty utworów są wyraźne?')
plt.show()

#analiza statystyczna zmiennej - loudness
print('Statystyki głośności:')
print(dane['loudness'].describe())
print('wariancja: ', dane['loudness'].var())
print('mediana: ', dane['loudness'].median())
print('dominanta: ', dane['loudness'].mode())

plt.hist(dane['loudness'], bins=20)
plt.xlabel('głośność [dB]')
plt.ylabel('liczba utworów')
plt.title('Głośność utworów')
plt.show()

#analiza zmiennej: valence
print('Statystyki pozytywności utworów:')
print(dane['valence'].describe())
print('wariancja: ', dane['valence'].var())
print('mediana: ', dane['valence'].median())
print('dominanta: ', dane['valence'].mode())

#analiza zmiennej: danceability
print('Statystyki taneczności utworów:')
print(dane['danceability'].describe())
print('wariancja: ', dane['danceability'].var())
print('mediana: ', dane['danceability'].median())
print('dominanta: ', dane['danceability'].mode())

#analiza zmiennej: energy
print('Statystyki energiczności utworów:')
print(dane['energy'].describe())
print('wariancja: ', dane['energy'].var())
print('mediana: ', dane['energy'].median())
print('dominanta: ', dane['energy'].mode())

#czy energiczność utworu koreluje z tanecznością
print(dane[['energy','danceability']])
taneczność = pd.Series(dane['danceability'])
energiczność = pd.Series(dane['energy'])
korealcja = st.pearsonr(taneczność, energiczność)
print('korelacja wynosi:\n',korealcja)

plt.scatter(taneczność, energiczność)
plt.xlim([0,0.2])
plt.ylim([0,0.2])
plt.xlabel('taneczność')
plt.ylabel('energiczność')
plt.title('Związek między tanecznością a energicznością utworów')
plt.show()

# Istotna statystycznie, o słabej sile

# korelacja: loudness - energy

loudness = pd.Series(dane['loudness'])
energy = pd.Series(dane['energy'])
kor = st.pearsonr(loudness, energy)
print(kor)

#istotny statystycznie silny związek między zmiennymi

#korelacja: danceability - valence

danceability = pd.Series(dane['danceability'])
valence = pd.Series(dane['valence'])
korr = st.pearsonr(danceability, valence)
print(korr)

#istotny statystycznie umiarkowany związek między zmiennymi

#korelacja loudness - valence
korelacja = st.pearsonr(loudness, valence)
print(korelacja)

#istotny statystycznie dodatni związek o słabej sile

