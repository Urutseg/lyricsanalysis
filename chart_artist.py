import matplotlib.pyplot as plt

from src.lc import LyricsCharter

nm = LyricsCharter("./data/jsons/nicki-minaj.json", "Nicki Minaj")

print(nm.data.head())

print(nm.data.lyrics[0])

nm.top_n_words_heatmap(by_decade=False, ngrams=(2, 2))
plt.show()
