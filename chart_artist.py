from src.lc import LyricsCharter
import matplotlib.pyplot as plt

nm = LyricsCharter("./data/jsons/nicki-minaj.json","Nicki Minaj")

print(nm.data.head())

print(nm.data.lyrics[0])

nm.top_n_words_heatmap(by_decade=False)
plt.show()