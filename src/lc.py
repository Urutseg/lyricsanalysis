import re

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from gensim.parsing import remove_stopwords
from gensim.utils import tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from src.custom_stop_words import csw


class LyricsCharter(object):
    """Class that hadles lyrics analysis for previously downloaded file"""
    def __init__(self, lyrics_file, artist_name):
        df = pd.read_json(lyrics_file)
        df.rename(columns={"name": "album"}, inplace=True)
        df["year"] = pd.to_numeric(df["year"], downcast="integer")
        df = df.dropna()
        df["decade"] = df["year"] - df["year"] % 10
        df["decade"] = df["decade"].astype("int")
        df.drop("URL", axis=1, inplace=True)
        df.sort_values(["year", "song"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop_duplicates(subset="lyrics", keep="first", inplace=True)
        df["song"] = df["song"].str.replace(
            r"(^{0} - )|( Lyrics$)".format(artist_name.title()), "")
        df["lyrics"] = df["lyrics"].str.replace("won't", "will not")
        df["lyrics"] = df["lyrics"].str.replace("can't", "can not")
        df["lyrics"] = df["lyrics"].str.replace("n't", " not")
        df["lyrics"] = df["lyrics"].str.replace("'m", " am")
        df["lyrics"] = df["lyrics"].str.replace("'re", " are")
        df["lyrics"] = df["lyrics"].str.replace("'ll", " will")
        df["lyrics"] = df["lyrics"].str.replace("'s", " is")
        df["lyrics"] = df["lyrics"].str.replace("'ve", " have")
        df["lyrics"] = df["lyrics"].str.replace(r"(\w+\s)\1+", r"\1")
        df["lyrics"] = df["lyrics"].str.replace(r"(\s\w+)\1+", r"\1")
        self.data = df
        self.data["word_count"] = self.data.lyrics.apply(
            lambda x: len(list(tokenize(remove_stopwords(x), lower=True))))
        self.data["unique_words"] = self.data.lyrics.apply(
            lambda x: len(set(list(tokenize(remove_stopwords(x), lower=True)))))

    def songs_by_decade(self):
        songs_by_decade = self.data.groupby("decade")[["song"]].count()
        songs_by_decade.reset_index(inplace=True)
        sns.barplot(data=songs_by_decade, x="decade", y="song")

    def word_counts_density(self):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots(1, 2)
        sns.distplot(self.data.word_count, ax=ax[0], kde=False)
        sns.distplot(self.data.unique_words, ax=ax[1], kde=False)
        # return ax

    def word_counts_trend(self):
        gb = self.data.groupby("year")["word_count", "unique_words"].mean()
        gb["lex_density"] = gb["unique_words"] / gb["word_count"]
        g = gb.unstack().reset_index()
        g.columns = ["cat", "year", "count"]
        ax = sns.lmplot(data=g, x="year", y="count",
                        hue="cat", col="cat", sharey=False)
        axes = ax.axes
        axes[0, 2].set_ylim(0, 1)
        # return ax

    def word_counts_vs_unique(self):
        sns.jointplot(data=self.data, x="unique_words", y="word_count")
        sns.jointplot(data=self.data, x="unique_words",
                    y="word_count", kind="hex")
        sns.jointplot(data=self.data, x="unique_words",
                    y="word_count", kind="kde")

    def words_by_year(self):
        words_by_year = self.data[["decade", "lyrics"]]
        words_by_year.columns = ["year", "lyrics"]

        wc_by_year = []
        for year, l in words_by_year.iterrows():
            l_ser = pd.Series(
                list(tokenize(remove_stopwords(l["lyrics"]), lower=True)))
            idf = pd.DataFrame(l_ser.groupby(l_ser).count(), columns=["count"])
            idf["decade"] = l["year"]
            wc_by_year.append(idf)

        wc_by_year_df = pd.DataFrame(columns=["count"],
                                    index=pd.MultiIndex(names=[None, "decade"], levels=[["and"], [1984]], labels=[[0], [0]]))

        for el in wc_by_year:
            wc_by_year_df = wc_by_year_df.add(
                el.set_index("decade", append=True), fill_value=0)

        wc_by_year_df.index = wc_by_year_df.index.swaplevel(0, 1)
        wc_by_year_df.dropna(inplace=True)
        wc_by_year_df["count"] = wc_by_year_df["count"].astype("int")
        wc_by_year_df = wc_by_year_df.reset_index()
        wc_by_year_df.columns = ["decade", "word", "count"]

        top10_by_year = wc_by_year_df.groupby(["decade"])
        plot_dfs = [{"decade": y, "df": g.set_index("word").drop("decade", axis=1).nlargest(
            10, "count").sort_values("count")} for y, g in top10_by_year]
        return plot_dfs

    def top_n_words_heatmap(self, n=10, ngrams=(1, 1), by_decade=True):
        df = self.data.copy()
        df.reset_index(drop=True, inplace=True)
        sw = ENGLISH_STOP_WORDS.union(csw)
        c = CountVectorizer(stop_words=sw, ngram_range=ngrams)

        m = c.fit_transform(df.lyrics)

        rev = {it: ind for ind, it in c.vocabulary_.items()}

        words = []
        for oi, m0 in enumerate(m):
            words.append({oi: {rev[ind]: it for ind, it in enumerate(m0.toarray().reshape(-1)) if it > 0}})

        words_df = [pd.DataFrame(word) for word in words]

        total_words = pd.concat(words_df, axis=1, sort=False)

        trans = total_words.T
        if by_decade:
            trans["_year"] = df["decade"].fillna(0)
        else:
            trans["_year"] = df["year"].fillna(0)

        trans["_year"] = pd.to_numeric(trans["_year"].fillna(0), downcast="integer")

        words_by_year = trans.groupby("_year").sum()

        top10 = {y: z.nlargest(n) for y, z in words_by_year.iterrows()}

        top10_df = pd.concat(top10, axis=1, sort=False)

        top10_df["sum"] = top10_df.sum(axis=1)
        top10_df.sort_values(by="sum", ascending=False, inplace=True)
        top10_df.drop("sum", axis=1, inplace=True)

        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(top10_df.T, square=True, cmap="YlGnBu", cbar_kws={"orientation":"horizontal"})


class LyricsDownloader(object):
    """Class that downloads and saves artist's lyrics from metrolyrics.com to the file"""
    def __init__(self, artist_url, artist_name):
        self.artist_url = artist_url
        self.artist_name = artist_name

    def process_artist(self):
        lst = []
        print("Processing {0}".format(self.artist_url))
        a, n = self.process_artist_page(self.artist_url)
        while True:
            print("Processing {n}".format(n=n))
            l = [self.parse_album(album) for album in a]
            lst.extend(l)
            a, n = self.process_artist_page(n)
            if not n:
                break
        all_songs = pd.concat([pd.DataFrame(l) for l in lst])
        return all_songs

    def process_artist_page(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        albums = soup.findAll("div", attrs={"class": re.compile(r"album-track-list module clearfix.*")})
        pagination = soup.find("p", "pagination")
        next_page_url = pagination.find("a", "button next")
        if next_page_url:
            next_page_url = next_page_url["href"]
        return albums, next_page_url

    def parse_album(self, album_soup):
        album = {}
        li = album_soup.findAll("li")
        album_name_year = album_soup.find("div", "grid_6 omega").header.h3.contents
        if len(album_name_year) == 2:
            album_name = album_name_year[0].contents[0]
            year = album_name_year[1].strip()
        else:
            album_name = album_name_year[0].contents[0]
            year = None
        lyrics_urls = [l.a["href"] for l in li]
        album["name"] = album_name
        album["year"] = year
        album["song_urls"] = lyrics_urls
        return album

    def download_lyrics(self, all_songs):
        song_urls = all_songs.song_urls.unique()
        url_df = pd.DataFrame(song_urls, columns=["URL"])
        get_song_text = self.get_song_text
        a = url_df["URL"].apply(self.get_song_text)
        df_a = pd.DataFrame(a.values.tolist(), columns=["song","lyrics"])
        songs_df = pd.concat([url_df, df_a], axis=1)
        return songs_df        

    def get_song_text(self, url):
        print("Getting lyrics for {url}".format(url=url))
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        h = soup.find("div", "banner-heading")
        song_name = h.h1.get_text()
        t = soup.findAll("p", "verse")
        text = "\n".join([i.getText() for i in t])
        return song_name, text

    def combine_artist_songs(self, all_songs, songs_df):
        artist_songs = all_songs.merge(songs_df, left_on="song_urls", right_on="URL")
        artist_songs.drop("song_urls", inplace=True, axis=1)
        artist_songs.drop_duplicates(inplace=True)
        artist_songs["year"] = artist_songs["year"].str.strip("()")
        return artist_songs        

    def songs_to_json(self, output_json):
        artist_df = self.process_artist()
        songs_df = self.download_lyrics(artist_df)
        combined_df = self.combine_artist_songs(artist_df, songs_df)
        combined_df.to_json(output_json)
