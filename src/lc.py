import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gensim.parsing import remove_stopwords
from gensim.utils import tokenize
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from src.custom_stop_words import csw

class LyricsCharter(object):
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

    def top_n_words_heatmap(self, n=10, ngrams=(1, 1)):
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
        trans["_year"] = df["decade"].fillna(0)
        trans["_year"] = pd.to_numeric(trans["_year"].fillna(0), downcast="integer")

        words_by_year = trans.groupby("_year").sum()

        top10 = {y: z.nlargest(n) for y, z in words_by_year.iterrows()}

        top10_df = pd.concat(top10, axis=1, sort=False)

        top10_df["sum"] = top10_df.sum(axis=1)
        top10_df.sort_values(by="sum", ascending=False, inplace=True)
        top10_df.drop("sum", axis=1, inplace=True)

        fig = plt.figure(figsize=(15, 10))
        sns.heatmap(top10_df.T, square=True, cmap="YlGnBu", cbar_kws={"orientation":"horizontal"})
