from src.lc import LyricsDownloader

n_minaj_downloader = LyricsDownloader(
    artist_url="http://www.metrolyrics.com/nicki-minaj-albums-list.html",
    artist_name="nicki-minaj")

n_minaj_downloader.songs_to_json("data/jsons/{name}.json".format(name="nicki-minaj"))
