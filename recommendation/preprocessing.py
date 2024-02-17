import pandas as pd
import numpy as np
import dateutil.parser as parser
import re
import nltk
import ssl
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from langdetect import detect
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import json


anime=pd.read_csv("/Users/jansetaskin/Desktop/data/anime_data/cleaned/animes.csv")
profiles=pd.read_csv("/Users/jansetaskin/Desktop/data/anime_data/cleaned/profiles.csv")
reviews=pd.read_csv("/Users/jansetaskin/Desktop/data/anime_data/cleaned/reviews.csv")


class Anime(object):
    '''Methods for preoprocessing the anime dataset
    Input: pandas dataframe of items and feautures'''

    def __init__(self, anime):
        self.anime = anime
    
    def clean_data(self):

        # sort by duplicate values
        vc = self.anime.uid.value_counts()
        self.anime[self.anime.uid.isin(vc.index[vc.gt(1)])].sort_values('uid')
        # remove duplicate values
        ids = self.anime['uid']
        self.anime[ids.isin(ids[ids.duplicated()])].sort_values('uid')

        self.anime = self.anime.drop_duplicates(subset=['uid'], keep='first')
        self.anime = self.anime[self.anime['synopsis'].notna()]
        self.anime.drop(['ranked', 'img_url', 'score', 'link', 'members'], axis=1, inplace=True)
        self.anime.loc[self.anime['aired'] != "Not available"]

        self.anime['synopsis'].replace('', np.nan, inplace=True)
        self.anime = self.anime[self.anime['synopsis'].notna()]
        return self.anime

    @staticmethod
    def date_parse(row):
        row = row.strip()
        if "to" in row:
            start_date_str, end_date_str = row.split(" to ")
            year_pattern = re.compile(r'\b\d{4}\b') # matches sequence of four digits - will detect year format
            matches = year_pattern.findall(start_date_str)
            for match in matches:
                years = int(match)
            return years
        else:
            return parser.parse(row).year
        
    def year_bins(self):
        labels = ['Before 1990', '1990-2000', '2000-2005', '2005-2010', '2010-2015', '2015-2021']
        bins = [0, 1990, 2000, 2005, 2010, 2015, 2021]

        self.anime['start_year_bins'] = pd.cut(self.anime['start_year'], bins=bins, labels=labels)
        return self.anime

    @staticmethod    
    def clean_synopsis_text(row):
        punctuation = re.compile("[.;:!\'?,\",()\[\]]")
        symbols = re.compile('[^0-9a-z #+_]]')
        symbols2 = re.compile('-|/|&|%|$|@|etc|--')
        digits = re.compile(r'[0-9]')
        result = str(row).lower()
        result = result.replace('\r', '')
        result = result.replace('\n', '')
        result = result.replace('  ', '')
        result = result.replace('[written by mal rewrite]', '')
        result = result.replace('[Written by MAL Rewrite]', '')
        result = punctuation.sub("", result)
        result = symbols.sub("", result)
        result = symbols2.sub("", result)
        result = digits.sub("", result)
        result = result.strip()
        
        return result

    @staticmethod
    def remove_non_english(row):
        tokens = row.split()
        cleaned = [token for token in tokens if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+", token)]
        return ' '.join(cleaned)

    def remove_stopwords(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('stopwords')
        stop = stopwords.words("english")
        self.anime['synopsis'] =self.anime['synopsis'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        return self.anime

    @staticmethod
    def detect_lang(row):
        language = detect(row)
        return language
    
    def keep_english(self):
        self.anime = self.anime[self.anime['synopsis_lang'] == 'en']
        self.anime.drop('synopsis_lang', axis=1, inplace=True)
        return self.anime

    @staticmethod
    def remove_non_english_characters(row):
        pattern = re.compile(r'[^\x00-\x7F]')
        return pattern.sub('', row)
    
    @staticmethod
    def stemmer(row):
        ps = PorterStemmer()
        tokens = row.split()
        stemmed_tokens = [ps.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    
    @staticmethod
    def convert_to_list(row):
        x = ast.literal_eval(row)
        return x
    
    def create_genre_dummies(self):
        genres = self.anime['genre']
        mlb = MultiLabelBinarizer()
        genre_dummies = pd.DataFrame(mlb.fit_transform(genres),columns=mlb.classes_, index=self.anime.index)
        self.anime = pd.concat([self.anime, genre_dummies], axis=1)
        return self.anime

    def create_year_dummies(self):
        year_dummies = pd.get_dummies(self.anime['start_year_bins'], dtype=int, drop_first=True)
        self.anime = pd.concat([self.anime, year_dummies], axis=1)
        return self.anime
    

class Profiles(object):
    '''Methods for preoprocessing the profiles dataset
    Input: pandas dataframe of items and feautures'''

    def __init__(self, profiles, anime, anime_uid_list):
        self.profiles = profiles
        self.anime = anime
        self.anime_uid_list = self.anime_uid_list

    def clean_data(self):
        prof_ids = self.profiles['profile']
        self.profiles[prof_ids.isin(prof_ids[prof_ids.duplicated()])].sort_values('profile')
        self.profiles = self.profiles.drop_duplicates(subset=['profile'], keep='first')
        self.profiles = self.profiles.loc[self.profiles['favorites_anime'] != '[]']
        self.profiles = self.profiles[['profile', 'favorites_anime']]
        return self.profiles

    @staticmethod
    def convert_list(row):
        x = ast.literal_eval(row)
        return x
    
    def filter_uids(self, row, list2):
        self.anime_uid_list = self.anime['uid'].tolist()
        self.anime_uid_list = [str(element) for element in self.anime_uid_list]

        new_list = [x for x in row if x in list2]
        return new_list
    
    def create_favorites_dummies(self):
        anime_names = self.anime[['uid', 'title']]
        anime_names['uid'] = anime_names['uid'].astype(str)
        favs = self.profiles['favorites']
        mlb = MultiLabelBinarizer()
        favs_dummies = pd.DataFrame(mlb.fit_transform(favs),columns=mlb.classes_, index=self.profiles.index)
        favs_dummies.columns = favs_dummies.columns.map({x: y for x, y in zip(anime_names['uid'], anime_names['title'])})
        self.profiles = pd.concat([self.profiles, favs_dummies], axis=1)

class Reviews(object):
    '''Methods for preoprocessing the reviews dataset
    Input: pandas dataframe of items and feautures'''

    def __init__(self, reviews, anime):
        self.reviews = reviews
        self.anime = anime
    
    def clean_data(self):
        rev_ids = self.reviews['uid']
        self.reviews[rev_ids.isin(rev_ids[rev_ids.duplicated()])].sort_values('uid')
        self.reviews = self.reviews.drop_duplicates(subset=['uid'], keep='first')
        self.reviews = self.reviews[['uid', 'profile', 'anime_uid', 'score', 'scores']]
        vc = self.reviews['profile'].value_counts()

        self.reviews = self.reviews[self.reviews['profile'].isin(vc.index[vc.gt(9)])]
        self.reviews['scores_dict'] = [i.replace("\'", "\"") for i in self.reviews['scores']]
        self.reviews['scores_dict2'] = self.reviews['scores_dict'].apply(json.loads)
        self.reviews['story'] = [i['Story'] for i in self.reviews['scores_dict2']]
        self.reviews['animation'] = [i['Animation'] for i in self.reviews['scores_dict2']]
        self.reviews['sound'] = [i['Sound'] for i in self.reviews['scores_dict2']]
        self.reviews['character'] = [i['Character'] for i in self.reviews['scores_dict2']]
        self.reviews['enjoyment'] = [i['Enjoyment'] for i in self.reviews['scores_dict2']]

        self.reviews['anime_uid'] = self.reviews['anime_uid'].astype(str)
        self.reviews['uid'] = self.reviews['uid'].astype(str)

        anime_names = self.anime[['uid', 'title']]
        anime_names['uid'] = anime_names['uid'].astype(str)
        self.reviews = self.reviews.merge(anime_names, left_on="anime_uid", right_on="uid", how="left")
        self.reviews = self.reviews[self.reviews['score'] != 0]
        self.reviews = self.reviews[self.reviews['score'] != 11]
        self.reviews = self.reviews.rename(columns={'uid_x': 'uid'})
        self.reviews = self.reviews[['uid', 'profile', 'anime_uid', 'score', 'story', 'animation', 'sound', 'character', 'enjoyment', 'title']]
        self.reviews = self.reviews[self.reviews['anime_uid'].isin(anime_names['uid'])]

# Apply preprocessing to anime dataset
Anime.clean_data(anime)
anime['start_year'] = anime['aired'].apply(Anime.date_parse)
Anime.year_bins(anime)
anime['synopsis'] = anime['synopsis'].apply(Anime.clean_synopsis_text)
anime['synopsis'] = anime['synopsis'].apply(Anime.remove_non_english)
Anime.remove_stopwords(anime)
anime['synopsis_lang'] = anime['synopsis'].apply(Anime.detect_lang)
Anime.keep_english(anime)
anime['synopsis'] = anime['synopsis'].apply(Anime.remove_non_english_characters)
anime['synopsis4'] = anime['synopsis3'].apply(Anime.stemmer)
anime['genre'] = anime['genre'].apply(Anime.convert_to_list)
Anime.create_genre_dummies(anime)
Anime.create_year_dummies(anime)

# Apply preprocessing to profiles dataset
profiles['favorites_anime'] = profiles['favorites_anime'].apply(Profiles.convert_list)
profiles['favorites'] = profiles.apply(lambda row: Profiles.filter_uids(row['favorites_anime'], Profiles.anime_uid_list), axis=1)

# apply preprocessing to reviews dataset
reviews = Reviews.clean_data(reviews)

anime.to_csv('/Users/jansetaskin/Desktop/data/anime_data/cleaned/testanimes.csv')
reviews.to_csv('/Users/jansetaskin/Desktop/data/anime_data/cleaned/testreviews.csv')
profiles.to_csv('/Users/jansetaskin/Desktop/data/anime_data/cleaned/testprofiles.csv')