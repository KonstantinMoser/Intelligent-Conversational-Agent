from surprise import SVD
from surprise import Dataset, Reader
import re
# from surprise import accuracy
# from surprise.model_selection.split import train_test_split

import pandas as pd
from os import path


class Recommender():
    """Agent that handles movie recommendation requests."""
    
    def __init__(self, ratings_path, movies_path):
        self.ratings_df, self.movies_df = self._load_data(ratings_path, movies_path)
        self.ratings_df_extd = None
        self.user_ratings_df = None
        self.new_uid = None
        self.model = None


        print("Recommender: Fixing movie titles... ", end="")        
        self._preprocess_movie_data()
        print("Done.")        
        
    def _load_data(self, ratings_path, movies_path):
        ratings_df = pd.read_csv(
            path.normpath(ratings_path), 
            dtype={'userId':str, 'movieId':str},
            usecols=['userId', 'movieId', 'rating']
        )
        # movies_df = pd.read_csv(path.normpath('../data/ml-25m/movies.csv'))
        movies_df = pd.read_csv(path.normpath(movies_path), dtype={'movieId':str})
        
        return ratings_df, movies_df
    
    
    def _rearrange_movie_title(self, title):
        # Use regular expression to extract movie name and year
        match = re.match(r'^(.*?)(,\s*The)?(,\s*An?)?\s*\((\d{4})\)$', title)
        
        if match:
            # Extract components from the match object
            movie_name = match.group(1)
            the_prefix = match.group(2)
            a_prefix = match.group(3)
            year = int(match.group(4))

            # Rearrange the title
            if the_prefix or a_prefix:
                prefix = the_prefix if the_prefix else a_prefix
                prefix = prefix.strip(', ')
                rearranged_title = f"{prefix} {movie_name}"
            else:
                rearranged_title = f"{movie_name}"
            return rearranged_title, year
        else:
            # Return the original title if no match is found
            return title, None
    
    
    def _preprocess_movie_data(self):
        # Move article from end of title to start, extract year into a new column
        # arranged_titles_years = self.movies_df['title']\
        #     .apply(lambda title: self._rearrange_movie_title(title))
        self.movies_df[['title_short', 'year']] = self.movies_df['title']\
            .apply(lambda title: pd.Series(self._rearrange_movie_title(title)))
            
        # self.movies_df['year'] = self.movies_df['year'].astype(int, errors='ignore')
        self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce').astype('Int64')
    
    
    def _add_new_user_and_ratings(self, user_ratings:list[tuple[str, float]], debug=False):
        """Add a new user with their ratings.

        Args:
            user_ratings (list[tuple[str, float]]): list of tuples containing movie name and rating
                                                    e.g. [
                                                            ('Pulp Fiction', 5.0),
                                                            ('Jumanji: Welcome to the Jungle', 4.0),
                                                            ('Deadpool 2', 5.0)
                                                        ]

        Returns:
            pd.Dataframe: self.df_ratings extended with new user ratings
        """
        self.new_uid = str(self.ratings_df.loc[:, 'userId'].astype(int).max() + 1)
        
        # Insert new_uid into each tuple of the list
        user_ratings = [(self.new_uid, ) + t for t in user_ratings]

        self.user_ratings_df = pd.DataFrame(user_ratings, columns=self.ratings_df.columns)
        self.user_ratings_df = self.user_ratings_df.rename(columns={'movieId': 'title_short'})
        self.ratings_df_extd = self.ratings_df

        # Convert movie titles (column movieId) to IDs and add rows to ratings DF
        
        self.user_ratings_df = pd.merge(self.user_ratings_df, self.movies_df[['movieId', 'title_short']], 
                                        on='title_short', how='left')
        self.user_ratings_df = self.user_ratings_df.dropna(how='any')
        if debug:
            print("Matched movies in ratings database: ", self.user_ratings_df['title_short'].tolist())
        self.user_ratings_df = self.user_ratings_df.drop('title_short', axis=1)
        

        if self.user_ratings_df.empty:
            print("No movies with that title found in ratings dataset")
            return False

        # Add new ratings to extended rating DF    
        self.ratings_df_extd = pd.concat([self.ratings_df_extd, self.user_ratings_df], ignore_index=True)
        
        if debug:
            print("New user recommendations added.")
        return True
    
    
    def _fit_recommender(self):
        """Build training set and fit the recommender model

        Returns:
            surprise.SVD: Recommender model
        """
        # A reader is still needed but only the rating_scale param is required.
        reader = Reader(rating_scale=(1, 5))

        # The columns must correspond to user id, item id and ratings (in that order).
        data = Dataset.load_from_df(self.ratings_df_extd[["userId","movieId","rating"]], reader)

        # Create a train-test split
        # trainset, testset = train_test_split(data, test_size=0.1, random_state=42)
        trainset = data.build_full_trainset()

        # Train the model
        self.model = SVD()
        # model = KNNWithMeans()
        self.model.fit(trainset)
        print("Recommender model fit.")
    
    
    def get_recommendations_for_user(self, user_ratings:list[tuple[str, float]], n:int=6, debug:bool=False):
        """Main interface for the class. Accepts a new user's ratings to be added to the ratings database
        and to train the recommender model. Returns a Pandas Dataframe of predicted ratings, movieIDs and titles

        Args:
            user_ratings (list[tuple[str, float]]): list of tuples containing movie name and rating
                                                    e.g. [
                                                            (new_uid, 'Pulp Fiction (1994)', 5.0),
                                                            (new_uid, 'Jumanji: Welcome to the Jungle (2017)', 4.0),
                                                            (new_uid, 'Deadpool 2 (2018)', 5.0)
                                                        ]
        """
        # if not self.ratings_df_new_extd:
        #     raise Exception("Error: You have to add a new user and their recommendations with " + \
        #                     "Recommender.add_new_user_and_ratings() first.")
        if not self._add_new_user_and_ratings(user_ratings, debug=debug):
            return []
        self._fit_recommender()
        
        # Compute predicted ratings of the new user for all movies
        unrated_movies_ids = self.ratings_df['movieId'].drop_duplicates()
        unrated_movies_ids = unrated_movies_ids[~unrated_movies_ids.isin(self.user_ratings_df['movieId'])]
        predictions_s = unrated_movies_ids.apply(lambda id: self.model.predict(self.new_uid, id, None).est)
        predicted_ratings = pd.DataFrame({'movieId': unrated_movies_ids, 'prediction':predictions_s})
        predicted_ratings.sort_values('prediction', ascending=False, inplace=True)

        # Add movie titles and years to the predictions
        preds_and_info = pd.merge(predicted_ratings, self.movies_df[['movieId', 'title_short', 'year', 'genres']], on='movieId', how='left')

        ### Year-based recommendation filtering
        user_movies_info = self.movies_df.loc[self.movies_df['movieId'].isin(self.user_ratings_df['movieId'])]
        boundary_yrs = 8
        # rec_year_range = range(user_movies_info['year'].min() - boundary_yrs, user_movies_info['year'].max() + 1 + boundary_yrs)
        rec_year_range = range(user_movies_info['year'].mean().astype(int) - boundary_yrs, 
                               user_movies_info['year'].mean().astype(int) + boundary_yrs)
        top_preds_inyears = preds_and_info[preds_and_info['year'].isin(rec_year_range)]
        
        ### Genre-based recommendation filtering
        # Genre filtering option 1 (more generic, filters by any genre included in liked movies)
        user_genres_set = user_movies_info['genres'].tolist()
        user_genres_set = set('|'.join(user_genres_set).split('|')) - {'IMAX'} # Remove useless/too narrowing genres
        top_preds_inyears_ingenres_broad = top_preds_inyears['genres'].apply(lambda x: any(genre in x for genre in user_genres_set))
        
        # Genre filtering option 2, halfway through 1 and 3: Take difference of genres of max (nr of matched genres)/3
        is_genres_matching = top_preds_inyears['genres']\
            .apply(lambda gs: len(set(gs.split('|')).symmetric_difference(user_genres_set)) <= max(1, len(user_genres_set)//3))
            
        top_preds_inyears_ingenres_narrow = top_preds_inyears[is_genres_matching]
        
        # Genre filtering option 3 (much more specific, filters by exact genre combos of liked movies)
        # top_preds_inyears_ingenres_narrow = top_preds_inyears[top_preds_inyears['genres'].isin(user_movies_info['genres'])]
        
        # choose top N
        if top_preds_inyears_ingenres_narrow.shape[0] > 0:
            final_recs = top_preds_inyears_ingenres_narrow.head(n) 
        else:
            final_recs = top_preds_inyears_ingenres_broad.head(n)
        
        ### Provide recommendations to the new user
        if debug:
            print("Genres combinations of liked movies: ", user_genres_set)
            print(f"Release year range to filter recommendations: {min(rec_year_range)} - {max(rec_year_range)}")
            # print(f"top {n} predictions:")
            # print(final_recs, '\n')
            # print("Count of top_preds_inyears_ingenres_narrow: ", top_preds_inyears_ingenres_narrow.shape[0])
            # print("Count of top_preds_inyears_ingenres_broad: ", top_preds_inyears_ingenres_broad.shape[0])
            
        final_recs_formatted = [f"{row['title_short']} ({row['year']}; {', '.join(row['genres'].split('|')[:3])})" 
                                for idx, row in final_recs.iterrows()]
        return final_recs_formatted
        