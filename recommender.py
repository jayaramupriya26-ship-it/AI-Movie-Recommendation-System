from movie_data import movies
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD

from movie_data import ratings, movies


class MovieRecommender:

    def __init__(self):

        self.movies = movies
        self.ratings = ratings
        self.model = None

        self.train_model()

    def train_model(self):

        df = pd.DataFrame(
            self.ratings,
            columns=["user_id", "movie", "rating"]
        )

        reader = Reader(rating_scale=(1, 5))

        data = Dataset.load_from_df(
            df[["user_id", "movie", "rating"]],
            reader
        )

        trainset = data.build_full_trainset()

        self.model = SVD(
            n_factors=50,
            n_epochs=30,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42
        )

        self.model.fit(trainset)

    def filter_movies(
        self,
        genre=None,
        language=None,
        min_imdb=0
    ):

        filtered = []

        for movie in self.movies:

            genre_match = True
            language_match = True
            imdb_match = True

            if genre:
                genre_match = genre in movie["genre"]

            if language:
                language_match = movie["language"] == language

            if min_imdb:
                imdb_match = movie["imdb"] >= min_imdb

            if genre_match and language_match and imdb_match:
                filtered.append(movie)

        return filtered

    def recommend(
        self,
        user_id,
        genre=None,
        language=None,
        min_imdb=0,
        top_n=5
    ):

        filtered_movies = self.filter_movies(
            genre,
            language,
            min_imdb
        )

        recommendations = []

        for movie in filtered_movies:

            prediction = self.model.predict(
                uid=user_id,
                iid=movie["title"]
            )

            recommendations.append({

                "title": movie["title"],

                "genre": ", ".join(movie["genre"]),

                "language": movie["language"],

                "year": movie["year"],

                "duration": movie["duration"],

                "imdb": movie["imdb"],

                "predicted_rating": round(
                    prediction.est,
                    2
                )

            })

        recommendations.sort(

            key=lambda x: x["predicted_rating"],

            reverse=True

        )

        return recommendations[:top_n]


recommender = MovieRecommender()


if __name__ == "__main__":

    user_id = 1

    genre = "Sci-Fi"

    language = "Hollywood"

    results = recommender.recommend(

        user_id=user_id,

        genre=genre,

        language=language,

        min_imdb=8,

        top_n=5

    )

    print()

    print("Movie Recommendations")

    print()

    for index, movie in enumerate(results, start=1):

        print(f"{index}. {movie['title']}")

        print(f"   Genre      : {movie['genre']}")

        print(f"   Language   : {movie['language']}")

        print(f"   IMDb       : {movie['imdb']}")

        print(f"   Year       : {movie['year']}")

        print(f"   Duration   : {movie['duration']} mins")

        print(f"   AI Score   : {movie['predicted_rating']}")

        print()