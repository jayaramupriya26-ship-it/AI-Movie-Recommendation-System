from flask import Flask, render_template, request
from recommender import recommender

app = Flask(__name__)


@app.route("/")
def home():

    genres = sorted(
        {
            genre
            for movie in recommender.movies
            for genre in movie["genre"]
        }
    )

    languages = sorted(
        {
            movie["language"]
            for movie in recommender.movies
        }
    )

    return render_template(
        "index.html",
        genres=genres,
        languages=languages,
        recommendations=None
    )


@app.route("/recommend", methods=["POST"])
def recommend():

    user_id = int(request.form["user_id"])

    genre = request.form["genre"]

    language = request.form["language"]

    min_imdb = float(request.form["imdb"])

    recommendations = recommender.recommend(

        user_id=user_id,

        genre=genre,

        language=language,

        min_imdb=min_imdb,

        top_n=5

    )

    genres = sorted(
        {
            genre
            for movie in recommender.movies
            for genre in movie["genre"]
        }
    )

    languages = sorted(
        {
            movie["language"]
            for movie in recommender.movies
        }
    )

    return render_template(
        "index.html",
        genres=genres,
        languages=languages,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)