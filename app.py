import joblib
import os
from flask import Flask, redirect, request, render_template, url_for, flash
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


from model import RnnModel
from photo_prep import Extractor

if not os.path.exists("uploads"):
    os.makedirs("uploads")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg"}


app = Flask(__name__)
app.secret_key = "wb"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

path_to_model = "final_model/model_3.288113594055176.h5"
path_to_vocab = "final_model/vocabulary.pkl"
model = RnnModel(load_model(path_to_model), joblib.load(path_to_vocab), 36)
extractor = Extractor()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", text="Upload your picture", prediction="")
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            return render_template(
                "index.html", text="No file was found. Please upload a picture"
            )

        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return render_template(
                "index.html", text="No file was found. Please upload a picture"
            )

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            features = extractor.extract_features(UPLOAD_FOLDER)
            prediction = model.predict(list(features.values())[0])
            caption = ""
            for word in prediction:
                caption += word + " "
            caption[:-1] + "."
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            return render_template("index.html", prediction=caption)
        return render_template(
            "index.html", text="Please upload a picture in .jpg format"
        )


if __name__ == "__main__":
    # You want to put the value of the env variable PORT if it exist (some services only open specifiques ports)
    port = int(os.environ.get("PORT", 4000))
    # Threaded option to enable multiple instances for
    # multiple user access support
    # You will also define the host to "0.0.0.0" because localhost will only be reachable from inside de server.
    app.run(debug=True, host="0.0.0.0", threaded=True, port=port)
