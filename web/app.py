import os
from datetime import datetime
from dotenv import load_dotenv
import json

# packages required for Flask
from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine, text

# ScamShieldPredictor class imported from predict.py
from ml.predict import ScamShieldPredictor

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
FLASK_SECRET = os.getenv("FLASK_SECRET", "dev-secret")

app = Flask(__name__)
app.secret_key = FLASK_SECRET

# ML predictor (loads vectorizer + logreg immediately; NN lazy-loads)
predictor = ScamShieldPredictor()

# DB engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def init_db():
    """Creates the scans table if it doesn't exist"""
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS scans (
           id INT AUTO_INCREMENT PRIMARY KEY,
           created_at DATETIME NOT NULL,
           model_used VARCHAR(10) NOT NULL,
           threshold FLOAT NOT NULL,
           message_text TEXT NOT NULL,
           probability FLOAT NOT NULL,
           risk_score INT NOT NULL,
           label INT NOT NULL, 
           verdict VARCHAR(10) NOT NULL,
           cluster_id INT NULL,
           cluster_terms TEXT NULL
        );
        """))


# @app.before_first_request
# def startup():
#     init_db()

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")


@app.route("/scan", methods=['POST'])
def scan():
    message = request.form.get("message", "").strip()
    model = request.form.get("model", "logreg")
    threshold = float(request.form.get("threshold", "0.5"))

    if not message:
        flash("Please paste a message first.")
        return redirect(url_for("index"))

    # predict
    result = predictor.predict(text = message,
                               model = model,
                               threshold = threshold,
                               return_embedding = False)

    if "error" in result:
        flash(result["error"])
        return redirect(url_for("index"))

    # extract cluster information
    cluster_id = None
    cluster_terms = None

    if result.get("cluster") and isinstance(result["cluster"], dict) and "error" not in result["cluster"]:
        cluster_id = result["cluster"].get("id")
        cluster_terms = json.dumps(result["cluster"].get("top_terms", []))


    # saving the information to database
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                    INSERT INTO scans (created_at, model_used, threshold, message_text, probability, risk_score, label, verdict, cluster_id, cluster_terms)
                    VALUES (:created_at, :model_used, :threshold, :message_text, :probability, :risk_score, :label, :verdict, :cluster_id, :cluster_terms)
                """),
            {
                "created_at": datetime.now(),
                "model_used": result["model_used"],
                "threshold": result["threshold"],
                "message_text": message,
                "probability": result["probability"],
                "risk_score": result["risk_score"],
                "label": result["label"],
                "verdict": result["verdict"],
                "cluster_id": cluster_id,
                "cluster_terms": cluster_terms
            }
        )

    return render_template("result.html", message = message, result = result)


@app.route("/history", methods=['GET'])
def history():
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, created_at, model_used, threshold, message_text, probability, risk_score, label, verdict
            FROM scans
            ORDER BY id DESC
            LIMIT 50
        """)).mappings().all()
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    init_db()
    app.run(debug = True)

    # host = "127.0.0.1"
    # port = 5000
    #
    # print("\n======================================")
    # print("ScamShield started successfully!")
    # print(f"Open in browser: http://{host}:{port}/")
    # print("======================================\n")
    #
    # app.run(host = host, port = port, debug = True)
