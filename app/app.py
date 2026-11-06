
from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, render_template, request


# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "titanic_artifacts.joblib"


# ==========================================
# 2. MEMUAT MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_PATH}"
    )

artifacts = joblib.load(MODEL_PATH)

model_A = artifacts["model_A"]
model_B = artifacts["model_B"]

preprocessing = artifacts["preprocessing"]

feature_columns = preprocessing["feature_columns"]

age_median = preprocessing["age_median"]
embarked_mode = preprocessing["embarked_mode"]


# ==========================================
# 3. HALAMAN UTAMA
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    selected_model = "model_A"
    error = None

    if request.method == "POST":

        try:
            # Mengambil input dari formulir
            pclass = int(request.form["pclass"])
            sex = request.form["sex"]

            age_input = request.form.get("age", "").strip()

            sibsp = int(request.form["sibsp"])
            parch = int(request.form["parch"])
            fare = float(request.form["fare"])

            embarked = request.form.get("embarked", "").strip()

            selected_model = request.form["model"]

            # Validasi input
            if pclass not in [1, 2, 3]:
                raise ValueError("Kelas penumpang tidak valid.")

            if sex not in ["male", "female"]:
                raise ValueError("Jenis kelamin tidak valid.")

            if sibsp < 0 or parch < 0 or fare < 0:
                raise ValueError("Jumlah keluarga dan tarif tidak boleh negatif.")

            if age_input == "":
                age = age_median
            else:
                age = float(age_input)

            if age < 0:
                raise ValueError("Usia tidak boleh negatif.")

            if embarked == "":
                embarked = embarked_mode

            if embarked not in ["C", "Q", "S"]:
                raise ValueError("Pelabuhan keberangkatan tidak valid.")

            # Membuat data input
            input_data = pd.DataFrame([{
                "Pclass": pclass,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Sex_female": 1 if sex == "female" else 0,
                "Sex_male": 1 if sex == "male" else 0,
                "Embarked_C": 1 if embarked == "C" else 0,
                "Embarked_Q": 1 if embarked == "Q" else 0,
                "Embarked_S": 1 if embarked == "S" else 0
            }])

            # Menyesuaikan urutan kolom dengan data training
            input_data = input_data.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # Memilih model
            if selected_model == "model_A":
                selected_model_object = model_A
            elif selected_model == "model_B":
                selected_model_object = model_B
            else:
                raise ValueError("Pilihan model tidak valid.")

            # Melakukan prediksi
            result = selected_model_object.predict(input_data)[0]

            if int(result) == 1:
                prediction = {
                    "class": "Selamat",
                    "description": "Model memprediksi penumpang selamat.",
                    "status": "success"
                }
            else:
                prediction = {
                    "class": "Tidak Selamat",
                    "description": "Model memprediksi penumpang tidak selamat.",
                    "status": "danger"
                }

        except (ValueError, KeyError, TypeError) as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        selected_model=selected_model,
        error=error
    )


# ==========================================
# 4. MENJALANKAN APLIKASI LOKAL
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
