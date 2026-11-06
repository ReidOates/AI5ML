
from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, render_template, request


# ==================================================
# 1. KONFIGURASI APLIKASI
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "titanic_artifacts.joblib"

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "public"),
    static_url_path=""
)


# ==================================================
# 2. MEMUAT MODEL DAN PREPROCESSING
# ==================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_PATH}"
    )

artifacts = joblib.load(MODEL_PATH)

model_A = artifacts["model_A"]
model_B = artifacts["model_B"]

preprocessing = artifacts["preprocessing"]

age_median = preprocessing["age_median"]
embarked_mode = preprocessing["embarked_mode"]
feature_columns = preprocessing["feature_columns"]


# ==================================================
# 3. HALAMAN UTAMA
# ==================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    error = None

    selected_model = "model_A"

    if request.method == "POST":

        selected_model = request.form.get(
            "model", "model_A"
        )

        try:

            # ======================================
            # 4. MENGAMBIL INPUT FORMULIR
            # ======================================

            pclass = int(request.form.get("pclass"))
            sex = request.form.get("sex")
            age_input = request.form.get("age", "").strip()

            sibsp = int(request.form.get("sibsp"))
            parch = int(request.form.get("parch"))

            fare = float(request.form.get("fare"))

            embarked = request.form.get(
                "embarked", ""
            ).strip().upper()

            # ======================================
            # 5. VALIDASI INPUT
            # ======================================

            if pclass not in [1, 2, 3]:
                raise ValueError(
                    "Kelas penumpang harus 1, 2, atau 3."
                )

            if sex not in ["male", "female"]:
                raise ValueError(
                    "Jenis kelamin tidak valid."
                )

            if sibsp < 0 or parch < 0:
                raise ValueError(
                    "Jumlah anggota keluarga tidak boleh negatif."
                )

            if fare < 0:
                raise ValueError(
                    "Tarif tiket tidak boleh negatif."
                )

            if age_input == "":
                age = float(age_median)
            else:
                age = float(age_input)

            if age < 0 or age > 100:
                raise ValueError(
                    "Usia harus berada antara 0 sampai 100 tahun."
                )

            if embarked == "":
                embarked = embarked_mode

            if embarked not in ["C", "Q", "S"]:
                raise ValueError(
                    "Pelabuhan harus C, Q, atau S."
                )

            if selected_model not in ["model_A", "model_B"]:
                raise ValueError(
                    "Pilihan model tidak valid."
                )

            # ======================================
            # 6. ONE-HOT ENCODING MANUAL
            # ======================================

            input_features = {
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
            }

            # Membuat DataFrame sesuai urutan fitur training
            input_data = pd.DataFrame(
                [input_features]
            ).reindex(
                columns=feature_columns,
                fill_value=0
            )

            # ======================================
            # 7. MEMILIH MODEL
            # ======================================

            if selected_model == "model_A":
                selected_model_object = model_A
                model_name = "Model A - Unpruned"

            else:
                selected_model_object = model_B
                model_name = "Model B - Pruned"

            # ======================================
            # 8. MELAKUKAN PREDIKSI
            # ======================================

            result = int(
                selected_model_object.predict(input_data)[0]
            )

            # ======================================
            # 9. MENENTUKAN HASIL PREDIKSI
            # ======================================

            if result == 1:

                prediction = {
                    "class": "SELAMAT",
                    "status": "success",
                    "description": (
                        "Model memprediksi bahwa penumpang "
                        "memiliki status selamat."
                    ),
                    "model": model_name
                }

            else:

                prediction = {
                    "class": "TIDAK SELAMAT",
                    "status": "danger",
                    "description": (
                        "Model memprediksi bahwa penumpang "
                        "memiliki status tidak selamat."
                    ),
                    "model": model_name
                }

        except (ValueError, TypeError, KeyError) as e:

            error = str(e)

        except Exception:

            error = (
                "Terjadi kesalahan saat melakukan prediksi. "
                "Periksa input dan konfigurasi model."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        selected_model=selected_model
    )


# ==================================================
# 10. MENJALANKAN APLIKASI SECARA LOKAL
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)
