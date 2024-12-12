from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model và scaler
model = pickle.load(open("best_xgboost_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Hàm dự đoán giá nhà
def pred(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, proximity):
    # Mapping proximity từ form vào giá trị encoding
    proximity_mapping = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    ocean_proximity = proximity_mapping[proximity]

    # Dummies encoding cho proximity
    ocean_mapping = {
        "<1H OCEAN": [1, 0, 0, 0, 0],
        "INLAND": [0, 1, 0, 0, 0],
        "NEAR OCEAN": [0, 0, 1, 0, 0],
        "NEAR BAY": [0, 0, 0, 1, 0],
        "ISLAND": [0, 0, 0, 0, 1],
    }
    ocean_dummies = ocean_mapping[ocean_proximity]

    # Tính toán thêm các feature
    rooms_per_household = total_rooms / households if households > 0 else 0
    bedrooms_per_room = total_bedrooms / total_rooms if total_rooms > 0 else 0
    population_per_household = population / households if households > 0 else 0

    # Tạo feature vector
    input_features = [
        longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
        population, households, median_income
    ] + ocean_dummies + [rooms_per_household, bedrooms_per_room, population_per_household]

    # Tạo DataFrame với đúng tên cột
    input_df = pd.DataFrame([input_features], columns=[
        "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
        "population", "households", "median_income", "_1H OCEAN", "INLAND",
        "ISLAND", "NEAR BAY", "NEAR OCEAN", "rooms_per_household", "bedrooms_per_room",
        "population_per_household"
    ])

    input_features_scaled = scaler.transform(input_df)

    prediction = model.predict(input_features_scaled)
    return prediction[0]

@app.route('/')
def index():
    csv_file = 'data_prj/train.csv'  
    data = pd.read_csv(csv_file)
    
    table_data = data.to_dict(orient='records')
    columns = data.columns.tolist()
    
    return render_template('index.html', columns=columns, data=table_data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Lấy dữ liệu từ query parameters
        longitude = request.args.get('longitude', '')
        latitude = request.args.get('latitude', '')
        age = request.args.get('housing_median_age', '')
        rooms = request.args.get('total_rooms', '')
        bedrooms = request.args.get('total_bedrooms', '')
        population = request.args.get('population', '')
        households = request.args.get('households', '')
        income = request.args.get('median_income', '')
        proximity = request.args.get('proximity', '')
        median_house_value = request.args.get('median_house_value', '')

        # Render trang predict.html với dữ liệu
        return render_template(
            'predict.html',
            longitude=longitude,
            latitude=latitude,
            age=age,
            rooms=rooms,
            bedrooms=bedrooms,
            population=population,
            households=households,
            income=income,
            proximity=proximity,
            median_house_value=median_house_value,
            result=None,
            error_value=None
        )

    elif request.method == 'POST':
        action = request.form.get('action', '')  # Lấy giá trị action từ form

        if action == 'clear':
            # Render lại form với các giá trị rỗng
            return render_template(
                'predict.html',
                longitude='',
                latitude='',
                age='',
                rooms='',
                bedrooms='',
                population='',
                households='',
                income='',
                proximity='',
                median_house_value='',
                result=None,
                error_value=None
            )
        elif action == 'predict':
            try:
                # Lấy dữ liệu từ form
                longitude = float(request.form.get('longitude', 0))
                latitude = float(request.form.get('latitude', 0))
                age = float(request.form.get('age', 0))
                rooms = float(request.form.get('rooms', 0))
                bedrooms = float(request.form.get('bedrooms', 0))
                population = float(request.form.get('population', 0))
                households = float(request.form.get('households', 0))
                income = float(request.form.get('income', 0))
                proximity = int(request.form.get('proximity', 0))  # Giá trị proximity
                median_house_value = float(request.form.get('median_house_value', 0))

                # Kiểm tra giá trị proximity
                if proximity < 0 or proximity > 4:
                    return render_template('predict.html', result="Invalid proximity selected")

                # Gọi hàm dự đoán
                predicted_price = pred(
                    longitude, latitude, age, rooms, bedrooms, population, households, income, proximity
                )

                # Tính toán sai số nếu giá trị thực > 0
                error_value = None
                if median_house_value > 0:
                    error_value = ((predicted_price - median_house_value) / median_house_value) * 100
                    error_value = f"+{round(error_value, 2)}%" if error_value > 0 else f"{round(error_value, 2)}%"

                # Hiển thị kết quả
                return render_template(
                    'predict.html',
                    longitude=longitude,
                    latitude=latitude,
                    age=age,
                    rooms=rooms,
                    bedrooms=bedrooms,
                    population=population,
                    households=households,
                    income=income,
                    proximity=proximity,
                    median_house_value=f"{median_house_value:,.2f}",
                    result=f"{predicted_price:,.2f}",
                    error_value=error_value 
                )
            except ValueError as e:
                # Xử lý lỗi ValueError khi dữ liệu không hợp lệ
                return render_template('predict.html', result="Invalid input. Please enter all fields correctly.")
            except Exception as e:
                # Xử lý lỗi khác
                return render_template('predict.html', result=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
