from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load your hospital info dataset (ensure it's in the same folder or provide full path)
hospital_df = pd.read_excel("Hospitals_list.xlsx")  # or .csv if you're using csv

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load('Model_Training\Hospital_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    age = int(request.form['age'])
    gender = request.form['gender']
    locality = request.form['locality']
    phone = request.form['phone']
    disease = request.form['disease']
    emergency = 1 if 'emergency' in request.form else 0

    # Symptoms and pre-conditions (checkboxes)
    symptom_list = ['Fever','Cough','Cold','Chest_Pain','Breathlessness','Headache',
                    'Stomach_Pain','Vomiting','Rash','Tooth_Pain','Ear_Pain','Eye_Irritation',
                    'Joint_Pain','Fatigue','Dizziness','Diabetes','B.P']

    symptoms = [1 if s in request.form else 0 for s in symptom_list]

    gender_lst =['Female','Male']
    diseases_lst = ["Dental", "Emergency", "ENT", "Eye", "Gastro", "General", "Gynae", "Heart",
                    "Lungs", "Mental Health", "Neuro", "Ortho", "Pediatrician", "Skin", "Urology"]
    Hospital_type_list = ["Multi-specialty", "ENT", "Eye","General","Gynecology", "Cardiology",
                              "Lung Care","Neuro Surgery", "Orthopedics", "Pediatrics", "Dermatology", "Urology"]
    
    # Encode categorical values
    gender_encoded = gender_lst.index(gender)
    disease_encoded = diseases_lst.index(disease)

    # Final input
    final_input = [age, gender_encoded] + symptoms + [disease_encoded, emergency]

    # Predict
    hospital_type = model.predict([final_input])[0]

    # Filter hospitals of that type (column with value 1)
    filtered_hospitals = hospital_df[hospital_df[hospital_type] == 1]

    # Sort by distance and pick top 5
    top5 = filtered_hospitals.sort_values('Distance (km)').head(5)

    # Convert to dictionary to send to HTML
    hospital_list = top5[['Hospital Name', 'Location', 'Distance (km)', 'Contact Details']].to_dict(orient='records')


    return render_template('result.html', prediction= hospital_type, hospitals=hospital_list)

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True)
