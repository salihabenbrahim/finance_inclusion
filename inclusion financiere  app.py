import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import streamlit as st

# Charger les données
data = pd.read_csv("./Financial_inclusion_dataset.csv")

# Encodage des colonnes catégoriques
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}

for column in categorical_columns:
    if column != 'uniqueid':  # Exclure la colonne uniqueid
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Diviser les données en variables indépendantes (X) et cible (y)
X = data.drop(columns=['bank_account', 'uniqueid'])  # Variable cible à prédire
y = data['bank_account']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
print("Rapport de classification :")
print(classification_report(y_test, y_pred))
print(f"Accuracy score : {accuracy_score(y_test, y_pred)}")

# Sauvegarder et charger le modèle
joblib.dump(model, 'model_financal.pkl')
model = joblib.load('model_financal.pkl')

# Interface Streamlit
st.title('Financial Inclusion Prediction App')

# Saisie utilisateur avec décodeur pour afficher les valeurs originales
def get_options(column):
    return label_encoders[column].inverse_transform(range(len(label_encoders[column].classes_)))

country = st.selectbox("Country", options=get_options('country'))
year = st.number_input("Year", min_value=data['year'].min(), max_value=data['year'].max(), step=1)
location_type = st.selectbox("Location Type", options=get_options('location_type'))
cellphone_access = st.selectbox("Cellphone Access", options=get_options('cellphone_access'))
household_size = st.number_input("Household Size", min_value=data['household_size'].min(), max_value=data['household_size'].max(), step=1)
age_of_respondent = st.number_input("Age of Respondent", min_value=data['age_of_respondent'].min(), max_value=data['age_of_respondent'].max(), step=1)
gender_of_respondent = st.selectbox("Gender", options=get_options('gender_of_respondent'))
relationship_with_head = st.selectbox("Relationship with Head", options=get_options('relationship_with_head'))
marital_status = st.selectbox("Marital Status", options=get_options('marital_status'))
education_level = st.selectbox("Education Level", options=get_options('education_level'))
job_type = st.selectbox("Job Type", options=get_options('job_type'))

# Préparer les données utilisateur
user_input = pd.DataFrame({
    'country': [label_encoders['country'].transform([country])[0]],
    'year': [year],
    'location_type': [label_encoders['location_type'].transform([location_type])[0]],
    'cellphone_access': [label_encoders['cellphone_access'].transform([cellphone_access])[0]],
    'household_size': [household_size],
    'age_of_respondent': [age_of_respondent],
    'gender_of_respondent': [label_encoders['gender_of_respondent'].transform([gender_of_respondent])[0]],
    'relationship_with_head': [label_encoders['relationship_with_head'].transform([relationship_with_head])[0]],
    'marital_status': [label_encoders['marital_status'].transform([marital_status])[0]],
    'education_level': [label_encoders['education_level'].transform([education_level])[0]],
    'job_type': [label_encoders['job_type'].transform([job_type])[0]]
})

# Prédiction
if st.button('Predict'):
    prediction = model.predict(user_input)
    result = 'Has a bank account' if prediction[0] == 1 else 'Does not have a bank account'
    st.write(result)
