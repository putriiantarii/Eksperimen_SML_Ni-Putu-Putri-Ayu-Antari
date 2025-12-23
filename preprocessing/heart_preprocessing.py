
#1. Import Library
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#2. Memuat Data

#Load Dataset
def preprocess_heart_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
df = pd.read_csv("heart.csv")

#3. Data Preprocessing
#Merapikan Nama Kolom
df = df.rename(columns={
        'chest pain type': 'chest_pain_type',
        'resting bp s': 'resting_bp_s',
        'fasting blood sugar': 'fasting_blood_sugar',
        'resting ecg': 'resting_ecg',
        'max heart rate': 'max_heart_rate',
        'exercise angina': 'exercise_angina',
        'ST slope': 'st_slope'
    })

#Menghapus Data Duplikat
df = df.drop_duplicates()

#Split Data (Fitur dan Target)
X = df.drop('target', axis=1)
y = df['target']

#Group Fitur
numeric_features = [
        'age',
        'resting_bp_s',
        'cholesterol',
        'max_heart_rate',
        'oldpeak'
    ]

categorical_features = [
        'sex',
        'chest_pain_type',
        'fasting_blood_sugar',
        'resting_ecg',
        'exercise_angina',
        'st_slope'
    ]

#Pipeline
numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

#Split Data
X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

#Transform
X_train_processed = preprocessor.fit_transform(X_train)

if sp.issparse(X_train_processed):
    X_train_processed = X_train_processed.toarray()

#Penamaan Fitur
cat_features = preprocessor.named_transformers_['cat'] \
    .named_steps['encoder'] \
    .get_feature_names_out(categorical_features)

feature_names = numeric_features + list(cat_features)

#4. Menyimpan File .csv
def preprocess_heart_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
df = pd.read_csv("heart.csv")

df_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

df_processed['target'] = y_train.values

#Save File .csv
df_processed.to_csv(output_csv, index=False)

print("Preprocessing selesai. File tersimpan:", output_csv)
