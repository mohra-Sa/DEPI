import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting model training...")

# Load data
df = pd.read_csv(r"C:\Users\DUBAI STORE\Downloads\clean_student_performance_final (2).csv")
print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Clean data - drop unnecessary columns
cols_to_drop = ["student_id", "name", "grade_level", "description", "country", 
                "teacher_name", "internet_access", "extracurricular_activities",
                "scholarship_status", "tutoring"]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Handle month
if 'month' in df.columns:
    df["month"] = pd.to_datetime(df["month"], errors='coerce')
    df["month"] = df["month"].dt.month.fillna(1).astype(int) - 9

# Fill missing values
df = df.fillna({
    'age': df['age'].median(),
    'score': df['score'].median(),
    'sleep_hours': 7,
    'study_hours': 3,
    'attendance_rate': 80,
    'homework_completion_rate': 80,
    'family_size': 4,
    'previous_gpa': 3.0,
    'exam_attempts': 1,
    'teacher_experience_years': 5,
    'parent_income': 50000,
    'feedback_rating': 4,
    'efficiency': 75,
    'hours_per_week': 10
})

print("✅ Data cleaned")

# Encode categorical columns
mappings = {
    "gender": {"female": 0, "male": 1},
    "difficulty_level": {"easy": 0, "medium": 1, "hard": 2},
    "parent_education": {"none": 0, "high school": 1, "college": 2, "postgrad": 3},
    "health_condition": {"normal": 0, "mild illness": 1, "chronic": 2},
    "performance_level": {"weak": 0, "average": 1, "good": 2, "excellent": 3}
}

for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(0)

print("✅ Categorical columns encoded")

# Encode target variable
le = LabelEncoder()
df["final_state"] = le.fit_transform(df["final_state"])
print(f"✅ Target classes: {', '.join(le.classes_)}")

# Target encoding for high-cardinality columns
target_maps = {}
target_encode_cols = ["subject_name", "city", "admission_year", "free_time_activity", "school_transport"]

for col in target_encode_cols:
    if col in df.columns:
        target_maps[col] = df.groupby(col)["final_state"].mean().to_dict()
        df[col] = df[col].map(target_maps[col]).fillna(0.5)

joblib.dump(target_maps, 'target_maps.pkl')
print("✅ Target encoding applied")

# Feature scaling - نعمل scaling لكل الأعمدة الرقمية + الـ target encoded فقط
scale_cols = [
    "age", "hours_per_week", "score", "sleep_hours", "study_hours",
    "attendance_rate", "homework_completion_rate", "family_size",
    "previous_gpa", "exam_attempts", "teacher_experience_years",
    "parent_income", "feedback_rating", "efficiency",
    "subject_name", "city", "admission_year", 
    "free_time_activity", "school_transport"
]

# نضمن إن كل الأعمدة موجودة فعلاً
scale_cols = [col for col in scale_cols if col in df.columns]

print(f"Scaling these {len(scale_cols)} columns: {scale_cols}")

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols].values)

# مهم جدًا: نحفظ أسماء الأعمدة اللي تم scaling عليها مع الـ scaler
scaler.scale_columns = scale_cols   # هيبقى موجود في الـ pkl
print(f"✅ Scaled {len(scale_cols)} features")

# Split data
x = df.drop(columns=["final_state"])
y = df["final_state"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {x_train.shape[0]} samples")
print(f"✅ Test set: {x_test.shape[0]} samples")

# Train model
print("\n⏳ Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print(f"🎯 Model Accuracy: {acc*100:.2f}%")
print("="*60)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save models
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

# نحفظ أسماء الأعمدة اللي تم scaling عليها (مهم للـ app)
joblib.dump(scaler.scale_columns, 'scale_columns.pkl')
print("Scale columns list saved as scale_columns.pkl")
print("\n✅ All models saved successfully!")
print("📦 Files created:")
print("  - model.pkl")
print("  - encoder.pkl")
print("  - scaler.pkl")
print("  - target_maps.pkl") 

print("\n🎉 Ready to run: streamlit run app.py") 