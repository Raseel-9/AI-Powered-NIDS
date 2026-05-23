import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# FEATURE NAMES DEFINITION

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

print("-" * 80)
print("NETWORK INTRUSION DETECTION SYSTEM - INITIALIZING CORE COMPONENTS")
print("-" * 80)


# STAGE 1: DATA INGESTION

print("\n[LOG] Loading training and testing datasets...")
try:
    train_df = pd.read_csv('train_data.txt', names=columns)
    test_df = pd.read_csv('test_data.txt', names=columns)
    print(f"[SUCCESS] Ingested {len(train_df)} training rows and {len(test_df)} testing rows.")
except Exception as e:
    print(f"[FATAL ERROR] Failed to load datasets: {str(e)}")


# STAGE 2: DATA PREPROCESSING

print("[LOG] Initiating feature cleaning and label transformation...")

train_df.drop(['difficulty_level'], axis=1, inplace=True)
test_df.drop(['difficulty_level'], axis=1, inplace=True)

train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

categorical_cols = ['protocol_type', 'service', 'flag']
encoder = LabelEncoder()

for col in categorical_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

print("[SUCCESS] Data preprocessing and encoding finalized.")


# STAGE 3: MODEL TRAINING (RANDOM FOREST)

print("\n" + "-" * 80)
print("EXECUTION: COMMENCING MODEL TRAINING PHASE")
print("-" * 80)

X_train = train_df.drop(['label'], axis=1)
y_train = train_df['label']
X_test = test_df.drop(['label'], axis=1)
y_test = test_df['label']

print("[STATUS] Classifier training in progress... Please wait.")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("[SUCCESS] Model training sequence completed.")


# STAGE 4: MODEL PERSISTENCE

print("\n[LOG] Serializing model for deployment...")
joblib.dump(clf, 'trained_ids_model.pkl')
print("[SUCCESS] Model exported to: 'trained_ids_model.pkl'")


# STAGE 5: SYSTEM EVALUATION

print("\n" + "-" * 80)
print("FINAL SYSTEM EVALUATION AND METRICS")
print("-" * 80)

predictions = clf.predict(X_test)
acc_score = accuracy_score(y_test, predictions)
print(f"OVERALL SYSTEM ACCURACY: {acc_score * 100:.2f}%")
print("\nCLASSIFICATION PERFORMANCE SUMMARY:")
print(classification_report(y_test, predictions))
print("-" * 80)
