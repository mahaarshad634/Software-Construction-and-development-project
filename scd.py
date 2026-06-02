import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

# CLASSIFICATION MODELS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# REGRESSION MODELS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# 1. Load and Clean Data FIRST
file_upload = input("📁 Enter your CSV file: ")
if file_upload:
    try:
        df = pd.read_csv(file_upload)
        print("📊 Data loaded:", df.shape)
        print(df.head())

        # Clean raw numerical data using mean substitution
        df = df.fillna(df.select_dtypes(include=[np.number]).mean())
        print(f"✅ Cleaned data: {df.shape}")

    except Exception as e:
        print(f"❌ Error: {e}")
        exit()
else:
    print("❌ No file")
    exit()


# 2. AutoMLTrainer CLASS
class AutoMLTrainer:

    def __init__(self):
        # Optimized parameters to ensure models can converge on a best-fit path
        self.class_models = {
            'small dataset': {
                'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
                'Logistic': LogisticRegression(random_state=42, max_iter=2000, C=0.1),
                'DecisionTree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=15, weights='distance'),
                'NaiveBayes': GaussianNB()
            },
            'medium dataset': {
                'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
                'Logistic': LogisticRegression(random_state=42, max_iter=3000),
                'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=21)
            },
            'big dataset': {
                'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
                'Logistic': LogisticRegression(solver='saga', random_state=42, max_iter=1000),
                'DecisionTree': DecisionTreeClassifier(max_depth=15, random_state=42)
            }
        }
      
        self.reg_models = {
            'small dataset': {
                'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
                'Ridge': Ridge(alpha=10.0, random_state=42),
                'Linear': LinearRegression(),
                'DecisionTree': DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=42),
                'KNN': KNeighborsRegressor(n_neighbors=15, weights='distance')
            },
            'medium dataset': {
                'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
                'Ridge': Ridge(random_state=42),
                'DecisionTree': DecisionTreeRegressor(random_state=42)
            },
            'big dataset': {
                'RandomForest': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
                'Ridge': Ridge(solver='saga', random_state=42),
                'DecisionTree': DecisionTreeRegressor(max_depth=15, random_state=42)
            }
        }
        self.model_type = None
        self.label_encoder = None

    @staticmethod
    def datasetcheck(rows, cols):
        if rows < 10000 and cols < 50:
            return "small dataset"
        elif rows < 100000 and cols < 500:
            return "medium dataset"
        else:
            return "big dataset"

    def model_detection(self, target):
        if target.dtype == "object" or target.dtype.name == "category":
            return "classification", target
        if str(target.dtype).startswith(('int', 'uint')) and target.nunique() <= 10:
            return "classification", target
            
        if np.issubdtype(target.dtype, np.number):
            skewness = target.skew()
            if abs(skewness) > 2.5:
                print(f"⚠️ Target skewness is severe ({skewness:.2f}). Auto-binning target into Classification.")
                binned_target = pd.qcut(target, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                return "classification", binned_target

        return "regression", target

    def prepare_data(self, df, target_col, y_series):
        """Advanced Cleaning Engine: Strips database noise while preserving pattern signals"""
        X = df.copy()

        # 1. Strip Identification Noise Immediately
        # Columns like user_id alter distance metrics radically and skew model learning
        cols_to_remove = [target_col]
        for col in X.columns:
            if col.lower() in ['user_id', 'id', 'unnamed: 0']:
                cols_to_remove.append(col)
        X = X.drop(columns=cols_to_remove)

        # 2. Extract Time-Series Factors from Dates
        for col in X.columns:
            if 'date' in col.lower() or 'timestamp' in col.lower() or X[col].dtype == 'object':
                sample_val = str(X[col].iloc[0])
                if '-' in sample_val or ':' in sample_val:
                    try:
                        converted_date = pd.to_datetime(X[col])
                        X[col + '_hour'] = converted_date.dt.hour
                        X[col + '_dayofweek'] = converted_date.dt.dayofweek
                        X = X.drop(columns=[col])
                        print(f"⚙️ Unpacked timeline signals from: '{col}'")
                    except:
                        pass

        # 3. Smart Feature Binning
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].nunique() > 20 and abs(X[col].skew()) > 1.5:
                print(f"📦 Auto-binned skewed numeric feature: '{col}'")
                X[col] = pd.qcut(X[col], q=4, labels=['Tier1', 'Tier2', 'Tier3', 'Tier4'], duplicates='drop')

        # 4. Standard Dummy Encoding
        X = pd.get_dummies(X, drop_first=True) # drop_first eliminates multicollinearity for Linear/Logistic models
        y = y_series.copy()

        # Isolate Label Encoding for Classification targets
        if self.model_type == "classification" and (y.dtype == "object" or y.dtype.name == "category" or isinstance(y.dtype, pd.CategoricalDtype)):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self.label_encoder = le

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Stratified style split if classification to protect pattern proportions
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, 
            stratify=y if self.model_type == "classification" else None
        )

        return X_train, X_test, y_train, y_test, scaler

    def train_all_models(self, df, target_col):
        target_sample = df[target_col]
        self.model_type, y_processed = self.model_detection(target_sample)
        print(f"\n🔍 AUTOML DESIGNATED MODE: {self.model_type.upper()}")

        rows, cols = df.shape
        size_group = self.datasetcheck(rows, cols)
        print(f"📊 DATASET GROUP: {size_group.upper()} ({rows} rows, {cols} cols)")

        # Run safe preprocessing pipeline
        X_train, X_test, y_train, y_test, scaler = self.prepare_data(df, target_col, y_processed)

        if self.model_type == "classification":
            models = self.class_models[size_group]
        else:
            models = self.reg_models[size_group]

        print(f"\n🚀 Training {len(models)} optimized models...")
        results = {}

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if self.model_type == "classification":
                    score = accuracy_score(y_test, y_pred)
                else:
                    score = r2_score(y_test, y_pred)

                results[name] = score
            except Exception as e:
                print(f"❌ Failed to train {name}: {str(e)}")

        if results:
            # Sort results to clearly display the accuracy rank ladder
            sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
            
            print("\n🏆 MODEL ACCURACY RANKINGS:")
            for rank, (name, score) in enumerate(sorted_results):
                icon = "🥇" if rank == 0 else "🔹"
                print(f"{icon} {name:14}: {score:.3f}")
                
            best_name = sorted_results[0][0]
            print(f"\n🎉 AUTOML BEST FIT CHAMPION: {best_name} ({results[best_name]:.3f})")
            return models[best_name], results, scaler
        else:
            print("❌ No models successfully trained.")
            return None, {}, scaler


# 3. MAIN EXECUTION
print("\n" + "="*60)
print("🤖 AUTO MACHINE LEARNING: OPTIMIZED FOR BEST FIT")
print("="*60)

print(f"Available columns: {list(df.columns)}")
target_col = input("🎯 Target column: ").strip().strip("'").strip('"')

trainer = AutoMLTrainer()
best_model, results, scaler = trainer.train_all_models(df, target_col)

print("\n✅ DONE!")