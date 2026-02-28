import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def generate_data(n=5000, random_state=42):
    """
    Generate synthetic classification dataset.
    """
    np.random.seed(random_state)
    df = pd.DataFrame({
        'age':               np.random.randint(18, 75, n),
        'income':            np.random.normal(55000, 20000, n).astype(int),
        'credit_score':      np.random.randint(300, 850, n),
        'loan_amount':       np.random.normal(15000, 8000, n).astype(int),
        'employment_years':  np.random.randint(0, 40, n),
        'num_accounts':      np.random.randint(1, 10, n),
        'missed_payments':   np.random.randint(0, 10, n),
        'education':         np.random.choice(['High School','Bachelor','Master','PhD'], n),
        'loan_purpose':      np.random.choice(['Home','Car','Education','Personal'], n),
        'marital_status':    np.random.choice(['Single','Married','Divorced'], n),
    })

    # Target: default probability influenced by features
    default_prob = (
        0.3
        - 0.002 * (df['credit_score'] - 300) / 550
        + 0.001 * df['missed_payments']
        - 0.0001 * df['income'] / 55000
        + 0.002 * df['loan_amount'] / 15000
    )
    default_prob = np.clip(default_prob, 0.05, 0.95)
    df['default'] = np.random.binomial(1, default_prob)

    print(f"Dataset generated — Shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")
    return df


def check_data_quality(df):
    """
    Print a full data quality report.
    """
    print("\n" + "="*50)
    print("        DATA QUALITY REPORT")
    print("="*50)
    print(f"Rows              : {len(df)}")
    print(f"Columns           : {df.shape[1]}")
    print(f"Duplicates        : {df.duplicated().sum()}")
    print(f"Missing Values    : {df.isnull().sum().sum()}")
    print(f"Numeric Columns   : {len(df.select_dtypes(include=np.number).columns)}")
    print(f"Categorical Cols  : {len(df.select_dtypes(include='object').columns)}")
    print("="*50)


def impute_missing(df):
    """
    Impute missing values — median for numeric, mode for categorical.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if num_cols:
        imputer = SimpleImputer(strategy='median')
        df[num_cols] = imputer.fit_transform(df[num_cols])

    if cat_cols:
        imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer.fit_transform(df[cat_cols])

    print("Missing values imputed.")
    return df


def encode_features(df, cat_cols):
    """
    Label encode categorical columns.
    """
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"Encoded: {col}")
    return df, encoders


def scale_features(df, num_cols):
    """
    StandardScaler on numeric columns.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"Scaled: {num_cols}")
    return df, scaler


def add_features(df):
    """
    Feature engineering — add interaction and ratio features.
    """
    df = df.copy()
    df['debt_to_income']      = df['loan_amount'] / (df['income'] + 1)
    df['payment_history']     = df['missed_payments'] / (df['num_accounts'] + 1)
    df['credit_utilization']  = df['loan_amount'] / (df['credit_score'] + 1)
    df['age_income_ratio']    = df['age'] / (df['income'] + 1)
    print("Feature engineering complete — 4 new features added.")
    return df


def full_pipeline(n=5000):
    """
    Run complete preprocessing pipeline.
    Returns X, y ready for modeling.
    """
    print("\n" + "="*50)
    print("   STARTING PREPROCESSING PIPELINE")
    print("="*50)

    # Generate data
    df = generate_data(n)

    # Quality check
    check_data_quality(df)

    # Impute
    df = impute_missing(df)

    # Feature engineering
    df = add_features(df)

    # Encode categoricals
    cat_cols = ['education', 'loan_purpose', 'marital_status']
    df, encoders = encode_features(df, cat_cols)

    # Scale numerics
    num_cols = ['age', 'income', 'credit_score', 'loan_amount',
                'employment_years', 'num_accounts', 'missed_payments',
                'debt_to_income', 'payment_history',
                'credit_utilization', 'age_income_ratio']
    df, scaler = scale_features(df, num_cols)

    # Split features and target
    X = df.drop('default', axis=1)
    y = df['default']

    print(f"\nFinal feature matrix: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print("="*50)

    return X, y, encoders, scaler


if __name__ == "__main__":
    X, y, encoders, scaler = full_pipeline()
    print("\nSample features:")
    print(X.head())
