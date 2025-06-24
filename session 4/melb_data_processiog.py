import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer

# Load CSV
df = pd.read_csv("/content/melb_data.csv")  # Use your actual file name
print("Step 0 - Data Loaded")
print(df.head(), "\n")

# 1. Check for nulls in each column
print("Step 1 - Nulls in each column:")
print(df.isnull().sum(), "\n")

# 2. Visualize missing values with heatmap
print("Step 2 - Heatmap of missing values:")
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Data Heatmap")
plt.show()

# 3. Drop rows where 'Car' is null
df = df.dropna(subset=['Car'])
print("Step 3 - Dropped rows where 'Car' is null. Shape:", df.shape, "\n")

# 4. Fill 'Car' nulls with median
med_car = df['Car'].median()
df['Car'].fillna(med_car, inplace=True)
print("Step 4 - Filled 'Car' with median:", med_car, "\n")

# 5. Fill 'BuildingArea' with mean
mean_ba = df['BuildingArea'].mean()
df['BuildingArea'].fillna(mean_ba, inplace=True)
print("Step 5 - Filled 'BuildingArea' with mean:", round(mean_ba, 2), "\n")

# 6. Fill 'YearBuilt' with mode
mode_yb = df['YearBuilt'].mode()[0]
df['YearBuilt'].fillna(mode_yb, inplace=True)
print("Step 6 - Filled 'YearBuilt' with mode:", mode_yb, "\n")

# 7. Drop 'CouncilArea'
if 'CouncilArea' in df.columns:
    df.drop('CouncilArea', axis=1, inplace=True)
    print("Step 7 - Dropped 'CouncilArea'\n")
else:
    print("Step 7 - 'CouncilArea' already not present\n")

# 8. Fill remaining nulls with column means
df.fillna(df.mean(numeric_only=True), inplace=True)
print("Step 8 - Filled remaining nulls with column means\n")

# 9. Show rows with more than 2 nulls
more_than_2_nulls = df[df.isnull().sum(axis=1) > 2]
print("Step 9 - Rows with more than 2 nulls:")
print(more_than_2_nulls, "\n")

# 10. Replace nulls in 'BuildingArea' with 0
df['BuildingArea'].fillna(0, inplace=True)
print("Step 10 - Replaced 'BuildingArea' nulls with 0\n")

# 11. Create new column for missing indicator
df['BA_missing'] = df['BuildingArea'].isnull().astype(int)
print("Step 11 - Added 'BA_missing' column (0: present, 1: missing):")
print(df[['BuildingArea', 'BA_missing']].head(), "\n")

# 12. Replace 'Car' nulls with random values from 1 to 4
rand_vals = np.random.randint(1, 5, size=df['Car'].isnull().sum())
df.loc[df['Car'].isnull(), 'Car'] = rand_vals
print("Step 12 - Replaced 'Car' nulls with random values [1–4]\n")

# 13. Use KNNImputer for numeric columns
knn = KNNImputer(n_neighbors=3)
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = knn.fit_transform(df[num_cols])
print("Step 13 - Used KNNImputer for numeric columns\n")

# 14. Show % of missing data per column
null_pct = df.isnull().mean() * 100
print("Step 14 - % of missing values per column:")
print(null_pct[null_pct > 0], "\n")

# 15. Save cleaned DataFrame
df.to_csv("Melbourne_cleaned.csv", index=False)
print("Step 15 - Cleaned data saved as 'Melbourne_cleaned.csv'")

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 0: Load dataset
df = pd.read_csv("/content/melb_data.csv")
print("✅ Loaded cleaned dataset\n")

# Step 16: Identify all object (categorical) columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("📌 Step 16 - Categorical Columns:\n", cat_cols, "\n")

# Step 17: Label Encode 'Type'
if 'Type' in df.columns:
    df['Type_LE'] = LabelEncoder().fit_transform(df['Type'].astype(str))
    print("✅ Step 17 - Label Encoded 'Type'\n")

# Step 18: Label Encode 'Method' and 'SellerG'
for col in ['Method', 'SellerG']:
    if col in df.columns:
        df[col + '_LE'] = LabelEncoder().fit_transform(df[col].astype(str))
        print(f"✅ Step 18 - Label Encoded '{col}'\n")

# Step 19: OneHotEncoder for 'Regionname'
if 'Regionname' in df.columns:
    dummies = pd.get_dummies(df['Regionname'], prefix='Region')
    df = pd.concat([df, dummies], axis=1)
    print("✅ Step 19 - OneHotEncoded 'Regionname'\n")

# Step 20: get_dummies on all object columns
df_dummies = pd.get_dummies(df[cat_cols], drop_first=True)
df = pd.concat([df, df_dummies], axis=1)
print("✅ Step 20 - get_dummies on all object-type columns\n")

# Step 21: Drop original categorical columns
df.drop(columns=cat_cols, inplace=True)
print("✅ Step 21 - Dropped original categorical columns\n")

# Step 22: Extract year and month from 'Date'
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    print("✅ Step 22 - Extracted 'Year' and 'Month' from 'Date'\n")

print("✅ Loaded cleaned dataset\n")
# Step 22: Extract year & month from 'Date'
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    print("✅ Step 22 - Extracted 'Year' and 'Month'\n")

# Step 23: Map Type to custom values
if 'Type' in df.columns:
    df['Type_custom'] = df['Type'].map({'h': 0, 'u': 1, 't': 2})
    print("✅ Step 23 - Mapped 'Type' to {'h': 0, 'u': 1, 't': 2}\n")

# Step 24: ColumnTransformer (not applied now, but ready for modeling)
if 'Method' in df.columns and 'SellerG' in df.columns:
    ct = ColumnTransformer([
        ('encode', OneHotEncoder(), ['Method', 'SellerG'])
    ], remainder='passthrough')
    print("✅ Step 24 - ColumnTransformer defined for modeling use\n")

# Step 25: .cat.codes on CouncilArea
if 'CouncilArea' in df.columns:
    df['CouncilArea'] = df['CouncilArea'].astype('category')
    df['CouncilArea_code'] = df['CouncilArea'].cat.codes
    print("✅ Step 25 - Used .cat.codes on 'CouncilArea'\n")

# Step 26: Frequency encode SellerG
if 'SellerG' in df.columns:
    freq = df['SellerG'].value_counts(normalize=True)
    df['SellerG_freq'] = df['SellerG'].map(freq)
    print("✅ Step 26 - Frequency encoded 'SellerG'\n")

# Step 27: Group rare SellerG
if 'SellerG' in df.columns:
    top10 = df['SellerG'].value_counts().nlargest(10).index
    df['SellerG_grouped'] = df['SellerG'].apply(lambda x: x if x in top10 else 'Other')
    df['SellerG_grouped_LE'] = LabelEncoder().fit_transform(df['SellerG_grouped'])
    print("✅ Step 27 - Grouped rare 'SellerG' values as 'Other'\n")

# Step 28: Function to label encode any column
def encode_column(data, col):
    le = LabelEncoder()
    data[col + '_LE'] = le.fit_transform(data[col].astype(str))
    print(f"✅ Step 28 - Encoded column '{col}'")

encode_column(df, 'Suburb') if 'Suburb' in df.columns else print("⚠️ 'Suburb' not found for Step 28\n")

# Step 29: Target encode Suburb using avg Price
if 'Suburb' in df.columns and 'Price' in df.columns:
    suburb_price = df.groupby('Suburb')['Price'].mean()
    df['Suburb_target'] = df['Suburb'].map(suburb_price)
    print("✅ Step 29 - Target encoded 'Suburb' using average Price\n")



# Step 30: Save final DataFrame
df.to_csv("Melbourne_encoded_final.csv", index=False)
print("📁 Step 30 - Saved encoded features to 'Melbourne_encoded_final.csv'")


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Load your encoded dataset
df = pd.read_csv("/content/Melbourne_encoded_final.csv")

# Step 31: List numerical features
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("📌 Step 31 - Numeric Columns:\n", num_cols)

# Step 32: StandardScaler
std_cols = ['Distance', 'Landsize', 'BuildingArea']
scaler_std = StandardScaler()
df_std = df.copy()
df_std[std_cols] = scaler_std.fit_transform(df_std[std_cols])
print("\n✅ Step 32 - Standard Scaled 'Distance', 'Landsize', 'BuildingArea'")

# Step 33: MinMaxScaler
mm_cols = ['Price', 'Rooms']
scaler_mm = MinMaxScaler()
df_std[mm_cols] = scaler_mm.fit_transform(df_std[mm_cols])
print("✅ Step 33 - MinMax Scaled 'Price' and 'Rooms'")

# Step 34: Plot comparison
for col in std_cols + mm_cols:
    plt.figure(figsize=(10, 3))
    sns.kdeplot(df[col], label='Original', fill=True)
    sns.kdeplot(df_std[col], label='Scaled', fill=True)
    plt.title(f"📊 Distribution of {col} (Original vs Scaled)")
    plt.legend()
    plt.show()

# Step 35: RobustScaler for outliers
outlier_cols = ['Landsize']
scaler_rb = RobustScaler()
df_std[outlier_cols] = scaler_rb.fit_transform(df[outlier_cols])
print("✅ Step 35 - Applied RobustScaler on 'Landsize'")

# Step 36: ColumnTransformer
ct = ColumnTransformer([
    ('scale_std', StandardScaler(), std_cols),
    ('scale_minmax', MinMaxScaler(), mm_cols)
], remainder='passthrough')

df_ct = ct.fit_transform(df)
print("✅ Step 36 - Applied ColumnTransformer to selected features")

# Step 37: Save the scaled data
df_std.to_csv("Melbourne_scaled.csv", index=False)
print("📁 Step 37 - Scaled data saved to 'Melbourne_scaled.csv'")

# Step 38: PowerTransformer on skewed features
skewed_cols = ['Price', 'Landsize']
pt = PowerTransformer()
df_std[skewed_cols] = pt.fit_transform(df[skewed_cols])
print("✅ Step 38 - Applied PowerTransformer to normalize skewed features")

# Step 39: Histogram of a scaled feature
df_std[skewed_cols].hist(bins=20, figsize=(10, 4), color='skyblue')
plt.suptitle("📊 Step 39 - Histograms of Scaled Features")
plt.show()

# Step 40: Function to apply any scaler
def scale_columns(df, cols, scaler):
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df_scaled[cols])
    print(f"✅ Step 40 - Applied {scaler.__class__.__name__} to columns: {cols}")
    return df_scaled

# Example usage:
df_example = scale_columns(df, ['Distance', 'Rooms'], StandardScaler())

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Melbourne_scaled.csv")
print("✅ Loaded data")

# Step 41: Define features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']
print("📌 Step 41 - Features and target defined\n")

# Step 42: Simple 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("📌 Step 42 - Basic train-test split done\n")

# Step 43: Stratified split by 'Regionname' (if exists)
if 'Regionname' in df.columns:
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(df, df['Regionname']):
        strat_train_set = df.loc[train_idx]
        strat_test_set = df.loc[test_idx]
    print("📌 Step 43 - Stratified split by 'Regionname'\n")
else:
    strat_train_set, strat_test_set = X_train.copy(), X_test.copy()
    print("⚠️ 'Regionname' not found. Used normal split.\n")

# Step 44: Print shapes
print("📏 Step 44 - Train shape:", X_train.shape)
print("📏 Step 44 - Test shape :", X_test.shape, "\n")

# Step 45: Using fixed random_state
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=123)
print("📌 Step 45 - train_test_split with fixed random_state\n")

# Step 46: Visualize price distribution
plt.figure(figsize=(10, 4))
sns.kdeplot(y_train, label='Train', fill=True)
sns.kdeplot(y_test, label='Test', fill=True)
plt.title("📊 Step 46 - Price Distribution in Train vs Test")
plt.legend()
plt.show()

# Step 47: Preprocessing pipeline
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

num_pipe = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

full_pipe = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

X_train_prepared = full_pipe.fit_transform(X_train)
X_test_prepared = full_pipe.transform(X_test)
print("🔧 Step 47 - Preprocessing done with Pipeline\n")

# Step 48: Linear Regression Model
model = LinearRegression()
model.fit(X_train_prepared, y_train)
score = model.score(X_test_prepared, y_test)
print(f"📈 Step 48 - Linear Regression R^2 Score: {score:.4f}\n")

# Step 49: Save train/test sets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("💾 Step 49 - Saved train/test sets to CSV\n")

# Step 50: Reusable function for splitting and preprocessing
def preprocess_and_split(df, target='Price', test_size=0.2, rs=42):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    pipe = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X_train_prep = pipe.fit_transform(X_train)
    X_test_prep = pipe.transform(X_test)

    return X_train_prep, X_test_prep, y_train, y_test

print("🔁 Step 50 - Reusable function created successfully")
