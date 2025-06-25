#CSV Data Analysis : Import a CSV file containing quarterly sales data using pandas, then display and examine the first 5 rows to understand the data structure.
import pandas as pd
df = pd.read_csv('/content/Region,Quarter,Product,Sales.csv')
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())


#DataFrame Calculation : Create a new column in an existing DataFrame that calculates a 5% sales tax based on values in the 'Revenue' column, using appropriate pandas methods.
df = pd.read_csv('/content/Region,Quarter,Product,Sales.csv')
df['Sales_Tax'] = df['Revenue'] * 0.05
df['Total_With_Tax'] = df['Revenue'] + df['Sales_Tax']
print(df.head())


#Conditional Data Filtering : Using boolean indexing, filter a DataFrame to display only employees with annual salaries below $60,000, and verify the result with appropriate methods.
df = pd.read_csv('/content/Name,Department,Annual_Salary.csv')
filtered_df = df[df['Annual_Salary'] < 60000]
print("Employees with Salary below $60,000:\n")
print(filtered_df)
print("\nHighest salary in filtered data:", filtered_df['Annual_Salary'].max())
