from sqlalchemy import create_engine
import pandas as pd

# Connect to PostgreSQL
engine = create_engine('postgresql://postgres:postgres123@localhost/india_salary_db')

# Load dataset
df = pd.read_csv('data/india_job_market_2024_2026.csv')

# Check it loaded correctly
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Push to PostgreSQL
df.columns = [col.lower() for col in df.columns]
df.to_sql('raw_salaries', engine, if_exists='replace', index=False)
print("Data loaded into PostgreSQL successfully with lowercase columns!")
