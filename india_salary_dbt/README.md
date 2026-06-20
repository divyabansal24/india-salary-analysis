# India Tech Job Market — Salary vs Cost of Living Analysis

A data analytics project that answers: which Indian cities actually give you the best life after paying your bills?

Live Dashboard: https://india-salary-analysis-production.up.railway.app

---

## The Problem

Salaries in Mumbai and Bangalore look impressive on paper. But after rent, groceries, and transport — how much do you actually keep? This project analyses 5,000 real job postings across 16 Indian cities to find out.

---

## Key Findings

- Kolkata ranks as the top hidden gem city with the highest disposable income at 19.98 LPA
- Mumbai and Delhi are value traps — high salaries are consumed by cost of living
- Vector DB and ML skills command a 35% salary premium over average
- Teaching roles show negative purchasing power in 14 of 16 cities analysed
- National average tech salary stands at 19.83 LPA

---

## What the Dashboard Includes

- Geo-Economics: City-wise Purchasing Power Index across India
- ML Archetypes: K-Means clustering of cities into 4 economic categories
- Skills Premium: Highest paying vs most in-demand skills
- Salary Predictor: Predict expected salary by city, experience, and industry
- City Recommender: Get top 3 best cities based on your preferences
- Market Trends: Hiring trends over time by city and industry
- Skills Gap: See which skills are missing for any job title

---

## Tech Stack

- Language: Python 3.13
- Database: PostgreSQL 16
- Data Transformation: dbt Core 1.12
- Analysis: pandas, numpy, seaborn, matplotlib
- Machine Learning: Scikit-learn, SHAP
- Dashboard: Plotly Dash 4.2
- Deployment: Railway

---

## Project Structure

```
india-salary-analysis/
├── dashboard/
│   └── app.py                    # Plotly Dash application
├── data/
│   └── india_job_market_2024_2026.csv
├── india_salary_dbt/
│   └── models/
│       ├── stg_salaries.sql
│       ├── stg_skills.sql
│       └── mart_city_economics.sql
├── src/
│   ├── load_data.py              # Data ingestion pipeline
│   └── analyze_and_cluster.py   # ML clustering and SHAP
├── Procfile
└── requirements.txt
```

---

## How to Run Locally

1. Clone the repository

```
git clone https://github.com/divyabansal24/india-salary-analysis.git
cd india-salary-analysis
```

2. Create and activate virtual environment

```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Set up PostgreSQL and update the connection string in src/load_data.py

5. Load data into database

```
python src/load_data.py
```

6. Run dbt transformations

```
cd india_salary_dbt
dbt run
```

7. Launch the dashboard

```
python dashboard/app.py
```

Open http://localhost:8050 in your browser.

---

## Dataset

Source: India Job Market Dataset 2024-2026 from Kaggle
Size: 5,000 job postings across 16 cities
Features: Job title, company, city, salary, skills, experience level, work mode, industry

---

## Author

Divya Bansal
GitHub: https://github.com/divyabansal24
