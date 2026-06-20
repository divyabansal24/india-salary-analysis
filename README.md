# India Tech Salary & Cost of Living Analysis

An end-to-end data engineering and interactive dashboard project analyzing tech developer salaries, cost of living indices, and purchasing power across 16 major Indian tech hubs and remote roles.

Designed as a modern analytics engineering project utilizing a robust data pipeline stack.

---

## 🛠️ Technology Stack
* **Data Ingestion**: Python (Pandas & SQLAlchemy)
* **Data Warehouse / Database**: PostgreSQL
* **Data Transformation**: DBT (Data Build Tool)
* **Machine Learning**: scikit-learn (Random Forest Salary Predictor & K-Means City Clustering)
* **Interactive Dashboard**: Plotly Dash (Web App UI)

---

## 📂 Project Structure
```text
india-salary-analysis/
├── dashboard/               # Dash Web Application
│   ├── app.py               # Main UI and reactive callbacks
│   └── assets/              # Style assets & static images
├── data/                    # Raw dataset CSVs
├── docs/                    # Documentation
├── india_salary_dbt/        # DBT Project
│   ├── models/              # SQL staging and mart models
│   ├── seeds/               # Cost of living index seeds
│   └── dbt_project.yml      # DBT project configuration
├── src/                     # Python source files
│   └── load_data.py         # DB Ingestion script
├── requirements.txt         # Python dependencies
└── README.md                # Root project guide (this file)
```

---

## 🚀 Getting Started & Setup

### 1. Database Setup
Ensure PostgreSQL is running locally and create a database named `india_salary_db`:
```sql
CREATE DATABASE india_salary_db;
```

### 2. Install Dependencies
Initialize your virtual environment and install the required packages:
```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Load Raw Data
Run the ingestion script to clean columns and push the raw CSV dataset into PostgreSQL:
```bash
python src/load_data.py
```

### 4. Run DBT Models
Run the DBT project to compile the sql staging views and build the analytical data marts in the database:
```bash
# Navigate to dbt directory
cd india_salary_dbt

# Run transformations and seed files
dbt build

# Return to root directory
cd ..
```

### 5. Launch the Dashboard
Run the Plotly Dash server locally:
```bash
python dashboard/app.py
```
Open **`http://127.0.0.1:8050`** in your web browser to view the interactive dashboard.

---

## 💡 Key Features
1. **Geo-Economics Map**: Displays purchasing power (disposable income) across Indian cities. Remote jobs are dynamically mapped to their respective hiring company's headquarters.
2. **K-Means City Clusters**: Automatically groups cities into 4 distinct economic archetypes based on developer salaries, cost of living, and active job openings.
3. **Salary Predictor**: A machine learning model predicting developer salaries based on city, experience level, industry, and work mode.
4. **Skills Premium & Gap Analysis**: Highlights the highest-paying and most in-demand technical credentials.
