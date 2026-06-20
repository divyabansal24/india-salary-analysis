import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Starting Chapter 3 & 4: Analysis, K-Means Clustering & SHAP...")

    # 1. Connect to PostgreSQL
    db_url = 'postgresql://postgres:postgres123@localhost/india_salary_db'
    engine = create_engine(db_url)

    # 2. Extract Data
    df_city = pd.read_sql("SELECT * FROM mart_city_economics", engine)
    
    # Query for skills analysis
    skills_query = """
        SELECT
            s.skill,
            COUNT(s.job_id) as job_count,
            ROUND(AVG(sa.salary_lpa)::numeric, 2) as avg_salary_lpa
        FROM stg_skills s
        JOIN stg_salaries sa ON s.job_id = sa.job_id
        GROUP BY 1
        ORDER BY avg_salary_lpa DESC
    """
    df_skills = pd.read_sql(skills_query, engine)

    print(f"Successfully loaded {len(df_city)} cities and {len(df_skills)} skills from database.")

    # 3. Chapter 3: Insights Generation
    print("\n--- Chapter 3: Insights ---")
    
    # Identify Hidden Gems (Sorted by disposable income)
    hidden_gems = df_city.sort_values(by='avg_disposable_income_lpa', ascending=False).head(5)
    print("\nTop 5 Hidden Gems (Highest Disposable Income / Savings Potential):")
    for idx, row in hidden_gems.iterrows():
        print(f" - {row['city']}: Salary = {row['avg_salary_lpa']} LPA, Expenses = {row['estimated_expenses_lpa']} LPA, Savings = {row['avg_disposable_income_lpa']} LPA")

    # Identify Value Traps (High Cost of Living Index but Lower Savings/Disposable Income)
    value_traps = df_city.sort_values(by='avg_disposable_income_lpa', ascending=True).head(5)
    print("\nTop 5 Value Traps (Lowest Disposable Income / High Relative Cost):")
    for idx, row in value_traps.iterrows():
        print(f" - {row['city']}: Salary = {row['avg_salary_lpa']} LPA, Expenses = {row['estimated_expenses_lpa']} LPA, Savings = {row['avg_disposable_income_lpa']} LPA")

    # Top paying skills
    top_skills = df_skills.sort_values(by='avg_salary_lpa', ascending=False).head(5)
    print("\nTop 5 Highest Paying Skills:")
    for idx, row in top_skills.iterrows():
        print(f" - {row['skill']}: Average Salary = {row['avg_salary_lpa']} LPA ({row['job_count']} jobs)")

    # 4. Chapter 4: K-Means Clustering on Cities
    print("\n--- Chapter 4: K-Means Clustering ---")
    
    # Features for clustering
    features = ['avg_salary_lpa', 'estimated_expenses_lpa', 'total_jobs', 'avg_company_rating']
    X = df_city[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run K-Means with K=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_city['cluster_id'] = kmeans.fit_predict(X_scaled)
    
    # Assign human-readable archetype names based on cluster centroids
    centroids = kmeans.cluster_centers_
    # We will compute the mean salary and cost of living for each cluster
    cluster_means = df_city.groupby('cluster_id')[['avg_salary_lpa', 'estimated_expenses_lpa']].mean()
    print("\nCluster Centroid Means:")
    print(cluster_means)

    # Let's map cluster labels to names based on their relative ranks in Salary and Expenses
    archetype_mapping = {}
    
    # Logic to classify clusters:
    # 1. High Salary, High Cost -> Metro Wealth Hubs
    # 2. High Salary, Low Cost -> High-Yield Hidden Gems
    # 3. Low/Mid Salary, High Cost -> Cost-of-Living Traps
    # 4. Low Salary, Low Cost -> Budget Growth Zones
    
    sorted_by_salary = cluster_means.sort_values(by='avg_salary_lpa', ascending=False)
    highest_salary_cluster = sorted_by_salary.index[0]
    second_highest_salary_cluster = sorted_by_salary.index[1]
    third_highest_salary_cluster = sorted_by_salary.index[2]
    lowest_salary_cluster = sorted_by_salary.index[3]
    
    # Map them based on expenses
    for cid in range(4):
        sal = cluster_means.loc[cid, 'avg_salary_lpa']
        exp = cluster_means.loc[cid, 'estimated_expenses_lpa']
        
        # Simple heuristic
        if sal >= cluster_means['avg_salary_lpa'].median():
            if exp >= cluster_means['estimated_expenses_lpa'].median():
                archetype_mapping[cid] = "Metro Wealth Hubs"
            else:
                archetype_mapping[cid] = "High-Yield Hidden Gems"
        else:
            if exp >= cluster_means['estimated_expenses_lpa'].median():
                archetype_mapping[cid] = "Cost-of-Living Traps"
            else:
                archetype_mapping[cid] = "Budget Growth Zones"
                
    # Re-verify uniqueness of mappings (just in case they overlap, fallback to unique keys)
    if len(set(archetype_mapping.values())) < 4:
        # Fallback to simple rank mapping
        archetype_mapping = {
            highest_salary_cluster: "Metro Wealth Hubs",
            second_highest_salary_cluster: "High-Yield Hidden Gems",
            third_highest_salary_cluster: "Cost-of-Living Traps",
            lowest_salary_cluster: "Budget Growth Zones"
        }

    df_city['archetype'] = df_city['cluster_id'].map(archetype_mapping)
    print("\nCity Archetype Assignments:")
    for idx, row in df_city.sort_values(by='archetype').iterrows():
        print(f" - {row['city']}: {row['archetype']} (Salary: {row['avg_salary_lpa']} LPA, Expenses: {row['estimated_expenses_lpa']} LPA)")

    # Save clusters back to Postgres
    df_city.to_sql("mart_city_clusters", engine, if_exists="replace", index=False)
    print("\nSaved clustering results to database table 'mart_city_clusters'.")

    # 5. SHAP Explainability
    print("\n--- SHAP Explainability ---")
    
    # We train a Random Forest Classifier to predict the archetype label using the features
    y = df_city['cluster_id']
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X, y)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    # Ensure assets directory exists
    os.makedirs("dashboard/assets", exist_ok=True)
    
    # Create the SHAP summary plot
    plt.figure(figsize=(8, 6))
    
    # SHAP returns a list of arrays for multiclass classification
    # We can plot the summary plot for the multi-class model
    # Convert archetype names for label display
    class_names = [archetype_mapping[c] for c in sorted(archetype_mapping.keys())]
    
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=features, 
        class_names=class_names,
        show=False
    )
    plt.title("SHAP Feature Importance for City Archetype Clustering", fontsize=12, pad=15)
    plt.tight_layout()
    
    # Save plot
    plot_path = "dashboard/assets/shap_summary.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to '{plot_path}'")
    print("Analysis and Clustering completed successfully!")

if __name__ == "__main__":
    main()
