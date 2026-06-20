import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, dash_table

# Initialize Dash application
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="India Tech Salaries & Cost of Living",
    suppress_callback_exceptions=True
)
server = app.server

# 1. Coordinate mapping for map display
CITY_COORDS = {
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
    'Delhi': {'lat': 28.7041, 'lon': 77.1025},
    'Pune': {'lat': 18.5204, 'lon': 73.8567},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
    'Chandigarh': {'lat': 30.7333, 'lon': 76.7794},
    'Indore': {'lat': 22.7196, 'lon': 75.8577},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873},
    'Lucknow': {'lat': 26.8467, 'lon': 80.9462},
    'Coimbatore': {'lat': 11.0168, 'lon': 76.9558},
    'Nagpur': {'lat': 21.1458, 'lon': 79.0882},
    'Bhubaneswar': {'lat': 20.2961, 'lon': 85.8245},
    'Kochi': {'lat': 9.9312, 'lon': 76.2673},
    'Remote': {'lat': 22.9734, 'lon': 78.6569}
}

# 2. Database Connection and Data Loader
def load_data():
    try:
        db_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres123@localhost/india_salary_db')
        engine = create_engine(db_url)
        
        # Load tables
        df_salaries = pd.read_sql("SELECT * FROM stg_salaries", engine)
        df_city = pd.read_sql("SELECT * FROM mart_city_clusters", engine)
        
        skills_query = """
            SELECT
                s.skill,
                sa.salary_lpa,
                sa.city,
                sa.experience_level,
                sa.work_mode,
                sa.job_title
            FROM stg_skills s
            JOIN stg_salaries sa ON s.job_id = sa.job_id
        """
        df_skills = pd.read_sql(skills_query, engine)
        print("Data successfully loaded from PostgreSQL database.")
        return df_salaries, df_city, df_skills
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}. Falling back to CSV files...")
        # Fallback to local files
        df_raw = pd.read_csv('data/india_job_market_2024_2026.csv')
        df_raw.columns = [c.lower() for c in df_raw.columns]
        
        df_col = pd.read_csv('india_salary_dbt/seeds/city_cost_of_living.csv')
        df_col.columns = [c.lower() for c in df_col.columns]
        
        df_salaries = df_raw[df_raw['salary_lpa'].notna() & df_raw['city'].notna()].copy()
        
        # Compute economics fallback
        df_city_sal = df_salaries.groupby(['city', 'location_tier']).agg(
            total_jobs=('job_id', 'count'),
            avg_salary_lpa=('salary_lpa', 'mean'),
            avg_company_rating=('company_rating', 'mean')
        ).reset_index()
        
        df_city = pd.merge(df_city_sal, df_col, on='city', how='left')
        df_city['avg_disposable_income_lpa'] = df_city['avg_salary_lpa'] - df_city['estimated_expenses_lpa']
        df_city['purchasing_power_ratio'] = df_city['avg_salary_lpa'] / df_city['cost_of_living_index']
        
        # Simple clustering fallback
        df_city['cluster_id'] = np.where(df_city['avg_disposable_income_lpa'] > df_city['avg_disposable_income_lpa'].median(), 0, 1)
        df_city['archetype'] = np.where(df_city['avg_disposable_income_lpa'] > df_city['avg_disposable_income_lpa'].median(), "High-Yield Hidden Gems", "Cost-of-Living Traps")
        
        # Skills fallback
        df_skills_list = []
        for idx, row in df_salaries.iterrows():
            skills = str(row['skills_required']).split(',')
            for s in skills:
                df_skills_list.append({
                    'skill': s.strip(),
                    'salary_lpa': row['salary_lpa'],
                    'city': row['city'],
                    'experience_level': row['experience_level'],
                    'work_mode': row['work_mode'],
                    'job_title': row['job_title']
                })
        df_skills = pd.DataFrame(df_skills_list)
        return df_salaries, df_city, df_skills

df_salaries, df_city, df_skills = load_data()

# 2.5 Train Salary Predictor Model
from sklearn.ensemble import RandomForestRegressor
print("Training Random Forest Salary Predictor...")
df_ml = df_salaries[['city', 'experience_level', 'industry', 'work_mode', 'salary_lpa']].copy()
X_ml = pd.get_dummies(df_ml[['city', 'experience_level', 'industry', 'work_mode']], drop_first=False)
y_ml = df_ml['salary_lpa']
rf_predictor = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_predictor.fit(X_ml, y_ml)
ml_columns = X_ml.columns.tolist()
print("Salary Predictor model trained successfully.")

# 3. Design System & Layout Constants
BG_COLOR = "#f1f5f9"  # Slate 100 (light blue-gray page background)
CARD_BG = "#ffffff"  # Pure White
TEXT_COLOR = "#0f172a"  # Slate 900 (dark charcoal text)
SUBTEXT_COLOR = "#64748b"  # Slate 500 (muted grey subtext)
ACCENT_BLUE = "#0a4c80"  # Corporate Blue (matching PwC reference style)
ACCENT_GREEN = "#4caf50"  # Corporate Green (matching PwC reference style)
BORDER_COLOR = "#e2e8f0"  # Slate 200 (light border)
SHADOW_STYLE = "0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03)"

TAB_STYLE = {
    "backgroundColor": "#ffffff", 
    "color": SUBTEXT_COLOR, 
    "border": f"1px solid {BORDER_COLOR}", 
    "fontWeight": "600",
    "padding": "6px 10px",
    "fontSize": "12px",
    "height": "auto",
    "minHeight": "40px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "flex": "1 1 auto"
}
TAB_SELECTED_STYLE = {
    "backgroundColor": ACCENT_BLUE, 
    "color": "#ffffff", 
    "border": f"1px solid {ACCENT_BLUE}", 
    "fontWeight": "700",
    "padding": "6px 10px",
    "fontSize": "12px",
    "height": "auto",
    "minHeight": "40px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "flex": "1 1 auto"
}

# 4. Dashboard Layout Construction
app.layout = html.Div(
    style={
        "backgroundColor": BG_COLOR,
        "color": TEXT_COLOR,
        "fontFamily": "'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        "minHeight": "100vh",
        "padding": "20px",
        "boxSizing": "border-box"
    },
    children=[
        # Header Section
        html.Div(
            style={
                "background": "#0a4c80",  # Corporate Blue Banner
                "borderRadius": "12px",
                "padding": "24px 32px",
                "marginBottom": "24px",
                "border": "1px solid #083c66",
                "boxShadow": SHADOW_STYLE,
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "15px",
                "color": "#ffffff"
            },
            children=[
                html.Div([
                    html.H1("India Tech Job Market", style={"margin": "0 0 6px 0", "fontSize": "28px", "fontWeight": "700", "color": "#ffffff"}),
                    html.Div("Purchasing Power vs. Cost of Living Analysis", style={"color": "#93c5fd", "fontSize": "14px", "fontWeight": "500"})
                ])
            ]
        ),

        # Metrics KPI Grid
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                "gap": "20px",
                "marginBottom": "24px"
            },
            children=[
                # KPI 1
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "20px 20px 20px 24px", "border": f"1px solid {BORDER_COLOR}", "borderLeft": "6px solid #0a4c80", "boxShadow": SHADOW_STYLE},
                    children=[
                        html.Div("Total Jobs Analyzed", style={"color": SUBTEXT_COLOR, "fontSize": "13px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1px"}),
                        html.Div(f"{len(df_salaries):,}", id="kpi-total-jobs", style={"fontSize": "32px", "fontWeight": "700", "margin": "8px 0 4px 0", "color": TEXT_COLOR}),
                        html.Div("Across 16 Cities + Remote roles", style={"color": ACCENT_BLUE, "fontSize": "12px", "fontWeight": "500"})
                    ]
                ),
                # KPI 2
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "20px 20px 20px 24px", "border": f"1px solid {BORDER_COLOR}", "borderLeft": "6px solid #4caf50", "boxShadow": SHADOW_STYLE},
                    children=[
                        html.Div("National Avg Tech Salary", style={"color": SUBTEXT_COLOR, "fontSize": "13px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1px"}),
                        html.Div(f"{df_salaries['salary_lpa'].mean():.2f} LPA", id="kpi-avg-salary", style={"fontSize": "32px", "fontWeight": "700", "margin": "8px 0 4px 0", "color": TEXT_COLOR}),
                        html.Div("Minimum threshold: Salary is not NULL", style={"color": ACCENT_BLUE, "fontSize": "12px", "fontWeight": "500"})
                    ]
                ),
                # KPI 3
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "20px 20px 20px 24px", "border": f"1px solid {BORDER_COLOR}", "borderLeft": "6px solid #0a4c80", "boxShadow": SHADOW_STYLE},
                    children=[
                        html.Div("Top Savings City (Hidden Gem)", style={"color": SUBTEXT_COLOR, "fontSize": "13px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1px"}),
                        html.Div(
                            f"{df_city.sort_values(by='avg_disposable_income_lpa', ascending=False).iloc[0]['city']}",
                            id="kpi-top-savings-city",
                            style={"fontSize": "32px", "fontWeight": "700", "margin": "8px 0 4px 0", "color": ACCENT_GREEN}
                        ),
                        html.Div(
                            f"Disposable Income: {df_city.sort_values(by='avg_disposable_income_lpa', ascending=False).iloc[0]['avg_disposable_income_lpa']:.2f} LPA",
                            id="kpi-top-savings-detail",
                            style={"color": SUBTEXT_COLOR, "fontSize": "12px", "fontWeight": "500"}
                        )
                    ]
                )
            ]
        ),

        # Main Interactive Panel (Sidebar + Graph Area)
        html.Div(
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "24px",
                "alignItems": "start"
            },
            children=[
                # Sidebar Filters
                html.Div(
                    style={
                        "background": CARD_BG,
                        "borderRadius": "12px",
                        "padding": "24px",
                        "border": f"1px solid {BORDER_COLOR}",
                        "boxShadow": SHADOW_STYLE,
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "20px",
                        "flex": "0 0 300px",
                        "minWidth": "280px"
                    },
                    children=[
                        html.H3("Interactive Filters", style={"margin": "0 0 10px 0", "fontSize": "18px", "fontWeight": "600", "borderBottom": f"1px solid {BORDER_COLOR}", "paddingBottom": "10px"}),
                        
                        # Filter 1: Job Title
                        html.Div([
                            html.Label("Job Title Focus", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                            dcc.Dropdown(
                                id="job-title-dropdown",
                                options=[{"label": title, "value": title} for title in sorted(df_salaries['job_title'].unique())],
                                multi=True,
                                placeholder="Select job titles (All by default)",
                                style={"color": "#000000"}
                            )
                        ]),
                        
                        # Filter 2: Experience Level
                        html.Div([
                            html.Label("Experience Level", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                            dcc.Dropdown(
                                id="exp-level-dropdown",
                                options=[{"label": exp, "value": exp} for exp in sorted(df_salaries['experience_level'].unique())],
                                multi=True,
                                placeholder="Select experience (All by default)",
                                style={"color": "#000000"}
                            )
                        ]),

                        # Filter 3: Work Mode
                        html.Div([
                            html.Label("Work Mode", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                            dcc.Dropdown(
                                id="work-mode-dropdown",
                                options=[{"label": mode, "value": mode} for mode in sorted(df_salaries['work_mode'].unique())],
                                multi=True,
                                placeholder="Select mode (All by default)",
                                style={"color": "#000000"}
                            )
                        ]),
                        
                        html.Div(
                            style={"marginTop": "10px", "fontSize": "12px", "color": SUBTEXT_COLOR, "lineHeight": "1.5"},
                            children=[
                                html.P("💡 Kolkata and Bhubaneswar are identified as strong 'Hidden Gems' due to high average software developer salaries combined with low cost of living indices."),
                                html.P("⚠️ Mumbai and Delhi function as 'Value Traps' since high housing/rental cost of living offsets higher salary offers.")
                            ]
                        )
                    ]
                ),

                # Graphs & Insights Tabs
                html.Div(
                    style={"flex": "1 1 600px", "minWidth": "300px"},
                    children=[
                        dcc.Tabs(
                            id="tabs-menu",
                            value="tab-geo",
                            colors={"border": BORDER_COLOR, "primary": ACCENT_BLUE, "background": "#f1f5f9"},
                            style={"height": "auto", "display": "flex", "flexWrap": "wrap"},
                            children=[
                                dcc.Tab(label="Geo-Economics", value="tab-geo", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                                dcc.Tab(label="ML Archetypes", value="tab-ml", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                                dcc.Tab(label="Skills Premium", value="tab-skills", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                                dcc.Tab(label="Salary Predictor", value="tab-predictor", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                                dcc.Tab(label="City Recommender", value="tab-recommender", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                                dcc.Tab(label="Market Trends", value="tab-trends", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                                dcc.Tab(label="Skills Gap", value="tab-skills-gap", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
                            ]
                        ),
                        html.Div(id="tab-content", style={"paddingTop": "24px"})
                    ]
                )
            ]
        )
    ]
)

# 5. Tab Renderer Callback
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs-menu", "value")]
)
def render_tab_content(tab_name):
    if tab_name == "tab-geo":
        return html.Div(
            style={"display": "flex", "flexDirection": "column", "gap": "24px"},
            children=[
                # Map Card
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE},
                    children=[
                        html.H4("Purchasing Power Index across Indian Tech Hubs", style={"margin": "0 0 16px 0", "fontSize": "18px", "fontWeight": "600"}),
                        dcc.Graph(id="india-mapbox", style={"height": "500px"})
                    ]
                ),
                # Table & Trend Row
                html.Div(
                    style={"display": "flex", "flexWrap": "wrap", "gap": "24px"},
                    children=[
                        # Table Card
                        html.Div(
                            style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 500px", "minWidth": "280px"},
                            children=[
                                html.H4("City Economics Index Data", style={"margin": "0 0 16px 0", "fontSize": "18px", "fontWeight": "600"}),
                                html.Div(id="city-table-container")
                            ]
                        ),
                        # Trend Line Card
                        html.Div(
                            style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 450px", "minWidth": "280px"},
                            children=[
                                html.H4("Salary Trends Over Time", style={"margin": "0 0 8px 0", "fontSize": "18px", "fontWeight": "600"}),
                                html.Div("Monthly average salary trends colored by work mode", style={"color": SUBTEXT_COLOR, "fontSize": "12px", "marginBottom": "16px"}),
                                dcc.Graph(id="trend-line-chart", style={"height": "400px"})
                            ]
                        )
                    ]
                )
            ]
        )
    elif tab_name == "tab-ml":
        return html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "24px"},
            children=[
                # K-Means Cluster Graph
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 400px", "minWidth": "280px"},
                    children=[
                        html.H4("K-Means City Archetypes (K=4)", style={"margin": "0 0 8px 0", "fontSize": "18px", "fontWeight": "600"}),
                        html.Div("Clustered on salary, expenses, openings, and company rating", style={"color": SUBTEXT_COLOR, "fontSize": "12px", "marginBottom": "16px"}),
                        dcc.Graph(id="kmeans-scatter", style={"height": "400px"})
                    ]
                ),
                # SHAP Feature Explanations
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 400px", "minWidth": "280px"},
                    children=[
                        html.H4("SHAP Explainability Summary", style={"margin": "0 0 8px 0", "fontSize": "18px", "fontWeight": "600"}),
                        html.Div("Feature impact explaining why a city falls into its specific archetype", style={"color": SUBTEXT_COLOR, "fontSize": "12px", "marginBottom": "16px"}),
                        html.Div(
                            style={"display": "flex", "justifyContent": "center", "alignItems": "center", "height": "400px", "overflow": "hidden", "borderRadius": "8px", "backgroundColor": "#ffffff", "border": f"1px solid {BORDER_COLOR}"},
                            children=[
                                html.Img(
                                    src="/assets/shap_summary.png" if os.path.exists("dashboard/assets/shap_summary.png") else "https://raw.githubusercontent.com/shap/shap/master/docs/art/shap_header.png", 
                                    style={"maxWidth": "100%", "maxHeight": "100%", "objectFit": "contain"}
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    elif tab_name == "tab-skills":
        return html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "24px"},
            children=[
                # Top Paying Skills
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 400px", "minWidth": "280px"},
                    children=[
                        html.H4("Highest Paying Skills in India (LPA)", style={"margin": "0 0 16px 0", "fontSize": "18px", "fontWeight": "600"}),
                        dcc.Graph(id="skills-pay-chart", style={"height": "450px"})
                    ]
                ),
                # Top In-Demand Skills (Treemap!)
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 400px", "minWidth": "280px"},
                    children=[
                        html.H4("Most In-Demand Tech Skills (Job Volume)", style={"margin": "0 0 16px 0", "fontSize": "18px", "fontWeight": "600"}),
                        dcc.Graph(id="skills-demand-chart", style={"height": "450px"})
                    ]
                )
            ]
        )
    elif tab_name == "tab-predictor":
        return html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "24px"},
            children=[
                # Inputs Card
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 400px", "minWidth": "280px"},
                    children=[
                        html.H4("Select Parameters", style={"margin": "0 0 16px 0", "fontSize": "18px", "fontWeight": "600"}),
                        
                        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "20px"}, children=[
                            # City Dropdown
                            html.Div([
                                html.Label("City / Location", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Dropdown(
                                    id="pred-city",
                                    options=[{"label": city, "value": city} for city in sorted(df_salaries['city'].unique())],
                                    value="Bangalore",
                                    clearable=False,
                                    style={"color": "#000000"}
                                )
                            ]),
                            
                            # Experience Level Dropdown
                            html.Div([
                                html.Label("Experience Level", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Dropdown(
                                    id="pred-exp",
                                    options=[{"label": exp, "value": exp} for exp in sorted(df_salaries['experience_level'].unique())],
                                    value="Mid (3-6 yrs)",
                                    clearable=False,
                                    style={"color": "#000000"}
                                )
                            ]),
                            
                            # Industry Dropdown
                            html.Div([
                                html.Label("Industry Sector", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Dropdown(
                                    id="pred-industry",
                                    options=[{"label": ind, "value": ind} for ind in sorted(df_salaries['industry'].unique())],
                                    value="Information Technology",
                                    clearable=False,
                                    style={"color": "#000000"}
                                )
                            ]),
                            
                            # Work Mode Dropdown
                            html.Div([
                                html.Label("Work Mode", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Dropdown(
                                    id="pred-mode",
                                    options=[{"label": mode, "value": mode} for mode in sorted(df_salaries['work_mode'].unique())],
                                    value="Hybrid",
                                    clearable=False,
                                    style={"color": "#000000"}
                                )
                            ])
                        ])
                    ]
                ),
                
                # Output Predictor Result Card
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "32px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 400px", "minWidth": "280px", "display": "flex", "flexDirection": "column", "justifyContent": "center", "alignItems": "center", "textAlign": "center"},
                    children=[
                        html.Div("Estimated Market Salary", style={"color": SUBTEXT_COLOR, "fontSize": "14px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "1.5px"}),
                        html.H1(id="pred-result-salary", style={"fontSize": "56px", "fontWeight": "800", "margin": "16px 0 8px 0", "color": ACCENT_BLUE}),
                        html.Div(id="pred-result-comparison", style={"fontSize": "14px", "fontWeight": "600", "marginBottom": "32px"}),
                        
                        html.Div(
                            style={"borderTop": f"1px solid {BORDER_COLOR}", "paddingTop": "24px", "width": "100%", "textAlign": "left"},
                            children=[
                                html.H5("Estimated Purchasing Power Breakdown:", style={"margin": "0 0 12px 0", "fontSize": "14px", "fontWeight": "700", "color": TEXT_COLOR}),
                                html.Div(id="pred-result-savings", style={"fontSize": "13px", "color": SUBTEXT_COLOR, "lineHeight": "1.6"})
                            ]
                        )
                    ]
                )
            ]
        )
    elif tab_name == "tab-recommender":
        return html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "24px", "alignItems": "start"},
            children=[
                # Inputs Card
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 300px", "minWidth": "280px"},
                    children=[
                        html.H4("Recommendation Criteria", style={"margin": "0 0 16px 0", "fontSize": "18px", "fontWeight": "600"}),
                        
                        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "20px"}, children=[
                            # Salary Expectation Input / Slider
                            html.Div([
                                html.Label("Salary Expectation (LPA)", id="salary-slider-label", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Slider(
                                    id="rec-salary",
                                    min=5,
                                    max=35,
                                    step=1,
                                    value=15,
                                    marks={i: f"{i}L" for i in range(5, 36, 5)},
                                )
                            ]),
                            
                            # Industry Dropdown
                            html.Div([
                                html.Label("Your Industry", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Dropdown(
                                    id="rec-industry",
                                    options=[{"label": ind, "value": ind} for ind in sorted(df_salaries['industry'].unique())],
                                    value="Information Technology",
                                    clearable=False,
                                    style={"color": "#000000"}
                                )
                            ]),
                            
                            # Work Mode Dropdown
                            html.Div([
                                html.Label("Preferred Work Mode", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Dropdown(
                                    id="rec-mode",
                                    options=[{"label": mode, "value": mode} for mode in sorted(df_salaries['work_mode'].unique())],
                                    value="Hybrid",
                                    clearable=False,
                                    style={"color": "#000000"}
                                )
                            ])
                        ])
                    ]
                ),
                
                # Recommendations Output Card
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 500px", "minWidth": "280px"},
                    children=[
                        html.H4("Top 3 Recommended Cities", style={"margin": "0 0 8px 0", "fontSize": "18px", "fontWeight": "600"}),
                        html.Div("Scored dynamically by expected disposable income (savings) based on your target salary, industry, and work mode.", style={"color": SUBTEXT_COLOR, "fontSize": "12px", "marginBottom": "24px"}),
                        
                        html.Div(id="recommender-results", style={"display": "flex", "flexDirection": "column", "gap": "20px"})
                    ]
                )
            ]
        )
    elif tab_name == "tab-trends":
        return html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "24px", "alignItems": "start"},
            children=[
                # Inputs Card
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "0 0 350px", "minWidth": "280px"},
                    children=[
                        html.H4("Hiring Trend Criteria", style={"margin": "0 0 16px 0", "fontSize": "18px", "fontWeight": "600"}),
                        
                        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "20px"}, children=[
                            # Cities Multi-Select Dropdown
                            html.Div([
                                html.Label("Select Cities", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Dropdown(
                                    id="trend-cities-dropdown",
                                    options=[{"label": city, "value": city} for city in sorted(df_salaries['city'].unique()) if city != 'Remote'],
                                    value=["Bangalore", "Mumbai", "Pune"],
                                    multi=True,
                                    placeholder="Select cities...",
                                    style={"color": "#000000"}
                                )
                            ]),
                            
                            # Industry Multi-Select Dropdown
                            html.Div([
                                html.Label("Select Industries", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Dropdown(
                                    id="trend-industries-dropdown",
                                    options=[{"label": ind, "value": ind} for ind in sorted(df_salaries['industry'].unique())],
                                    value=["Information Technology", "Banking & Finance", "Consulting"],
                                    multi=True,
                                    placeholder="Select industries...",
                                    style={"color": "#000000"}
                                )
                            ]),

                            # Split Lines By Radio Button
                            html.Div([
                                html.Label("Split Graph Lines By", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.RadioItems(
                                    id="trend-split-radio",
                                    options=[
                                        {"label": " City Location", "value": "city"},
                                        {"label": " Industry Segment", "value": "industry"}
                                    ],
                                    value="city",
                                    labelStyle={"display": "block", "marginBottom": "8px", "fontSize": "13px"}
                                )
                            ]),
                            
                            # Time Aggregation Radio Button
                            html.Div([
                                html.Label("Time Grouping", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.RadioItems(
                                    id="trend-grouping-radio",
                                    options=[
                                        {"label": " Monthly View", "value": "M"},
                                        {"label": " Quarterly View", "value": "Q"}
                                    ],
                                    value="M",
                                    labelStyle={"display": "block", "marginBottom": "8px", "fontSize": "13px"}
                                )
                            ])
                        ])
                    ]
                ),
                
                # Recommendations Output Card
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 600px", "minWidth": "280px"},
                    children=[
                        html.H4("Hiring Volume & Salary Offer Trends", style={"margin": "0 0 8px 0", "fontSize": "18px", "fontWeight": "600"}),
                        html.Div("Compare recruitment volume and average market salary trends side-by-side.", style={"color": SUBTEXT_COLOR, "fontSize": "12px", "marginBottom": "24px"}),
                        
                        html.Div(
                            style={"display": "flex", "flexDirection": "column", "gap": "32px"},
                            children=[
                                html.Div([
                                    html.H5("Hiring Volume Trend (Job Openings Count)", style={"fontSize": "14px", "fontWeight": "700", "marginBottom": "12px", "color": TEXT_COLOR}),
                                    dcc.Graph(id="trend-volume-chart", style={"height": "350px"})
                                ]),
                                html.Div([
                                    html.H5("Average Salary Offer Trend (LPA)", style={"fontSize": "14px", "fontWeight": "700", "marginBottom": "12px", "color": TEXT_COLOR}),
                                    dcc.Graph(id="trend-salary-chart", style={"height": "350px"})
                                ])
                            ]
                        )
                    ]
                )
            ]
        )
    elif tab_name == "tab-skills-gap":
        default_title = df_skills['job_title'].value_counts().index[0] if not df_skills.empty else "Software Engineer"
        
        return html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "24px", "alignItems": "start"},
            children=[
                # Sidebar Controls Card
                html.Div(
                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "0 0 350px", "minWidth": "280px"},
                    children=[
                        html.H4("Select Role Profile", style={"margin": "0 0 16px 0", "fontSize": "18px", "fontWeight": "600"}),
                        
                        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "20px"}, children=[
                            # Job Title Single-Select Dropdown
                            html.Div([
                                html.Label("Job Title", style={"display": "block", "marginBottom": "8px", "fontSize": "12px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                dcc.Dropdown(
                                    id="skills-gap-title-dropdown",
                                    options=[{"label": title, "value": title} for title in sorted(df_skills['job_title'].dropna().unique())],
                                    value=default_title,
                                    clearable=False,
                                    style={"color": "#000000"}
                                )
                            ]),
                            
                            # Skills Insight Card (Description)
                            html.Div(
                                style={"marginTop": "10px", "fontSize": "13px", "color": SUBTEXT_COLOR, "lineHeight": "1.6", "borderTop": f"1px solid {BORDER_COLOR}", "paddingTop": "15px"},
                                children=[
                                    html.H5("💡 Skills Strategy Guide", style={"margin": "0 0 8px 0", "fontSize": "14px", "fontWeight": "700", "color": TEXT_COLOR}),
                                    html.P([
                                        html.Strong("Core Skills (In-Demand): ", style={"color": ACCENT_BLUE}),
                                        "These are the baseline credentials. Focus on these first to get your foot in the door (highest occurrence in job postings)."
                                    ], style={"marginBottom": "10px"}),
                                    html.P([
                                        html.Strong("Niche Skills (Rarest): ", style={"color": "#418ab3"}),
                                        "Emerging or highly specialized skills. Having these makes you stand out in competitive applicant pools."
                                    ], style={"marginBottom": "10px"}),
                                    html.P([
                                        html.Strong("Salary Premium (Lucrative): ", style={"color": ACCENT_GREEN}),
                                        "Skills that command the highest average salary package. Acquiring these is the fastest way to increase your earning power."
                                    ])
                                ]
                            )
                        ])
                    ]
                ),
                
                # Three-Column Charts Layout Container
                html.Div(
                    style={"flex": "1 1 600px", "minWidth": "300px", "display": "flex", "flexDirection": "column", "gap": "24px"},
                    children=[
                        # Two columns layout that wraps on small screens
                        html.Div(
                            style={"display": "flex", "flexWrap": "wrap", "gap": "24px"},
                            children=[
                                # Card 1: Core Skills
                                html.Div(
                                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 350px", "minWidth": "280px"},
                                    children=[
                                        html.H5("Core Required Skills (In-Demand)", style={"margin": "0 0 8px 0", "fontSize": "16px", "fontWeight": "700", "color": TEXT_COLOR}),
                                        html.Div("Percentage of job postings requiring this skill", style={"color": SUBTEXT_COLOR, "fontSize": "12px", "marginBottom": "16px"}),
                                        dcc.Graph(id="skills-core-chart", style={"height": "350px"})
                                    ]
                                ),
                                # Card 2: Niche Skills
                                html.Div(
                                    style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "flex": "1 1 350px", "minWidth": "280px"},
                                    children=[
                                        html.H5("Niche / Specialized Skills (Rare)", style={"margin": "0 0 8px 0", "fontSize": "16px", "fontWeight": "700", "color": TEXT_COLOR}),
                                        html.Div("Rarest skills (count > 1) to avoid entry-level saturation", style={"color": SUBTEXT_COLOR, "fontSize": "12px", "marginBottom": "16px"}),
                                        dcc.Graph(id="skills-niche-chart", style={"height": "350px"})
                                    ]
                                )
                            ]
                        ),
                        # Card 3: Salary Premium Skills
                        html.Div(
                            style={"background": CARD_BG, "borderRadius": "12px", "padding": "24px", "border": f"1px solid {BORDER_COLOR}", "boxShadow": SHADOW_STYLE, "width": "100%", "boxSizing": "border-box"},
                            children=[
                                html.H5("Skills Salary Premium (Lucrative)", style={"margin": "0 0 8px 0", "fontSize": "16px", "fontWeight": "700", "color": TEXT_COLOR}),
                                html.Div("Average market salary offered (LPA) when this skill is specified", style={"color": SUBTEXT_COLOR, "fontSize": "12px", "marginBottom": "16px"}),
                                dcc.Graph(id="skills-premium-chart", style={"height": "350px"})
                            ]
                        )
                    ]
                )
            ]
        )
# 6a. Callback for Geo-Economics Tab & KPIs
@app.callback(
    [Output("india-mapbox", "figure"),
     Output("city-table-container", "children"),
     Output("trend-line-chart", "figure"),
     Output("kpi-total-jobs", "children"),
     Output("kpi-avg-salary", "children"),
     Output("kpi-top-savings-city", "children"),
     Output("kpi-top-savings-detail", "children")],
    [Input("job-title-dropdown", "value"),
     Input("exp-level-dropdown", "value"),
     Input("work-mode-dropdown", "value")]
)
def update_geo_economics_tab(selected_titles, selected_exps, selected_modes):
    filtered_salaries = df_salaries.copy()
    if selected_titles:
        filtered_salaries = filtered_salaries[filtered_salaries['job_title'].isin(selected_titles)]
    if selected_exps:
        filtered_salaries = filtered_salaries[filtered_salaries['experience_level'].isin(selected_exps)]
    if selected_modes:
        filtered_salaries = filtered_salaries[filtered_salaries['work_mode'].isin(selected_modes)]
        
    # Recalculate City Stats
    city_agg = filtered_salaries.groupby(['city', 'location_tier']).agg(
        total_jobs=('job_id', 'count'),
        avg_salary_lpa=('salary_lpa', 'mean'),
        avg_company_rating=('company_rating', 'mean')
    ).reset_index()
    
    # Merge with cost of living data
    df_col_raw = df_city[['city', 'numbeo_index', 'cost_of_living_index', 'estimated_expenses_lpa', 'cluster_id', 'archetype']].drop_duplicates('city')
    city_stats = pd.merge(city_agg, df_col_raw, on='city', how='left')
    
    # Fill NAs
    city_stats['numbeo_index'] = city_stats['numbeo_index'].fillna(20.0)
    city_stats['cost_of_living_index'] = city_stats['cost_of_living_index'].fillna(1.0)
    city_stats['estimated_expenses_lpa'] = city_stats['estimated_expenses_lpa'].fillna(4.0)
    city_stats['avg_disposable_income_lpa'] = city_stats['avg_salary_lpa'] - city_stats['estimated_expenses_lpa']
    city_stats['purchasing_power_ratio'] = city_stats['avg_salary_lpa'] / city_stats['cost_of_living_index']
    
    # Map Coordinates
    map_data = city_stats.copy()
    map_data['lat'] = map_data['city'].map(lambda x: CITY_COORDS.get(x, {}).get('lat', np.nan))
    map_data['lon'] = map_data['city'].map(lambda x: CITY_COORDS.get(x, {}).get('lon', np.nan))
    map_data = map_data.dropna(subset=['lat', 'lon'])
    
    # Map Figure (Standard geographical scatter map to ensure offline-readiness and avoid WebGL errors)
    fig_map = px.scatter_geo(
        map_data,
        lat="lat",
        lon="lon",
        color="avg_disposable_income_lpa",
        size="total_jobs",
        hover_name="city",
        hover_data={
            "lat": False, "lon": False,
            "avg_salary_lpa": ":.2f LPA",
            "numbeo_index": ":.1f",
            "estimated_expenses_lpa": ":.2f LPA",
            "avg_disposable_income_lpa": ":.2f LPA",
            "total_jobs": True
        },
        color_continuous_scale=["#e0f2fe", "#0a4c80", "#4caf50"], # PwC Corporate Blue to Green
        size_max=25,
        projection="mercator"
    )
    fig_map.update_geos(
        center={"lat": 20.5937, "lon": 78.9629},
        projection_scale=4.5,
        fitbounds="locations" if len(map_data.drop_duplicates(subset=['lat', 'lon'])) > 1 else False,
        visible=True,
        showcountries=True,
        countrycolor="#94a3b8", # slate gray country border
        showland=True, landcolor="#f1f5f9", # light gray land color
        showocean=True, oceancolor="#e0f2fe", # light blue ocean color
        showlakes=True, lakecolor="#cbd5e1" # lake color
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT_COLOR,
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar=dict(
            title="Disposable Income (LPA)",
            thicknessmode="pixels", thickness=15,
            lenmode="fraction", len=0.6,
            x=0.95, y=0.5
        )
    )

    if map_data.empty:
        fig_map.add_annotation(
            text="Remote roles do not have a physical city location.<br>Select 'Hybrid' or 'On-Site' in the filters to view physical locations on the map.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=13, color="#64748b", family="sans-serif"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            borderpad=10
        )

    # Table
    display_table = city_stats[['city', 'location_tier', 'total_jobs', 'avg_salary_lpa', 'numbeo_index', 'estimated_expenses_lpa', 'avg_disposable_income_lpa', 'purchasing_power_ratio']].copy()
    display_table.columns = ['City', 'Tier', 'Jobs Volume', 'Avg Salary (LPA)', 'Numbeo Index (NYC=100)', 'Est. Expenses (LPA)', 'Disposable Income (LPA)', 'PPI Ratio']
    
    # Round values
    display_table = display_table.round(2)
    display_table = display_table.sort_values('Disposable Income (LPA)', ascending=False)
    
    table_view = dash_table.DataTable(
        data=display_table.to_dict('records'),
        columns=[{"name": i, "id": i} for i in display_table.columns],
        style_header={
            'backgroundColor': '#f1f5f9',
            'color': '#0f172a',
            'fontWeight': 'bold',
            'border': '1px solid #e2e8f0'
        },
        style_cell={
            'backgroundColor': '#ffffff',
            'color': '#0f172a',
            'textAlign': 'left',
            'border': '1px solid #f1f5f9',
            'padding': '12px 16px',
            'fontSize': '13px'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Disposable Income (LPA)', 'filter_query': '{Disposable Income (LPA)} >= 16.0'},
                'color': ACCENT_GREEN,
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Disposable Income (LPA)', 'filter_query': '{Disposable Income (LPA)} < 13.5'},
                'color': '#ef4444', # Light Red
                'fontWeight': 'bold'
            }
        ],
        page_size=10,
        style_table={'overflowX': 'auto'}
    )

    # Trend Line Chart
    df_trend_data = filtered_salaries.copy()
    df_trend_data['month'] = pd.to_datetime(df_trend_data['date_posted']).dt.to_period('M').astype(str)
    df_trend = df_trend_data.groupby(['month', 'work_mode']).agg(
        avg_salary=('salary_lpa', 'mean')
    ).reset_index().sort_values('month')
    
    fig_trend = px.line(
        df_trend,
        x="month",
        y="avg_salary",
        color="work_mode",
        markers=True,
        color_discrete_map={
            "Remote": "#4caf50", # Emerald Green
            "Hybrid": "#0a4c80", # Corporate Navy Blue
            "On-Site": "#418ab3"  # Muted Teal/Blue
        },
        labels={"month": "Month", "avg_salary": "Avg Salary (LPA)", "work_mode": "Work Mode"}
    )
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font_color=TEXT_COLOR,
        margin={"t":10,"b":40,"l":40,"r":10},
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_trend.update_traces(line=dict(width=3.5), marker=dict(size=8))

    # Calculate dynamic KPI values
    kpi_jobs = f"{len(filtered_salaries):,}"
    
    if len(filtered_salaries) > 0:
        kpi_salary = f"{filtered_salaries['salary_lpa'].mean():.2f} LPA"
    else:
        kpi_salary = "0.00 LPA"
        
    if len(city_stats) > 0 and city_stats['total_jobs'].sum() > 0:
        top_city_row = city_stats.sort_values(by='avg_disposable_income_lpa', ascending=False).iloc[0]
        kpi_city = top_city_row['city']
        kpi_savings = f"Disposable Income: {top_city_row['avg_disposable_income_lpa']:.2f} LPA"
    else:
        kpi_city = "N/A"
        kpi_savings = "Disposable Income: 0.00 LPA"

    return fig_map, table_view, fig_trend, kpi_jobs, kpi_salary, kpi_city, kpi_savings


# 6b. Callback for ML Archetypes Tab
@app.callback(
    Output("kmeans-scatter", "figure"),
    [Input("job-title-dropdown", "value"),
     Input("exp-level-dropdown", "value"),
     Input("work-mode-dropdown", "value")]
)
def update_ml_archetypes_tab(selected_titles, selected_exps, selected_modes):
    filtered_salaries = df_salaries.copy()
    if selected_titles:
        filtered_salaries = filtered_salaries[filtered_salaries['job_title'].isin(selected_titles)]
    if selected_exps:
        filtered_salaries = filtered_salaries[filtered_salaries['experience_level'].isin(selected_exps)]
    if selected_modes:
        filtered_salaries = filtered_salaries[filtered_salaries['work_mode'].isin(selected_modes)]
        
    city_agg = filtered_salaries.groupby(['city', 'location_tier']).agg(
        total_jobs=('job_id', 'count'),
        avg_salary_lpa=('salary_lpa', 'mean')
    ).reset_index()
    
    df_col_raw = df_city[['city', 'estimated_expenses_lpa', 'archetype']].drop_duplicates('city')
    city_stats = pd.merge(city_agg, df_col_raw, on='city', how='left')
    city_stats['estimated_expenses_lpa'] = city_stats['estimated_expenses_lpa'].fillna(4.0)

    fig_kmeans = px.scatter(
        city_stats,
        x="avg_salary_lpa",
        y="estimated_expenses_lpa",
        color="archetype",
        size="total_jobs",
        hover_name="city",
        color_discrete_map={
            "Metro Wealth Hubs": "#0a4c80",       # Corporate Blue
            "High-Yield Hidden Gems": "#4caf50",  # Corporate Green
            "Cost-of-Living Traps": "#ef4444",    # Coral Red
            "Budget Growth Zones": "#0ea5e9"      # Sky Blue
        },
        labels={
            "avg_salary_lpa": "Average Salary (LPA)",
            "estimated_expenses_lpa": "Estimated Living Expenses (LPA)",
            "archetype": "City Archetype"
        }
    )
    fig_kmeans.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font_color=TEXT_COLOR,
        margin={"t":20,"b":40,"l":40,"r":20},
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig_kmeans


# 6c. Callback for Skills Premium Tab
@app.callback(
    [Output("skills-pay-chart", "figure"),
     Output("skills-demand-chart", "figure")],
    [Input("job-title-dropdown", "value"),
     Input("exp-level-dropdown", "value"),
     Input("work-mode-dropdown", "value")]
)
def update_skills_premium_tab(selected_titles, selected_exps, selected_modes):
    filtered_skills = df_skills.copy()
    if selected_titles:
        filtered_skills = filtered_skills[filtered_skills['job_title'].isin(selected_titles)]
    if selected_exps:
        filtered_skills = filtered_skills[filtered_skills['experience_level'].isin(selected_exps)]
    if selected_modes:
        filtered_skills = filtered_skills[filtered_skills['work_mode'].isin(selected_modes)]
        
    skill_stats = filtered_skills.groupby('skill').agg(
        avg_salary=('salary_lpa', 'mean'),
        job_count=('salary_lpa', 'count')
    ).reset_index()
    
    top_paying_skills = skill_stats[skill_stats['job_count'] >= 5].sort_values(by='avg_salary', ascending=False).head(15)
    if top_paying_skills.empty:
        fig_skills_pay = go.Figure()
        fig_skills_pay.update_layout(
            title="No skills data matching criteria.",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_COLOR
        )
    else:
        fig_skills_pay = px.bar(
            top_paying_skills,
            y="skill",
            x="avg_salary",
            orientation="h",
            color="skill", # Corporate cohesive sequence of blues and greens
            color_discrete_sequence=["#0a4c80", "#1e3b70", "#1d5893", "#418ab3", "#6baed6", "#9ecae1", "#c6dbef", "#4caf50", "#2e7d32", "#66bb6a", "#81c784", "#a5d6a7", "#c8e6c9", "#10b981", "#34d399"],
            labels={"avg_salary": "Average Salary (LPA)", "skill": "Skill"},
            hover_data={"job_count": True}
        )
        fig_skills_pay.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font_color=TEXT_COLOR,
            margin={"t":10,"b":40,"l":120,"r":10},
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(showgrid=False, categoryorder='total ascending')
        )

    top_demand_skills = skill_stats.sort_values(by='job_count', ascending=False).head(15)
    if top_demand_skills.empty:
        fig_skills_demand = go.Figure()
        fig_skills_demand.update_layout(
            title="No skills data matching criteria.",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_COLOR
        )
    else:
        fig_skills_demand = px.treemap(
            top_demand_skills,
            path=[px.Constant("All Skills"), "skill"],
            values="job_count",
            color="job_count",
            color_continuous_scale=["#e0f2fe", "#0a4c80", "#4caf50"], # PwC Corporate Blue to Green Scale
            labels={"job_count": "Job Count", "skill": "Skill"}
        )
        fig_skills_demand.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_COLOR,
            margin={"t":20,"b":20,"l":20,"r":20}
        )
        
    return fig_skills_pay, fig_skills_demand



# 8. Salary Prediction Callback
@app.callback(
    [Output("pred-result-salary", "children"),
     Output("pred-result-comparison", "children"),
     Output("pred-result-savings", "children")],
    [Input("pred-city", "value"),
     Input("pred-exp", "value"),
     Input("pred-industry", "value"),
     Input("pred-mode", "value"),
     Input("job-title-dropdown", "value"),
     Input("exp-level-dropdown", "value"),
     Input("work-mode-dropdown", "value")]
)
def predict_salary(city, exp, industry, mode, selected_titles, selected_exps, selected_modes):
    input_df = pd.DataFrame(0, index=[0], columns=ml_columns)
    
    col_city = f"city_{city}"
    col_exp = f"experience_level_{exp}"
    col_ind = f"industry_{industry}"
    col_mode = f"work_mode_{mode}"
    
    if col_city in input_df.columns:
        input_df[col_city] = 1
    if col_exp in input_df.columns:
        input_df[col_exp] = 1
    if col_ind in input_df.columns:
        input_df[col_ind] = 1
    if col_mode in input_df.columns:
        input_df[col_mode] = 1
        
    pred_val = rf_predictor.predict(input_df)[0]
    
    filtered_salaries = df_salaries.copy()
    if selected_titles:
        filtered_salaries = filtered_salaries[filtered_salaries['job_title'].isin(selected_titles)]
    if selected_exps:
        filtered_salaries = filtered_salaries[filtered_salaries['experience_level'].isin(selected_exps)]
    if selected_modes:
        filtered_salaries = filtered_salaries[filtered_salaries['work_mode'].isin(selected_modes)]
        
    national_avg = filtered_salaries['salary_lpa'].mean() if len(filtered_salaries) > 0 else df_salaries['salary_lpa'].mean()
    diff_pct = ((pred_val - national_avg) / national_avg) * 100
    
    if diff_pct >= 0:
        comp_text = f"▲ {diff_pct:.1f}% above national average ({national_avg:.2f} LPA)"
        comp_color = ACCENT_GREEN
    else:
        comp_text = f"▼ {abs(diff_pct):.1f}% below national average ({national_avg:.2f} LPA)"
        comp_color = "#ef4444"
        
    comp_element = html.Span(comp_text, style={"color": comp_color})
    
    city_row = df_city[df_city['city'] == city]
    if len(city_row) > 0:
        expenses = city_row.iloc[0]['estimated_expenses_lpa']
        col_index = city_row.iloc[0]['numbeo_index']
    else:
        expenses = 4.0
        col_index = 20.0
        
    disposable = pred_val - expenses
    
    savings_element = html.Div([
        html.P(f"• Predicted Salary: {pred_val:.2f} LPA"),
        html.P(f"• Estimated Living Cost: {expenses:.2f} LPA (Numbeo Cost Index: {col_index:.1f})"),
        html.P(style={"fontWeight": "bold", "color": ACCENT_GREEN if disposable > 0 else "#ef4444", "marginTop": "8px"},
               children=f"• Est. Annual Savings / Disposable Income: {disposable:.2f} LPA")
    ])
    
    return f"{pred_val:.2f} LPA", comp_element, savings_element


# 9. City Recommendation Callback
@app.callback(
    Output("recommender-results", "children"),
    [Input("rec-salary", "value"),
     Input("rec-industry", "value"),
     Input("rec-mode", "value"),
     Input("job-title-dropdown", "value"),
     Input("exp-level-dropdown", "value"),
     Input("work-mode-dropdown", "value")]
)
def update_recommendations(expected_salary, industry, mode, selected_titles, selected_exps, selected_modes):
    # Filter salaries dataset based on global filters
    filtered_salaries = df_salaries.copy()
    if selected_titles:
        filtered_salaries = filtered_salaries[filtered_salaries['job_title'].isin(selected_titles)]
    if selected_exps:
        filtered_salaries = filtered_salaries[filtered_salaries['experience_level'].isin(selected_exps)]
    if selected_modes:
        filtered_salaries = filtered_salaries[filtered_salaries['work_mode'].isin(selected_modes)]
        
    if filtered_salaries.empty:
        return html.Div(
            "No matching data found for selected global filters. Clear some filters to view recommendations.",
            style={"textAlign": "center", "color": "#ef4444", "padding": "20px", "fontWeight": "600"}
        )
        
    # All physical cities in dataset (excluding Remote)
    physical_cities = [c for c in filtered_salaries['city'].unique() if c != 'Remote']
    
    # We will build a single DataFrame to predict in batch for efficiency
    predict_rows = []
    exp_levels = filtered_salaries['experience_level'].unique()
    
    for city in physical_cities:
        pred_city = 'Remote' if mode == 'Remote' else city
        for exp in exp_levels:
            predict_rows.append({
                'city': pred_city,
                'experience_level': exp,
                'industry': industry,
                'work_mode': mode,
                'target_city': city
            })
            
    pred_df = pd.DataFrame(predict_rows)
    pred_encoded = pd.get_dummies(pred_df[['city', 'experience_level', 'industry', 'work_mode']], drop_first=False)
    pred_encoded = pred_encoded.reindex(columns=ml_columns, fill_value=0)
    pred_df['pred_salary'] = rf_predictor.predict(pred_encoded)
    
    # Group by target_city and get mean predicted salary
    city_salaries = pred_df.groupby('target_city')['pred_salary'].mean().to_dict()
    
    # Let's count jobs per city for this industry and work mode
    job_counts = filtered_salaries[
        (filtered_salaries['industry'] == industry) & 
        (filtered_salaries['work_mode'] == mode)
    ].groupby('city').size().to_dict()
    
    # Aggregate city archetype info
    df_col_raw = df_city[['city', 'estimated_expenses_lpa', 'numbeo_index', 'location_tier', 'archetype']].drop_duplicates('city')
    city_info = df_col_raw.set_index('city').to_dict('index')

    results = []
    for city in physical_cities:
        # Predicted market salary
        pred_sal = city_salaries.get(city, filtered_salaries['salary_lpa'].mean())
        
        # Get city expenses & info
        c_info = city_info.get(city, {})
        expenses = c_info.get('estimated_expenses_lpa', 4.0)
        numbeo = c_info.get('numbeo_index', 20.0)
        tier = c_info.get('location_tier', 'Tier 2')
        archetype = c_info.get('archetype', 'Budget Growth Zones')
            
        # 1. Salary Compatibility Score (0 - 100)
        if pred_sal >= expected_salary:
            salary_score = 100.0
        else:
            salary_score = max(0.0, (pred_sal / expected_salary) * 100.0)
            
        # 2. Expected Savings
        expected_savings = pred_sal - expenses
        
        # 3. Job Openings count
        jobs_count = job_counts.get(city, 0)
        
        results.append({
            'city': city,
            'tier': tier,
            'archetype': archetype,
            'predicted_salary': pred_sal,
            'expenses': expenses,
            'numbeo_index': numbeo,
            'expected_savings': expected_savings,
            'jobs_count': jobs_count,
            'salary_score': salary_score
        })
        
    df_res = pd.DataFrame(results)
    
    # Normalizing Savings Score (0 - 100)
    min_sav = df_res['expected_savings'].min()
    max_sav = df_res['expected_savings'].max()
    if max_sav != min_sav:
        df_res['savings_score'] = ((df_res['expected_savings'] - min_sav) / (max_sav - min_sav)) * 100.0
    else:
        df_res['savings_score'] = 100.0
        
    # Normalizing Job Market Score (0 - 100)
    df_res['log_jobs'] = np.log1p(df_res['jobs_count'])
    min_jobs = df_res['log_jobs'].min()
    max_jobs = df_res['log_jobs'].max()
    if max_jobs != min_jobs:
        df_res['jobs_score'] = ((df_res['log_jobs'] - min_jobs) / (max_jobs - min_jobs)) * 100.0
    else:
        df_res['jobs_score'] = 100.0
        
    # Calculate Final Match Score
    if mode == 'Remote':
        df_res['match_score'] = (0.3 * df_res['salary_score']) + (0.7 * df_res['savings_score'])
    else:
        df_res['match_score'] = (0.4 * df_res['salary_score']) + (0.4 * df_res['savings_score']) + (0.2 * df_res['jobs_score'])
        
    df_res = df_res.sort_values(by='match_score', ascending=False)
    top_cities = df_res.head(3)
    
    if len(top_cities) == 0:
        return html.Div(
            "No recommendations could be generated. Try adjusting your preferences.",
            style={"textAlign": "center", "color": "#ef4444", "padding": "20px", "fontWeight": "600"}
        )
        
    badges = [
        {"title": "🏆 #1 Best Choice", "border_color": "1px solid #f59e0b", "badge_bg": "rgba(245, 158, 11, 0.1)", "badge_color": "#d97706"},
        {"title": "🥈 #2 Great Value", "border_color": "1px solid #94a3b8", "badge_bg": "rgba(148, 163, 184, 0.1)", "badge_color": "#475569"},
        {"title": "🥉 #3 Strong Alternative", "border_color": "1px solid #b45309", "badge_bg": "rgba(180, 83, 9, 0.1)", "badge_color": "#b45309"}
    ]
    
    card_elements = []
    for idx, (_, row) in enumerate(top_cities.iterrows()):
        badge = badges[idx]
        city = row['city']
        tier = row['tier']
        archetype = row['archetype']
        pred_sal = row['predicted_salary']
        expenses = row['expenses']
        numbeo = row['numbeo_index']
        expected_savings = row['expected_savings']
        jobs_count = int(row['jobs_count'])
        match_score = row['match_score']
        
        # Color schemes for archetypes
        archetype_colors = {
            "Metro Wealth Hubs": {"bg": "rgba(245, 158, 11, 0.1)", "color": "#d97706", "border": "1px solid rgba(245, 158, 11, 0.3)"},
            "High-Yield Hidden Gems": {"bg": "rgba(16, 185, 129, 0.1)", "color": "#059669", "border": "1px solid rgba(16, 185, 129, 0.3)"},
            "Cost-of-Living Traps": {"bg": "rgba(239, 68, 68, 0.1)", "color": "#dc2626", "border": "1px solid rgba(239, 68, 68, 0.3)"},
            "Budget Growth Zones": {"bg": "rgba(59, 130, 246, 0.1)", "color": "#2563eb", "border": "1px solid rgba(59, 130, 246, 0.3)"}
        }
        
        arch_style = archetype_colors.get(archetype, {"bg": "rgba(100, 116, 139, 0.1)", "color": "#475569", "border": "1px solid rgba(100, 116, 139, 0.3)"})
        
        # Dynamic Verdict Generation
        salary_fit_status = "fully meets or exceeds" if pred_sal >= expected_salary else "is slightly below"
        if pred_sal < expected_salary * 0.8:
            salary_fit_status = "is lower than"
            
        mode_text = "hybrid" if mode == "Hybrid" else ("remote" if mode == "Remote" else "on-site")
        
        if mode == "Remote":
            verdict = (
                f"Living in {city} ({archetype}) while working remotely is an excellent choice. "
                f"Since your role is remote, you earn a high rate of {pred_sal:.2f} LPA. "
                f"By living in {city}, where annual living costs are low ({expenses:.2f} LPA), you maximize your "
                f"disposable income to {expected_savings:.2f} LPA."
            )
        else:
            verdict = (
                f"{city} is a '{archetype}'. The predicted market salary of {pred_sal:.2f} LPA "
                f"{salary_fit_status} your target of {expected_salary:.2f} LPA for {mode_text} roles. "
                f"With annual living expenses estimated at {expenses:.2f} LPA, it offers a strong savings potential of "
                f"{expected_savings:.2f} LPA, supported by {jobs_count} active job openings in this segment."
            )
            
        card_elements.append(
            html.Div(
                style={
                    "border": f"1px solid {BORDER_COLOR}",
                    "borderRadius": "16px",
                    "padding": "24px",
                    "marginBottom": "20px",
                    "background": CARD_BG,
                    "boxShadow": SHADOW_STYLE,
                    "display": "flex",
                    "flexWrap": "wrap",
                    "justifyContent": "space-between",
                    "gap": "20px"
                },
                children=[
                    # Left Column: Badges, Title, Verdict
                    html.Div(
                        style={"flex": "1 1 350px", "minWidth": "280px"},
                        children=[
                            # Badges Line
                            html.Div(
                                style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "12px"},
                                children=[
                                    # Rank Badge
                                    html.Span(
                                        badge['title'],
                                        style={
                                            "backgroundColor": badge['badge_bg'],
                                            "color": badge['badge_color'],
                                            "border": badge['border_color'],
                                            "padding": "4px 10px",
                                            "borderRadius": "20px",
                                            "fontSize": "11px",
                                            "fontWeight": "700"
                                        }
                                    ),
                                    # Tier Badge
                                    html.Span(
                                        tier,
                                        style={
                                            "backgroundColor": "#f1f5f9",
                                            "color": "#475569",
                                            "border": f"1px solid {BORDER_COLOR}",
                                            "padding": "4px 10px",
                                            "borderRadius": "20px",
                                            "fontSize": "11px",
                                            "fontWeight": "600"
                                        }
                                    ),
                                    # Archetype Badge
                                    html.Span(
                                        archetype,
                                        style={
                                            "backgroundColor": arch_style['bg'],
                                            "color": arch_style['color'],
                                            "border": arch_style['border'],
                                            "padding": "4px 10px",
                                            "borderRadius": "20px",
                                            "fontSize": "11px",
                                            "fontWeight": "600"
                                        }
                                    )
                                ]
                            ),
                            # City Name
                            html.H3(
                                f"{city}", 
                                style={"margin": "0 0 10px 0", "fontSize": "26px", "fontWeight": "800", "color": TEXT_COLOR}
                            ),
                            # Verdict Text
                            html.P(
                                verdict,
                                style={"margin": "0", "fontSize": "13px", "color": SUBTEXT_COLOR, "lineHeight": "1.6"}
                            )
                        ]
                    ),
                    
                    # Right Column: Score & Metric Box
                    html.Div(
                        style={
                            "flex": "0 0 320px",
                            "minWidth": "280px",
                            "background": "#f8fafc",
                            "padding": "20px",
                            "borderRadius": "12px",
                            "border": f"1px solid {BORDER_COLOR}",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "14px"
                        },
                        children=[
                            # Score & Pill
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                children=[
                                    html.Span("Match Score", style={"fontSize": "12px", "fontWeight": "700", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                    html.Span(
                                        f"{match_score:.1f}%", 
                                        style={
                                            "fontSize": "16px", 
                                            "fontWeight": "800", 
                                            "color": ACCENT_BLUE,
                                            "backgroundColor": "rgba(79, 70, 229, 0.1)",
                                            "padding": "4px 10px",
                                            "borderRadius": "8px"
                                        }
                                    )
                                ]
                            ),
                            # Custom Progress Bar
                            html.Div(
                                style={
                                    "width": "100%",
                                    "height": "6px",
                                    "backgroundColor": "#e2e8f0",
                                    "borderRadius": "3px",
                                    "overflow": "hidden"
                                },
                                children=[
                                    html.Div(
                                        style={
                                            "width": f"{match_score}%",
                                            "height": "100%",
                                            "backgroundColor": ACCENT_BLUE,
                                            "borderRadius": "3px"
                                        }
                                    )
                                ]
                            ),
                            # Metrics Grid
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1fr 1fr",
                                    "gap": "12px",
                                    "marginTop": "4px"
                                },
                                children=[
                                    # Expected Savings
                                    html.Div([
                                        html.Div("Expected Savings", style={"fontSize": "10px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                        html.Div(f"{expected_savings:.2f} LPA", style={"fontSize": "16px", "fontWeight": "800", "color": ACCENT_GREEN})
                                    ]),
                                    # Predicted Salary
                                    html.Div([
                                        html.Div("Market Salary", style={"fontSize": "10px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                        html.Div(f"{pred_sal:.2f} LPA", style={"fontSize": "15px", "fontWeight": "700", "color": TEXT_COLOR})
                                    ]),
                                    # Cost of Living
                                    html.Div([
                                        html.Div("Cost of Living", style={"fontSize": "10px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                        html.Div(f"{expenses:.2f} LPA", style={"fontSize": "14px", "fontWeight": "600", "color": TEXT_COLOR})
                                    ]),
                                    # Active Openings
                                    html.Div([
                                        html.Div("Active Openings", style={"fontSize": "10px", "fontWeight": "600", "color": SUBTEXT_COLOR, "textTransform": "uppercase"}),
                                        html.Div(f"{jobs_count} openings", style={"fontSize": "14px", "fontWeight": "600", "color": TEXT_COLOR})
                                    ])
                                ]
                            )
                        ]
                    )
                ]
            )
        )
        
    return card_elements


# 10. Job Market Trend Analysis Callback
@app.callback(
    [Output("trend-volume-chart", "figure"),
     Output("trend-salary-chart", "figure")],
    [Input("trend-cities-dropdown", "value"),
     Input("trend-industries-dropdown", "value"),
     Input("trend-split-radio", "value"),
     Input("trend-grouping-radio", "value"),
     Input("job-title-dropdown", "value"),
     Input("exp-level-dropdown", "value"),
     Input("work-mode-dropdown", "value")]
)
def update_trend_analysis_charts(selected_cities, selected_industries, split_by, grouping, selected_titles, selected_exps, selected_modes):
    # Fallback to defaults if empty
    if not selected_cities:
        selected_cities = []
    if not selected_industries:
        selected_industries = []
        
    # Copy dataset
    df_trend_data = df_salaries.copy()
    
    # Filter based on global sidebar filters
    if selected_titles:
        df_trend_data = df_trend_data[df_trend_data['job_title'].isin(selected_titles)]
    if selected_exps:
        df_trend_data = df_trend_data[df_trend_data['experience_level'].isin(selected_exps)]
    if selected_modes:
        df_trend_data = df_trend_data[df_trend_data['work_mode'].isin(selected_modes)]
    
    # Parse dates and group
    df_trend_data['date_parsed'] = pd.to_datetime(df_trend_data['date_posted'])
    
    if grouping == "Q":
        df_trend_data['period'] = df_trend_data['date_parsed'].dt.to_period('Q').astype(str)
    else:
        df_trend_data['period'] = df_trend_data['date_parsed'].dt.to_period('M').astype(str)
        
    # Filter
    df_filtered = df_trend_data[
        (df_trend_data['city'].isin(selected_cities)) &
        (df_trend_data['industry'].isin(selected_industries))
    ]
    
    if len(df_filtered) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No matching data found for selected filters.",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_COLOR
        )
        return empty_fig, empty_fig
        
    # Group by period and split_by
    df_grouped = df_filtered.groupby(['period', split_by]).agg(
        hiring_volume=('job_id', 'count'),
        avg_salary=('salary_lpa', 'mean')
    ).reset_index().sort_values('period')
    
    # Render Volume Chart
    fig_vol = px.line(
        df_grouped,
        x="period",
        y="hiring_volume",
        color=split_by,
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Prism if split_by == 'industry' else ["#0a4c80", "#4caf50", "#418ab3", "#6baed6", "#9ecae1", "#c6dbef", "#2e7d32", "#66bb6a", "#81c784"],
        labels={"period": "Time Period", "hiring_volume": "Job Openings Count", split_by: split_by.capitalize()}
    )
    fig_vol.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font_color=TEXT_COLOR,
        margin={"t":10,"b":40,"l":40,"r":10},
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_vol.update_traces(line=dict(width=3.5), marker=dict(size=8))
    
    # Render Salary Chart
    fig_sal = px.line(
        df_grouped,
        x="period",
        y="avg_salary",
        color=split_by,
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Prism if split_by == 'industry' else ["#0a4c80", "#4caf50", "#418ab3", "#6baed6", "#9ecae1", "#c6dbef", "#2e7d32", "#66bb6a", "#81c784"],
        labels={"period": "Time Period", "avg_salary": "Avg Salary Offered (LPA)", split_by: split_by.capitalize()}
    )
    fig_sal.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font_color=TEXT_COLOR,
        margin={"t":10,"b":40,"l":40,"r":10},
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_sal.update_traces(line=dict(width=3.5), marker=dict(size=8))
    
    return fig_vol, fig_sal


# 11. Skills Gap Callback
@app.callback(
    [Output("skills-core-chart", "figure"),
     Output("skills-niche-chart", "figure"),
     Output("skills-premium-chart", "figure")],
    [Input("skills-gap-title-dropdown", "value"),
     Input("job-title-dropdown", "value"),
     Input("exp-level-dropdown", "value"),
     Input("work-mode-dropdown", "value")]
)
def update_skills_gap_analysis(selected_title, selected_titles, selected_exps, selected_modes):
    # Filter global datasets first
    filtered_skills = df_skills.copy()
    filtered_salaries = df_salaries.copy()
    
    if selected_exps:
        filtered_skills = filtered_skills[filtered_skills['experience_level'].isin(selected_exps)]
        filtered_salaries = filtered_salaries[filtered_salaries['experience_level'].isin(selected_exps)]
    if selected_modes:
        filtered_skills = filtered_skills[filtered_skills['work_mode'].isin(selected_modes)]
        filtered_salaries = filtered_salaries[filtered_salaries['work_mode'].isin(selected_modes)]
        
    # 1. Filter skills dataset
    df_sub = filtered_skills[filtered_skills['job_title'] == selected_title].copy()
    
    total_jobs = len(filtered_salaries[filtered_salaries['job_title'] == selected_title])
    if total_jobs == 0:
        total_jobs = 1
        
    # Check if empty
    if df_sub.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No skills data available for this job title.",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_COLOR
        )
        return empty_fig, empty_fig, empty_fig
        
    skill_counts = df_sub['skill'].value_counts()
    
    # --- Core Skills ---
    df_core = (skill_counts / total_jobs * 100.0).reset_index()
    df_core.columns = ['skill', 'frequency']
    df_core['skill'] = df_core['skill'].str.strip()
    df_core = df_core.head(10).sort_values(by='frequency', ascending=True)
    
    fig_core = px.bar(
        df_core,
        x="frequency",
        y="skill",
        orientation="h",
        labels={"frequency": "Frequency (%)", "skill": "Skill"}
    )
    fig_core.update_traces(
        marker_color="#0a4c80",
        hovertemplate="<b>%{y}</b><br>Required in: %{x:.1f}% of postings<extra></extra>"
    )
    fig_core.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font_color=TEXT_COLOR,
        margin={"t":10,"b":40,"l":120,"r":10},
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", ticksuffix="%"),
        yaxis=dict(showgrid=False)
    )
    
    # --- Niche Skills ---
    niche_counts = skill_counts[skill_counts > 1]
    if niche_counts.empty:
        niche_counts = skill_counts
    
    df_niche = (niche_counts / total_jobs * 100.0).reset_index()
    df_niche.columns = ['skill', 'frequency']
    df_niche['skill'] = df_niche['skill'].str.strip()
    df_niche = df_niche.tail(10).sort_values(by='frequency', ascending=True)
    
    fig_niche = px.bar(
        df_niche,
        x="frequency",
        y="skill",
        orientation="h",
        labels={"frequency": "Frequency (%)", "skill": "Skill"}
    )
    fig_niche.update_traces(
        marker_color="#418ab3",
        hovertemplate="<b>%{y}</b><br>Required in: %{x:.1f}% of postings<extra></extra>"
    )
    fig_niche.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font_color=TEXT_COLOR,
        margin={"t":10,"b":40,"l":120,"r":10},
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", ticksuffix="%"),
        yaxis=dict(showgrid=False)
    )
    
    # --- Premium Skills ---
    significant_skills = skill_counts[skill_counts >= 3].index
    if len(significant_skills) < 5:
        significant_skills = skill_counts[skill_counts >= 1].index
        
    df_sig = df_sub[df_sub['skill'].isin(significant_skills)]
    df_premium = df_sig.groupby('skill')['salary_lpa'].mean().reset_index()
    df_premium.columns = ['skill', 'avg_salary']
    df_premium['skill'] = df_premium['skill'].str.strip()
    df_premium = df_premium.sort_values(by='avg_salary', ascending=False).head(10)
    df_premium = df_premium.sort_values(by='avg_salary', ascending=True)
    
    fig_premium = px.bar(
        df_premium,
        x="avg_salary",
        y="skill",
        orientation="h",
        labels={"avg_salary": "Avg Salary (LPA)", "skill": "Skill"}
    )
    fig_premium.update_traces(
        marker_color="#4caf50",
        hovertemplate="<b>%{y}</b><br>Avg Salary: %{x:.2f} LPA<extra></extra>"
    )
    fig_premium.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font_color=TEXT_COLOR,
        margin={"t":10,"b":40,"l":120,"r":10},
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", ticksuffix=" LPA"),
        yaxis=dict(showgrid=False)
    )
    
    return fig_core, fig_niche, fig_premium


# 7. Run App Server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
