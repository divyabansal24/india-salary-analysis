WITH city_salaries AS (
    SELECT
        City,
        Location_Tier,
        COUNT(Job_ID) AS Total_Jobs,
        ROUND(AVG(Salary_LPA)::numeric, 2) AS Avg_Salary_LPA,
        ROUND(AVG(Company_Rating)::numeric, 2) AS Avg_Company_Rating
    FROM {{ ref('stg_salaries') }}
    GROUP BY 1, 2
),
city_col AS (
    SELECT
        City,
        Numbeo_Index,
        Cost_of_Living_Index,
        Estimated_Expenses_LPA
    FROM {{ ref('city_cost_of_living') }}
)
SELECT
    s.City,
    s.Location_Tier,
    s.Total_Jobs,
    s.Avg_Salary_LPA,
    c.Numbeo_Index,
    c.Cost_of_Living_Index,
    c.Estimated_Expenses_LPA,
    -- Purchasing Power Index (Avg Disposable Income) = Avg_Salary_LPA - Estimated_Expenses_LPA
    ROUND((s.Avg_Salary_LPA - c.Estimated_Expenses_LPA)::numeric, 2) AS Avg_Disposable_Income_LPA,
    ROUND((s.Avg_Salary_LPA / c.Cost_of_Living_Index)::numeric, 2) AS Purchasing_Power_Ratio,
    s.Avg_Company_Rating
FROM city_salaries s
LEFT JOIN city_col c ON s.City = c.City

