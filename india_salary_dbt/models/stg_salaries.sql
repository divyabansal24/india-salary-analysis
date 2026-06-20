SELECT
    Job_ID,
    Job_Title,
    Company,
    Industry,
    City,
    Location_Tier,
    Experience_Level,
    Work_Mode,
    Salary_LPA,
    Company_Rating,
    Date_Posted
FROM {{ source('public', 'raw_salaries') }}
WHERE Salary_LPA IS NOT NULL
AND City IS NOT NULL
