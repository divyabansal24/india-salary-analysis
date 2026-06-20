SELECT
    Job_ID,
    TRIM(skill) AS Skill
FROM {{ source('public', 'raw_salaries') }},
UNNEST(STRING_TO_ARRAY(Skills_Required, ',')) AS skill
WHERE Skills_Required IS NOT NULL
