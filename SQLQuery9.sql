CREATE DATABASE StudentDW;
GO
USE StudentDW;
GO

CREATE TABLE DimStudent1 (
    [student_id] tinyint,
    [name] nvarchar(50),
    [age] tinyint,
    [gender] nvarchar(50),
    [grade_level] nvarchar(50),
    [subject_name] nvarchar(50),
    [teacher_name] nvarchar(50),
    [difficulty_level] nvarchar(50),
    [hours_per_week] tinyint,
    [description] nvarchar(50),
    [month] nvarchar(10),
    [score] float,
    [sleep_hours] float,
    [study_hours] float,
    [attendance_rate] tinyint,
    [internet_access] bit,
    [free_time_activity] nvarchar(50),
    [parent_education] nvarchar(50),
    [homework_completion_rate] tinyint,
    [extracurricular_activities] bit,
    [family_size] tinyint,
    [previous_gpa] float,
    [school_transport] nvarchar(50),
    [health_condition] nvarchar(50),
    [tutoring] bit,
    [admission_year] smallint,
    [city] nvarchar(50),
    [country] nvarchar(50),
    [scholarship_status] bit,
    [exam_attempts] tinyint,
    [teacher_experience_years] tinyint,
    [parent_income] smallint,
    [feedback_rating] float,
    [efficiency] float,
    [performance_level] nvarchar(50)
)

CREATE TABLE DimSubject (
    SubjectKey INT IDENTITY(1,1) PRIMARY KEY,
    Subject_Name NVARCHAR(50),
    Teacher_Name NVARCHAR(50),
    Difficulty_Level NVARCHAR(20)
);



CREATE TABLE factss1 (
    [student_id] tinyint,
    [name] nvarchar(50),
    [age] tinyint,
    [gender] nvarchar(50),
    [grade_level] nvarchar(50),
    [subject_name] nvarchar(50),
    [teacher_name] nvarchar(50),
    [difficulty_level] nvarchar(50),
    [hours_per_week] tinyint,
    [description] nvarchar(50),
    [month] nvarchar(10),
    [score] float,
    [sleep_hours] float,
    [study_hours] float,
    [attendance_rate] tinyint,
    [internet_access] bit,
    [free_time_activity] nvarchar(50),
    [parent_education] nvarchar(50),
    [homework_completion_rate] tinyint,
    [extracurricular_activities] bit,
    [family_size] tinyint,
    [previous_gpa] float,
    [school_transport] nvarchar(50),
    [health_condition] nvarchar(50),
    [tutoring] bit,
    [admission_year] smallint,
    [city] nvarchar(50),
    [country] nvarchar(50),
    [scholarship_status] bit,
    [exam_attempts] tinyint,
    [teacher_experience_years] tinyint,
    [parent_income] smallint,
    [feedback_rating] float,
    [efficiency] float,
    [performance_level] nvarchar(50)
)


CREATE TABLE DimTime (
    TimeKey INT IDENTITY(1,1) PRIMARY KEY,
    MonthName NVARCHAR(20),
    Year INT
);

SELECT * FROM DimStudent1;
SELECT * FROM DimSubject;
SELECT * FROM factss1;




SELECT COUNT(*) FROM DimStudent1;
SELECT COUNT(*) FROM DimSubject;
SELECT COUNT(*) FROM factss1;

ALTER TABLE DimStudent1
ADD CONSTRAINT PK_DimStudent1 PRIMARY KEY (student_id);

ALTER TABLE DimSubject
ADD CONSTRAINT PK_DimSubject PRIMARY KEY (SubjectKey);





ALTER TABLE factss1
ADD CONSTRAINT FK_factss1_DimStudent1
FOREIGN KEY (student_id)
REFERENCES DimStudent1(student_id);

ALTER TABLE factss1
ADD CONSTRAINT FK_factss1_DimSubjectName
FOREIGN KEY (subject_name)
REFERENCES DimSubject(Subject_Name);


SELECT 
    f.student_id,
    ds.name AS student_name,
    ds.grade_level,
    s.Subject_Name,
    f.score,
    f.month
FROM factss1 f
JOIN DimStudent1 ds ON f.student_id = ds.student_id
JOIN DimSubject s ON f.subject_name = s.Subject_Name;


---تحليل بيانات كل طالب 
SELECT 
    f.student_id,
    s.name AS Student_Name,
    AVG(f.score) AS Avg_Score,
    COUNT(DISTINCT f.subject_name) AS Subjects_Count
FROM factss1 f
JOIN DimStudent1 s 
    ON f.student_id = s.student_id
GROUP BY f.student_id, s.name
ORDER BY Avg_Score DESC;
---
--متوسط الدرجات حسب المادة
SELECT 
    f.subject_name,
    AVG(f.score) AS Average_Score
FROM factss1 f
GROUP BY f.subject_name
ORDER BY Average_Score DESC;
-----
--تحليل العلاقة بين المذاكرة والدرجات
SELECT 
    f.student_id,
    s.name AS Student_Name,
    f.study_hours,
    f.score
FROM factss1 f
JOIN DimStudent1 s ON f.student_id = s.student_id
ORDER BY f.study_hours DESC;
-----
-- مقارنة عدد الطلاب حسب المدينة
SELECT 
    s.city,
    COUNT(DISTINCT s.student_id) AS Total_Students
FROM DimStudent1 s
GROUP BY s.city
ORDER BY Total_Students DESC;
