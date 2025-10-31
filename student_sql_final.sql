 
CREATE DATABASE StudentDB;
GO

USE StudentDB;
GO

-- ÃœÊ· «·ÿ·«»
CREATE TABLE Student (
    Student_ID INT PRIMARY KEY,
    name NVARCHAR(100) NOT NULL,
    age INT,
    gender NVARCHAR(10),
    grade_level NVARCHAR(20),
    health_condition NVARCHAR(50),
    school_transport NVARCHAR(50),
    free_time_activity NVARCHAR(50),
    parent_education NVARCHAR(50),
    internet_access BIT,
    extracurricular_activities BIT,
    family_size DECIMAL(5, 2),
    tutoring BIT,
    previous_gpa DECIMAL(4, 2),
    admission_year INT,
    city NVARCHAR(50),
    country NVARCHAR(50),
    scholarship_status BIT,
    parent_income INT,
	efficiency DECIMAL(4,2),           
    feedback_rating DECIMAL(4,2) 
	 
	 );

-- ÃœÊ· «·„Ê«œ
CREATE TABLE Subject (
    subject_name NVARCHAR(50) PRIMARY KEY,
    teacher_name NVARCHAR(50),
    difficulty_level NVARCHAR(20),
    hours_per_week INT,
    description NVARCHAR(200),
    teacher_experience_years INT
);

-- ÃœÊ· «·√œ«¡
CREATE TABLE performance (
    performance_id INT IDENTITY(1,1) PRIMARY KEY,
    Student_ID INT,
    subject_name NVARCHAR(50),
    month date,
    sleep_hours DECIMAL(4, 2),
    score DECIMAL(5, 2),
    study_hours DECIMAL(4, 2),
    attendance_rate DECIMAL(5, 2),
    homework_completion_rate DECIMAL(5, 2),
    exam_attempts INT,
    feedback_rating DECIMAL(4, 2),
    efficiency DECIMAL(4, 2),
    performance_level NVARCHAR(20),
    FOREIGN KEY (Student_ID) REFERENCES Student(Student_ID),
    FOREIGN KEY (subject_name) REFERENCES Subject(subject_name)
);
GO


WITH UniqueStudents AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY Student_ID ORDER BY Student_ID) AS rn
    FROM Clean_Student_Performance_Final
)
INSERT INTO Student (
    Student_ID, name, age, gender, grade_level, health_condition, school_transport, free_time_activity,
    parent_education, internet_access, extracurricular_activities, family_size, tutoring, previous_gpa,
    admission_year, city, country, scholarship_status , parent_income, efficiency, feedback_rating
)
SELECT
    Student_ID, name, age, gender, grade_level, health_condition, school_transport, free_time_activity,
    parent_education, internet_access, extracurricular_activities, family_size, tutoring, previous_gpa,
    admission_year, city, country, scholarship_status , parent_income, efficiency, feedback_rating
FROM UniqueStudents
WHERE rn = 1;   


 
 WITH UniqueSubjects AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY subject_name ORDER BY subject_name) AS rn
    FROM Clean_Student_Performance_Final
)
INSERT INTO Subject (
    subject_name, teacher_name, difficulty_level, hours_per_week, description, teacher_experience_years
)
SELECT
    subject_name, teacher_name, difficulty_level, hours_per_week, description, teacher_experience_years
FROM UniqueSubjects
WHERE rn = 1;


 INSERT INTO Performance (
    Student_ID, subject_name, month, sleep_hours, score, study_hours,
    attendance_rate, homework_completion_rate, exam_attempts,
    feedback_rating, efficiency, performance_level
)
SELECT
    Student_ID, subject_name, month, sleep_hours, score, study_hours,
    attendance_rate, homework_completion_rate, exam_attempts,
    feedback_rating, efficiency, performance_level
FROM Clean_Student_Performance_Final;



SELECT  * FROM Student;
SELECT  * FROM Subject;
SELECT  * FROM Performance;
SELECT  * FROM clean_student_performance_final;
--ALTER TABLE Performance
--DROP COLUMN month;
--truncate table clean_student_performance_final; -- use this line only if u want to clear ole db destination in visual studio
--DROP TABLE Clean_Student_Performance_Final;

-- Top performance by subject

WITH MaxScores AS (
    SELECT subject_name, MAX(score) AS max_score
    FROM Performance
    GROUP BY subject_name
)
SELECT 
    p.subject_name,
    s.name AS student_name,
    p.score
FROM Performance p
JOIN Student s ON p.Student_ID = s.Student_ID
JOIN MaxScores m 
    ON p.subject_name = m.subject_name 
   AND p.score = m.max_score
ORDER BY p.subject_name;

--Attendance trends by subject

SELECT 
    subject_name,
    AVG(attendance_rate) AS avg_attendance
FROM Performance
GROUP BY subject_name
ORDER BY avg_attendance DESC;

--Average score by subject

SELECT 
    subject_name,
    AVG(score) AS avg_score
FROM Performance
GROUP BY subject_name
ORDER BY avg_score DESC;

--Top 10 Students by Overall Performance

SELECT TOP 10 
    s.name,
    AVG(p.score) AS avg_score,
    AVG(p.efficiency) AS avg_efficiency,
    AVG(p.attendance_rate) AS avg_attendance
FROM Performance p
JOIN Student s ON p.Student_ID = s.Student_ID
GROUP BY s.name
ORDER BY avg_score DESC;

--Gender-Based Performance Comparison
SELECT 
    s.gender,
    AVG(p.score) AS avg_score,
    AVG(p.attendance_rate) AS avg_attendance,
    AVG(p.efficiency) AS avg_efficiency
FROM Performance p
JOIN Student s ON p.Student_ID = s.Student_ID
GROUP BY s.gender;

 --Most Difficult Subjects (Lowest Average Score)
 SELECT 
    subject_name,
    AVG(score) AS avg_score
FROM Performance
GROUP BY subject_name
ORDER BY avg_score ASC;

--Attendance vs. Score by Subject
SELECT 
    subject_name,
    ROUND(AVG(score), 2) AS avg_score,
    ROUND(AVG(attendance_rate), 2) AS avg_attendance
FROM Performance
GROUP BY subject_name
ORDER BY avg_score DESC;

--Effect of Family Income on Student Performance
SELECT 
    CASE 
        WHEN parent_income < 3000 THEN 'Low Income'
        WHEN parent_income BETWEEN 3000 AND 7000 THEN 'Medium Income'
        ELSE 'High Income'
    END AS income_level,
    ROUND(AVG(p.score), 2) AS avg_score
FROM Student s
JOIN Performance p ON s.Student_ID = p.Student_ID
GROUP BY 
    CASE 
        WHEN parent_income < 3000 THEN 'Low Income'
        WHEN parent_income BETWEEN 3000 AND 7000 THEN 'Medium Income'
        ELSE 'High Income'
    END
ORDER BY avg_score DESC;

--Impact of Private Tutoring on Scores
SELECT 
    s.tutoring,
    ROUND(AVG(p.score), 2) AS avg_score
FROM Student s
JOIN Performance p ON s.Student_ID = p.Student_ID
GROUP BY s.tutoring;

 --Teacher Performance Based on Student Results
SELECT 
    sub.teacher_name,
    ROUND(AVG(p.score), 2) AS avg_student_score,
    ROUND(AVG(p.efficiency), 2) AS avg_efficiency
FROM Performance p
JOIN Subject sub ON p.subject_name = sub.subject_name
GROUP BY sub.teacher_name
ORDER BY avg_student_score DESC;
