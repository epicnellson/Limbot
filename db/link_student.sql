-- Links the owner's WhatsApp number to the seeded sample student so the deployed bot resolves
-- it on the first message.
--
-- Run AFTER db/seed.sql against the same database. docker compose applies the three initdb
-- files in lexical order (01-schema, 02-seed, 03-link), so a fresh stack is linked out of the
-- box. Against an existing database:
--
--     psql "$POSTGRES_DSN" -f db/schema.sql
--     psql "$POSTGRES_DSN" -f db/seed.sql
--     psql "$POSTGRES_DSN" -f db/link_student.sql
--
-- The number is what Meta sends in the webhook "from"/"contacts[].wa_id" field: international
-- digits without '+', spaces or leading zeros. This is the test sender's wa_id seen in the
-- live logs. Override the default with:
--
--     psql "$POSTGRES_DSN" -v phone=23288436147 -f db/link_student.sql
--
-- Idempotent: re-running never duplicates rows.
\set phone 23279826564

-- 1. Repoint the seeded sample student (Ada Lovelace, the richest record: a weekly timetable,
--    upcoming assignment deadlines and a graded submission) to the target number. Because
--    enrollments, submissions, exams and grades are keyed by student_id, not by number, the
--    whole record moves intact and nothing is re-inserted.
UPDATE students
   SET wa_id = :'phone'
 WHERE full_name = 'Ada Lovelace'
   AND wa_id <> :'phone';

-- 2. If no Ada row exists yet (schema.sql was applied but seed.sql was not, or the seed was
--    edited), create the sample profile under the target number so the link never silently
--    no-ops.
INSERT INTO students (wa_id, full_name, email, programme, year_of_study, gpa)
SELECT :'phone', 'Ada Lovelace', 'ada@example.edu', 'BSc Computer Science', 2, 3.72
 WHERE NOT EXISTS (
     SELECT 1 FROM students WHERE full_name = 'Ada Lovelace' OR wa_id = :'phone'
 );

-- 3. Show the handle the bot will resolve and what data it carries. If the counts come back
--    zero, re-run db/seed.sql first: the timetable, deadlines, grades and exams are created
--    there, keyed to the courses this student is enrolled in.
SELECT
    s.id AS student_id,
    s.wa_id,
    s.full_name,
    (SELECT COUNT(*) FROM enrollments e JOIN courses c ON c.id = e.course_id
      WHERE e.student_id = s.id) AS courses_enrolled,
    (SELECT COUNT(DISTINCT t.course_id)
       FROM timetable_entries t
       JOIN enrollments e ON e.course_id = t.course_id
      WHERE e.student_id = s.id) AS timetable_courses,
    (SELECT COUNT(*)
       FROM assignments a
       JOIN enrollments e ON e.course_id = a.course_id
      WHERE e.student_id = s.id AND a.due_at >= NOW()) AS upcoming_deadlines,
    (SELECT COUNT(*)
       FROM submissions g
      WHERE g.student_id = s.id AND g.status = 'graded' AND g.grade IS NOT NULL) AS graded_work
  FROM students s
 WHERE s.wa_id = :'phone' AND s.full_name = 'Ada Lovelace';