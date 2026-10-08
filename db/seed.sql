-- Demo data for local development. Replaces anything previously seeded.
-- wa_id values are placeholder numbers: swap them for the real WhatsApp numbers you want the
-- bot to recognise, otherwise every tool call answers "not linked".

TRUNCATE students, courses, timetable_entries, assignments, submissions, exams,
    exam_registrations, enrollments RESTART IDENTITY CASCADE;

INSERT INTO students (wa_id, full_name, email, programme, year_of_study, gpa) VALUES
    ('15550001111', 'Ada Lovelace',      'ada@example.edu',      'BSc Computer Science', 2, 3.72),
    ('15550002222', 'Grace Hopper',      'grace@example.edu',    'BSc Computer Science', 3, 3.91),
    ('15550003333', 'Alan Turing',       'alan@example.edu',     'BSc Mathematics',      2, 3.40),
    ('15550004444', 'Katherine Johnson', 'katherine@example.edu', 'BSc Computer Science', 4, 3.85);

INSERT INTO courses (code, name, credits) VALUES
    ('CS1010', 'Introduction to Programming', 15),
    ('CS2010', 'Data Structures',           15),
    ('MA1010', 'Linear Algebra',            15),
    ('ST1010', 'Statistics',                10);

-- Ada and Grace
INSERT INTO enrollments (student_id, course_id)
SELECT s.id, c.id FROM students s, courses c
 WHERE s.wa_id IN ('15550001111', '15550002222')
   AND c.code IN ('CS1010', 'CS2010', 'MA1010');

-- Alan and Katherine
INSERT INTO enrollments (student_id, course_id)
SELECT s.id, c.id FROM students s, courses c
 WHERE s.wa_id IN ('15550003333', '15550004444')
   AND c.code IN ('CS1010', 'MA1010', 'ST1010');

-- Day of week: 1 monday .. 7 sunday
INSERT INTO timetable_entries (course_id, day_of_week, start_time, end_time, room)
SELECT c.id, entry.day_of_week, entry.start_time, entry.end_time, entry.room
  FROM courses c
  JOIN (VALUES
      ('CS1010', 1, '09:00'::time, '10:30'::time, 'B-104'),
      ('CS1010', 3, '09:00'::time, '10:30'::time, 'B-104'),
      ('CS1010', 5, '09:00'::time, '10:30'::time, 'B-104'),
      ('CS2010', 1, '11:00'::time, '12:30'::time, 'B-210'),
      ('CS2010', 3, '11:00'::time, '12:30'::time, 'B-210'),
      ('CS2010', 5, '11:00'::time, '12:30'::time, 'B-210'),
      ('MA1010', 2, '14:00'::time, '15:30'::time, 'A-007'),
      ('MA1010', 4, '14:00'::time, '15:30'::time, 'A-007'),
      ('ST1010', 2, '16:00'::time, '17:00'::time, 'C-002'),
      ('ST1010', 4, '16:00'::time, '17:00'::time, 'C-002')
  ) AS entry(code, day_of_week, start_time, end_time, room) ON entry.code = c.code;

-- Due dates are relative to the moment the seed runs, so the demo always shows live deadlines.
INSERT INTO assignments (course_id, title, due_at, weight_pct)
SELECT c.id, entry.title, NOW() + entry.due_offset, entry.weight_pct
  FROM courses c
  JOIN (VALUES
      ('CS2010', 'Linked list implementation',  '3 days'::interval, 15.00),
      ('CS2010', 'Binary search tree report',   '9 days'::interval, 20.00),
      ('CS2010', 'Hash table assignment',       '17 days'::interval, 15.00),
      ('MA1010', 'Eigenvalue problem set',      '5 days'::interval, 10.00),
      ('MA1010', 'Matrix transformations',      '12 days'::interval, 10.00),
      ('ST1010', 'Confidence intervals lab',    '7 days'::interval, 10.00)
  ) AS entry(code, title, due_offset, weight_pct) ON entry.code = c.code;

INSERT INTO submissions (assignment_id, student_id, status, submitted_at, grade, feedback)
SELECT a.id, s.id, sub.status,
       CASE WHEN sub.status = 'not_started' THEN NULL ELSE NOW() - '4 days'::interval END,
       sub.grade, sub.feedback
  FROM (VALUES
      ('15550001111', 'CS2010', 'Linked list implementation', 'graded',
       78.00::numeric, 'Correct, but the recursion is hard to follow.'),
      ('15550001111', 'MA1010', 'Eigenvalue problem set',    'submitted',
       NULL::numeric,  NULL)
  ) AS sub(wa_id, code, title, status, grade, feedback)
  JOIN students s ON s.wa_id = sub.wa_id
  JOIN courses c ON c.code = sub.code
  JOIN assignments a ON a.course_id = c.id AND a.title = sub.title;

-- A second student with a graded submission, so grade queries return more than one row.
INSERT INTO submissions (assignment_id, student_id, status, submitted_at, grade, feedback)
SELECT a.id, s.id, 'graded', NOW() - '6 days'::interval, 91.00, 'Excellent write-up.'
  FROM students s
  JOIN courses c ON c.code = 'MA1010'
  JOIN assignments a ON a.course_id = c.id AND a.title = 'Eigenvalue problem set'
 WHERE s.wa_id = '15550004444';

INSERT INTO exams (course_id, exam_date, starts_at, room)
SELECT c.id, CURRENT_DATE + entry.exam_day_offset, '09:00'::time, entry.room
  FROM (VALUES
      ('CS2010', 14, 'A-100'),
      ('MA1010', 21, 'B-200'),
      ('ST1010', 28, 'C-002')
  ) AS entry(code, exam_day_offset, room)
  JOIN courses c ON c.code = entry.code;

INSERT INTO exam_registrations (student_id, exam_id, seat)
SELECT s.id, x.id, entry.seat
  FROM (VALUES
      ('15550001111', 'CS2010', 'A-14'),
      ('15550002222', 'CS2010', 'A-15'),
      ('15550003333', 'MA1010', 'A-16'),
      ('15550004444', 'MA1010', 'A-17')
  ) AS entry(wa_id, code, seat)
  JOIN students s ON s.wa_id = entry.wa_id
  JOIN courses c ON c.code = entry.code
  JOIN exams x ON x.course_id = c.id;
