-- Limbot student database schema.
-- Safe to run repeatedly: every statement is guarded.
-- The bot process connects read only; this file is applied by an admin session, or by the
-- init container on first boot.

CREATE TABLE IF NOT EXISTS students (
    id            BIGSERIAL PRIMARY KEY,
    wa_id         TEXT        NOT NULL UNIQUE,
    full_name     TEXT        NOT NULL,
    email         TEXT,
    programme     TEXT,
    year_of_study SMALLINT    CHECK (year_of_study BETWEEN 1 AND 7),
    gpa           NUMERIC(3,2) CHECK (gpa >= 0 AND gpa <= 4.00)
);

CREATE TABLE IF NOT EXISTS courses (
    id      BIGSERIAL PRIMARY KEY,
    code    TEXT        NOT NULL UNIQUE,
    name    TEXT        NOT NULL,
    credits SMALLINT    CHECK (credits BETWEEN 1 AND 60)
);

CREATE TABLE IF NOT EXISTS enrollments (
    student_id BIGINT NOT NULL REFERENCES students (id) ON DELETE CASCADE,
    course_id  BIGINT NOT NULL REFERENCES courses (id)  ON DELETE CASCADE,
    PRIMARY KEY (student_id, course_id)
);

CREATE TABLE IF NOT EXISTS timetable_entries (
    id          BIGSERIAL PRIMARY KEY,
    course_id   BIGINT      NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
    day_of_week SMALLINT    NOT NULL CHECK (day_of_week BETWEEN 1 AND 7),
    start_time  TIME        NOT NULL,
    end_time    TIME        NOT NULL,
    room        TEXT,
    CHECK (end_time > start_time)
);

CREATE INDEX IF NOT EXISTS timetable_entries_course_day_idx
    ON timetable_entries (course_id, day_of_week, start_time);

CREATE TABLE IF NOT EXISTS assignments (
    id         BIGSERIAL PRIMARY KEY,
    course_id  BIGINT      NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
    title      TEXT        NOT NULL,
    due_at     TIMESTAMPTZ NOT NULL,
    weight_pct NUMERIC(5,2) CHECK (weight_pct >= 0 AND weight_pct <= 100)
);

CREATE INDEX IF NOT EXISTS assignments_due_idx ON assignments (due_at);

CREATE TABLE IF NOT EXISTS submissions (
    id            BIGSERIAL PRIMARY KEY,
    assignment_id BIGINT      NOT NULL REFERENCES assignments (id) ON DELETE CASCADE,
    student_id    BIGINT      NOT NULL REFERENCES students (id)   ON DELETE CASCADE,
    status        TEXT        NOT NULL CHECK (
        status IN ('not_started', 'in_progress', 'submitted', 'graded')
    ),
    submitted_at  TIMESTAMPTZ,
    grade         NUMERIC(5,2) CHECK (grade >= 0 AND grade <= 100),
    feedback      TEXT,
    PRIMARY KEY (assignment_id, student_id)
);

CREATE TABLE IF NOT EXISTS exams (
    id        BIGSERIAL PRIMARY KEY,
    course_id BIGINT NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
    exam_date DATE   NOT NULL,
    starts_at TIME,
    room      TEXT,
    UNIQUE (course_id, exam_date)
);

CREATE TABLE IF NOT EXISTS exam_registrations (
    student_id BIGINT NOT NULL REFERENCES students (id) ON DELETE CASCADE,
    exam_id    BIGINT NOT NULL REFERENCES exams (id)    ON DELETE CASCADE,
    seat       TEXT,
    PRIMARY KEY (student_id, exam_id)
);

-- Row level security would be the next step for a multi-tenant deployment. The bot connects
-- with default_transaction_read_only, so the app layer cannot write even without it.
