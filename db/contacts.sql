-- Access control seed. Grants the owner elevated access and documents how to add testers.
--
-- Run AFTER db/schema.sql and db/seed.sql. docker compose applies everything under db/ in
-- lexical order (01-schema, 02-seed, 03-link, 04-contacts), so a fresh stack is set up out of
-- the box. Against an existing database:
--
--     psql "$POSTGRES_DSN" -f db/schema.sql
--     psql "$POSTGRES_DSN" -f db/contacts.sql
--
-- The bot never writes here: its Postgres pool is opened read only. Revoke access by lowering
-- the role, for example:
--
--     UPDATE contacts SET role = 'blocked', updated_at = NOW() WHERE wa_id = '...';
--
-- Idempotent: re-running updates the row in place rather than failing.
\set phone 23279826564

-- The owner's number: full access, so administration and end-to-end testing work. Because the
-- students row for this number already exists (db/link_student.sql), the record tools resolve
-- through students.wa_id; student_id is left null.
INSERT INTO contacts (wa_id, role, note)
VALUES (:'phone', 'admin', 'Owner: full access for administration and testing.')
ON CONFLICT (wa_id) DO UPDATE
   SET role = EXCLUDED.role,
       note = EXCLUDED.note,
       updated_at = NOW();

-- To let somebody else test, grant them student-level access that is trivial to revoke. Point
-- student_id at a students row so the record tools resolve without hijacking anyone's number:
--
--   INSERT INTO contacts (wa_id, role, student_id)
--   VALUES ('<their-wa-id>', 'tester', <a-students-id>)
--   ON CONFLICT (wa_id) DO UPDATE
--      SET role = EXCLUDED.role, student_id = EXCLUDED.student_id, updated_at = NOW();
--
-- A normal student needs no contacts row at all: link their number to a students row the way
-- db/link_student.sql does. A row here is only for overriding the default or blocking a number.

SELECT wa_id, role, student_id, note FROM contacts ORDER BY wa_id;
