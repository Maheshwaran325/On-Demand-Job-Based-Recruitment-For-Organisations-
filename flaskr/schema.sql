DROP TABLE IF EXISTS user;


CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);

-- CREATE TABLE resumes (
--     id INTEGER PRIMARY KEY,
--     name TEXT,
--     skills TEXT,
--     experience TEXT,
--     education TEXT
--     );