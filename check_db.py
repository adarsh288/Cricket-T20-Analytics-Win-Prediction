import sqlite3
import os

print(os.path.abspath('cricket_t20.db'))
conn = sqlite3.connect('cricket_t20.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
conn.close()