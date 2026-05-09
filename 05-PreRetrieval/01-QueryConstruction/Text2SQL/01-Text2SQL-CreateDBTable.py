# Connect to SQLite database
import sqlite3
conn = sqlite3.connect('90-Data/tourism.db')
cursor = conn.cursor()
# Create scenic spot information table
cursor.execute('''
CREATE TABLE IF NOT EXISTS scenic_spots (
    scenic_id INTEGER PRIMARY KEY,
    scenic_name VARCHAR(100) NOT NULL,
    city VARCHAR(50) NOT NULL,
    level VARCHAR(20),
    monthly_visitors INTEGER
)''')
# Create city information table
cursor.execute('''
CREATE TABLE IF NOT EXISTS city_info (
    city_id INTEGER PRIMARY KEY,
    city_name VARCHAR(50) NOT NULL,
    annual_tourism_income INTEGER,
    famous_dish VARCHAR(100)
)''')
# Insert sample data into the scenic spot information table
sample_scenic_spots = [
    (1, 'Jinci Temple', 'Taiyuan City', 'AAAAA', 50000),
    (2, 'Mount Wutai', 'Xinzhou City', 'AAAAA', 80000),
    (3, 'Yungang Grottoes', 'Datong City', 'AAAAA', 70000),
    (4, 'Pingyao Ancient City', 'Jinzhong City', 'AAAAA', 90000),
    (5, 'Qiao Family Compound', 'Jinzhong City', 'AAAA', 45000)
]
cursor.executemany('INSERT OR REPLACE INTO scenic_spots VALUES (?, ?, ?, ?, ?)', sample_scenic_spots)
# Insert sample data into the city information table
sample_city_info = [
    (1, 'Taiyuan City', 200000000, 'Dao Xiao Mian (Knife-cut noodles)'),
    (2, 'Datong City', 180000000, 'Datong Vinegar'),
    (3, 'Jinzhong City', 150000000, 'Saozi Noodles'),
    (4, 'Xinzhou City', 120000000, 'Youmian Kao Lao Lao'),
    (5, 'Yuncheng City', 130000000, 'Yuncheng Braised Cake')
]
cursor.executemany('INSERT OR REPLACE INTO city_info VALUES (?, ?, ?, ?)', sample_city_info)
# Commit changes and close connection
conn.commit()
conn.close()
print("Database tables created and sample data inserted successfully.")
