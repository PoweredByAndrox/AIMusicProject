import psycopg2
conn = psycopg2.connect(host='postgres', database='music_recommendations', user='postgres', password='secret')
cursor = conn.cursor()
cursor.execute("SELECT id, title, file_path, duration FROM tracks WHERE title LIKE '%UKF%2010%' OR title LIKE '%Dubstep 2010%'")
for r in cursor.fetchall():
    print(f'{r[0]}: {r[1]} ({r[3]/60:.1f}min) - {r[2]}')
cursor.close()
conn.close()
