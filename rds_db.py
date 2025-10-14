# rds_db.py
import mysql.connector

def get_rds_connection():
    return mysql.connector.connect(
        host="imagefilter-db.chooo8ucmy4k.ap-south-1.rds.amazonaws.com",
        user="admin",
        password="Shubh1814",
        database="imagefilter"  # the database you created
    )

def add_filter_history(filename, filter_type):
    try:
        conn = get_rds_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO filter_history (filename, filter_type) VALUES (%s, %s)",
            (filename, filter_type)
        )
        conn.commit()
    except Exception as e:
        print(f"RDS error: {e}")
    finally:
        cursor.close()
        conn.close()
