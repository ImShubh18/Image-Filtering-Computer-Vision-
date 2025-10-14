# rds_db.py
import mysql.connector
from mysql.connector import Error

def get_rds_connection():
    return mysql.connector.connect(
        host="imagefilter-db.chooo8ucmy4k.ap-south-1.rds.amazonaws.com",
        user="admin",
        password="Shubh1814",
        database="imagefilter"
    )

def add_filter_history(filename, filter_type):
    cursor = None
    conn = None
    try:
        conn = get_rds_connection()
        cursor = conn.cursor()
        print(f"Attempting to insert into RDS: {filename} | {filter_type}")
        cursor.execute(
            "INSERT INTO filter_history (filename, filter_type) VALUES (%s, %s)",
            (filename, filter_type)
        )
        conn.commit()
        print(f"✅ Successfully inserted: {filename} | {filter_type}")
    except Error as e:
        print(f"⚠️ RDS error while inserting {filename}, {filter_type}: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
