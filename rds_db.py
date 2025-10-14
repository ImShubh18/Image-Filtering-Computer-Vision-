# rds_db.py
import mysql.connector
from mysql.connector import Error

def get_rds_connection():
    """Create and return a new RDS connection"""
    return mysql.connector.connect(
        host="imagefilter-db.chooo8ucmy4k.ap-south-1.rds.amazonaws.com",
        user="admin",
        password="Shubh1814",
        database="imagefilter"  # Ensure this database exists
    )

def add_filter_history(filename, filter_type):
    """Insert a filter history record into the RDS database"""
    conn = None
    cursor = None
    try:
        conn = get_rds_connection()
        cursor = conn.cursor()

        # Ensure table exists (optional: can remove if table is already created)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filter_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                filter_type VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert the record
        cursor.execute(
            "INSERT INTO filter_history (filename, filter_type) VALUES (%s, %s)",
            (filename, filter_type)
        )
        conn.commit()
        print(f"✅ Inserted into RDS: {filename} | {filter_type}")

    except Error as e:
        print(f"⚠️ RDS error while inserting {filename}, {filter_type}: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
