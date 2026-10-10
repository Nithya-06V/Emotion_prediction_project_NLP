"""MySQL connection and prediction-history operations."""

import mysql.connector
import re

from config import BASE_DIR, DB_CONFIG


def database_connection(include_database=True):
    settings = dict(DB_CONFIG)
    if not include_database:
        settings.pop("database", None)
    return mysql.connector.connect(**settings, connection_timeout=3)


def initialize_database():
    database_name = DB_CONFIG["database"].replace("`", "``")
    connection = database_connection(include_database=False)
    try:
        cursor = connection.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.close()
    finally:
        connection.close()

    connection = database_connection()
    try:
        cursor = connection.cursor()
        schema_path = BASE_DIR / "schema.sql"
        schema_sql = schema_path.read_text(encoding="utf-8")
        schema_sql = re.sub(r"(?m)^\s*--.*$", "", schema_sql)
        if not schema_sql:
            raise ValueError(f"Database schema file is empty: {schema_path}")
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if not statement:
                continue
            if statement.upper().startswith(("CREATE DATABASE", "USE ")):
                continue
            cursor.execute(statement)
        connection.commit()
        cursor.close()
    finally:
        connection.close()


def insert_prediction(input_text, predicted_emotion):
    connection = database_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO prediction_history (input_text, predicted_emotion) VALUES (%s, %s)",
            (input_text, predicted_emotion),
        )
        connection.commit()
        cursor.close()
    finally:
        connection.close()


def get_prediction_history(limit=500):
    connection = database_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, input_text, predicted_emotion, predicted_at "
            "FROM prediction_history ORDER BY predicted_at DESC, id DESC LIMIT %s",
            (limit,),
        )
        records = cursor.fetchall()
        cursor.close()
        return records
    finally:
        connection.close()


def clear_prediction_history():
    connection = database_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM prediction_history")
        connection.commit()
        cursor.close()
    finally:
        connection.close()