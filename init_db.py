"""Create the local MySQL database and prediction history table."""

from mysql.connector import Error

from config import DB_CONFIG
from database import initialize_database


def main():
    try:
        initialize_database()
        print(f"Database '{DB_CONFIG['database']}' and prediction_history table are ready.")
        return 0
    except Error as exc:
        print(f"Could not initialize MySQL: {exc}")
        print("Check that MySQL Server is running and config.py has the correct host, port, username, and password.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())