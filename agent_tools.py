import sqlite3
from datetime import datetime
from database import DB_PATH

def log_customer_data(customer_contact: str, user_id: str, metric_name: str, metric_value: str, notes: str = None) -> str:
    """
    Logs a specific data point for a customer (e.g., weight, blood sugar).
    Args:
        customer_contact (str): The customer's unique identifier (e.g., phone number).
        user_id (str): The ID of the business user this customer belongs to.
        metric_name (str): The name of the metric being logged (e.g., 'Weight', 'Blood Glucose').
        metric_value (str): The value of the metric (e.g., '75kg', '6.5 mmol/L').
        notes (str, optional): Any additional context or notes from the customer.
    Returns:
        A confirmation message.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO customer_progress (customer_contact, user_id, metric_name, metric_value, notes) VALUES (?, ?, ?, ?, ?)",
            (customer_contact, user_id, metric_name, metric_value, notes)
        )
        conn.commit()
        conn.close()
        return f"Successfully logged {metric_name} as {metric_value} for the customer."
    except Exception as e:
        return f"Failed to log data: {e}"

def generate_progress_report(customer_contact: str, user_id: str) -> str:
    """
    Generates a summary of all data logged for a specific customer.
    Args:
        customer_contact (str): The customer's unique identifier.
        user_id (str): The ID of the business user this customer belongs to.
    Returns:
        A formatted string containing the customer's progress report, or a message if no data exists.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT metric_name, metric_value, notes, strftime('%Y-%m-%d %H:%M', log_time) FROM customer_progress WHERE customer_contact = ? AND user_id = ? ORDER BY log_time DESC",
            (customer_contact, user_id)
        )
        records = cursor.fetchall()
        conn.close()

        if not records:
            return "No progress has been logged for this customer yet."

        report = "Here is the customer's progress report so far:\n"
        for record in records:
            notes_str = f" (Notes: {record[2]})" if record[2] else ""
            report += f"- On {record[3]}: {record[0]} was {record[1]}{notes_str}\n"
        
        return report
    except Exception as e:
        return f"Failed to generate report: {e}"

def get_all_customer_reports(user_id: str) -> dict:
    """
    Retrieves all progress reports for all customers of a specific business user.
    Used by the Admin UI.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT customer_contact, metric_name, metric_value, notes, strftime('%Y-%m-%d %H:%M', log_time) FROM customer_progress WHERE user_id = ? ORDER BY customer_contact, log_time DESC",
            (user_id,)
        )
        records = cursor.fetchall()
        conn.close()

        reports = {}
        for record in records:
            contact = record[0]
            if contact not in reports:
                reports[contact] = []
            
            notes_str = f" (Notes: {record[3]})" if record[3] else ""
            log_entry = f"- On {record[4]}: {record[1]} was {record[2]}{notes_str}"
            reports[contact].append(log_entry)
        
        return reports

    except Exception as e:
        print(f"Failed to fetch all customer reports: {e}")
        return {}

