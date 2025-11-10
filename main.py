import gspread
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials

from datetime import datetime
from datetime import timedelta
from google import genai
from google.genai.errors import APIError
from browser_agent import BrowserAgent 
try:
    import pandas as pd
except ImportError:
    # Handle case where pandas is not installed
    pd = None
import re
import io
import base64
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import time
import requests 
import streamlit as st
import base64
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# ... (other core imports)

SERPAPI_KEY = "83f7b9c91b6f9c1f57f939f8e49f5a218e7ac0950e5d19eee0adaf5029931101" 
os.environ["SERPAPI_KEY"] = SERPAPI_KEY
os.environ["GEMINI_API_KEY"] = "AIzaSyCo24il1vGTZZeIpT75Rr4WZzy7TR0Mhck"
private_key = st.secrets["gsheets"]["private_key"].replace('\\n', '\n')
creds = {
    "type": "service_account",
    "project_id": "agentswarm-mvp",  # Ya jo bhi aapka project ID hai
    "private_key": private_key,       # Nayi, Saaf Key
    "client_email": st.secrets["gspread"]["client_email"],
    "token_uri": "https://oauth2.googleapis.com/token",
}
try:
    # Service account se connect karein
    gc = gspread.service_account_from_dict(creds)
    # Baaki ka code jahan sheet open hoti hai
    spreadsheet_id = st.secrets["gsheets"]["MASTER_SHEET_ID"]
    spreadsheet = gc.open_by_key(spreadsheet_id)
    st.success("Google Sheets Connected Successfully!")

except Exception as e:
    st.error(f"Google Sheets Connection Error: {e}")
    # Fallback to Mock Data Mode
    st.info("Running in Mock Data Mode.")

from notification import (
    send_telegram_notification,
    send_email_notification,
    clean_for_tts
)

try:
    from streamlit_mic_recorder import mic_recorder
    USE_MIC_RECORDER = True
except ImportError:
    USE_MIC_RECORDER = False

st.set_page_config(layout="wide")

# ----------------------------------------------------
# --- NEW: 1. Configuration and Environment Variables ---
# ----------------------------------------------------

# --- Configuration (Update these environment variables/files) ---
SERVICE_ACCOUNT_FILE = 'agentswarm-mvp-ff5f0ed199ed.json'
SPREADSHEET_NAME = 'Agent Swarm MVP Tasks'
TASKS_SHEET_NAME = 'Master Tasks'
XP_LEDGER_SHEET_NAME = 'XP Ledger'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- NEW: User provided data (Use os.environ.get for actual secrets) ---
# NOTE: For security, use os.environ.get for actual production environment
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") 
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") 

# WHOOP Mock Data (Used in get_whoop_status and get_paei_weights)
WHOOP_RECOVERY = int(os.environ.get("WHOOP_RECOVERY", 65))
WHOOP_STRAIN = float(os.environ.get("WHOOP_STRAIN", 12.5))
WHOOP_SLEEP = float(os.environ.get("WHOOP_SLEEP", 7.5))

# Weather/Surf API (We will use a mock or a simple free one for MVP)
OPEN_WEATHER_API_KEY = os.environ.get("OPEN_WEATHER_API_KEY") 
USER_LOCATION_CITY = "Delhi, IN" # Fixed location for Weather Agent

ORCHESTRATION_SHEETS = {
    "Producer": 'P_Tasks',
    "Administrator": 'A_Tasks',
    "Entrepreneur": 'E_Tasks',
    "Integrator": 'I_Tasks'
}

scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive'
]

CLIENT = None 
SPREADSHEET = None
TASKS_SHEET = None
XP_LEDGER_SHEET = None
SHEET_CONNECTION_ERROR = None 

# --- Sheets Connection Setup and Mocking ---
try:
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    CLIENT = gspread.authorize(creds)
    SPREADSHEET = CLIENT.open(SPREADSHEET_NAME)
    TASKS_SHEET = SPREADSHEET.worksheet(TASKS_SHEET_NAME)
    XP_LEDGER_SHEET = SPREADSHEET.worksheet(XP_LEDGER_SHEET_NAME)
except Exception as e:
    SHEET_CONNECTION_ERROR = str(e)

# MOCKING for robustness (if connection fails)
if CLIENT is None:
    class MockSheet:
        def __init__(self, title):
            self.title = title
        def get_all_values(self):
            if self.title == 'Master Tasks':
                return [
                    ['Task Name', 'Category', 'Status', 'XP'],
                    ['Finish sales deck', 'Producer', 'Pending', '50'],
                    ['Schedule quarterly review', 'Administrator', 'Completed', '10'],
                    ['Brainstorm Q3 strategy', 'Entrepreneur', 'Pending', '30']
                ]
            elif self.title == 'XP Ledger':
                return [
                    ['Date', 'Task Name', 'Category', 'XP Gained', 'Total XP', 'Level', 'P_XP', 'A_XP', 'E_XP', 'I_XP'],
                    ['User Summary', 'Total XP', 'Level', 'Tasks Completed', 120, 2, 60, 40, 20, 0], 
                    ['2025-11-01 10:00:00', 'Task A', 'Producer', 50, 50, 1, 60, 40, 20, 0], # Note: XP columns are usually aggregated on row 2, these are mock history
                    ['2025-11-02 11:00:00', 'Task B', 'Administrator', 20, 70, 1, 60, 40, 20, 0]
                ]
            else: 
                return [['Task Name', 'Category', 'Status']]
        
        # Mocking cell values for row 2 (Summary Row)
        def cell(self, row, col):
            if row == 2 and col == 2: return type('obj', (object,), {'value': 120})() # Total XP
            if row == 2 and col == 3: return type('obj', (object,), {'value': 2})()  # Level
            if row == 2 and col == 4: return type('obj', (object,), {'value': 4})()  # Tasks Completed
            if row == 2 and col == 5: return type('obj', (object,), {'value': 60})() # Total XP (already handled by col 2, but following ledger structure)
            if row == 2 and col == 6: return type('obj', (object,), {'value': 2})()  # Level (already handled by col 3, but following ledger structure)
            if row == 2 and col == 7: return type('obj', (object,), {'value': 60})() # P_XP 
            if row == 2 and col == 8: return type('obj', (object,), {'value': 40})() # A_XP
            if row == 2 and col == 9: return type('obj', (object,), {'value': 20})() # E_XP
            if row == 2 and col == 10: return type('obj', (object,), {'value': 0})()  # I_XP
            if row == 3 and col == 1: return type('obj', (object,), {'value': '2025-11-01 10:00:00'})() 
            return type('obj', (object,), {'value': ''})() 
        
        def append_row(self, row): pass
        def update_cell(self, row, col, value): pass
        def find(self, task_name): 
            # Simplified mock find for the first task in the list
            if task_name == 'Finish sales deck': return type('obj', (object,), {'row': 2})() 
            return None
        def append_rows(self, rows): pass
    
    class MockSpreadsheet:
        def worksheet(self, name):
            return MockSheet(name)

    CLIENT = True 
    SPREADSHEET = MockSpreadsheet()
    TASKS_SHEET = SPREADSHEET.worksheet(TASKS_SHEET_NAME)
    XP_LEDGER_SHEET = SPREADSHEET.worksheet(XP_LEDGER_SHEET_NAME)
# --- End Sheets Mock ---
# ✅ BROWSER AGENT INITIALIZATION
if 'browser_agent' not in st.session_state:
    st.session_state.browser_agent = BrowserAgent() 
    
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
    
if 'tts_audio' not in st.session_state:
    st.session_state['tts_audio'] = None
# ----------------------------------------------------
# --- 2. HELPER FUNCTIONS (PAEI, XP, STATUS) ---
# ----------------------------------------------------

def google_sheets_date_to_number(value):
    """Converts a value that might be a date string back to its serial number (XP).
    Note: This is often used to get around Gspread's interpretation of numbers as dates."""
    if isinstance(value, str):
        # This part is complex and often unreliable across different sheet formats.
        # We focus on a robust integer conversion for XP values.
        pass
    try:
        return int(float(value)) # Use float first to handle cases like '120.0'
    except (ValueError, TypeError):
        return 0
    import re # Make sure 'import re' is at the very top of your file

def clean_for_tts(text):
    """Removes common Markdown characters and emojis from text before sending to TTS."""
    # 1. Remove common markdown symbols (*, #, _, [, ], etc.)
    cleaned_text = re.sub(r'[*\-_#\[\]()]', '', text)
    
    # 2. Remove emojis and checkmarks (basic pattern)
    # This removes non-word, non-space characters that often include emojis/checkmarks
    cleaned_text = re.sub(r'[^\w\s.,?!]', '', cleaned_text)
    
    # 3. Clean up extra spaces
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text


def get_all_status():
    """Fetches Total XP, Level, and Tasks Completed from XP Ledger Row 2."""
    if XP_LEDGER_SHEET is None:
        return {"Total XP": 0, "Current Level": 1, "Tasks Completed": 0}
    try:
        USER_ROW = 2
        # XP Ledger Row 2 structure:
        # Col 2: Total XP, Col 3: Level, Col 4: Tasks Completed
        current_xp_raw = XP_LEDGER_SHEET.cell(USER_ROW, 2).value 
        current_level_raw = XP_LEDGER_SHEET.cell(USER_ROW, 3).value 
        tasks_completed_raw = XP_LEDGER_SHEET.cell(USER_ROW, 4).value 
        
        current_xp = google_sheets_date_to_number(current_xp_raw)
        current_level = int(current_level_raw) if current_level_raw and current_level_raw.isdigit() else 1
        tasks_completed = int(tasks_completed_raw) if tasks_completed_raw and tasks_completed_raw.isdigit() else 0
        
        return {"Total XP": current_xp, "Current Level": current_level, "Tasks Completed": tasks_completed}
    except Exception:
        return {"Total XP": 0, "Current Level": 1, "Tasks Completed": 0}


def get_paei_levels():
    """Calculates PAEI levels based on XP Ledger columns."""
    if XP_LEDGER_SHEET is None:
        return {r: {'xp': 0, 'level': 1} for r in ORCHESTRATION_SHEETS.keys()}
    try:
        USER_ROW = 2
        # XP Ledger Row 2 structure:
        # Col 7: P_XP, Col 8: A_XP, Col 9: E_XP, Col 10: I_XP
        paei_cols = {'Producer': 7, 'Administrator': 8, 'Entrepreneur': 9, 'Integrator': 10}
        paei_data = {}
        for role, col_num in paei_cols.items():
            xp_raw = XP_LEDGER_SHEET.cell(USER_ROW, col_num).value
            xp = google_sheets_date_to_number(xp_raw)
            level = (xp // 100) + 1 # Simple level formula
            paei_data[role] = {'xp': xp, 'level': level}
        return paei_data
    except Exception:
        return {r: {'xp': 0, 'level': 1} for r in ORCHESTRATION_SHEETS.keys()}

def get_streak_status():
    """Simulates fetching the streak status."""
    return {"Current Streak": 5, "Max Streak": 12}

# --- NEW FUNCTION: Weather Agent Logic ---
@st.cache_data(ttl=600) # Cache for 10 minutes
def fetch_realtime_weather(city: str, api_key: str = None):
    """
    Fetches real-time weather data for a city.
    Mocks the data if API key is not provided or API call fails.
    """
    if api_key:
        # Placeholder for actual API call logic
        # For simplicity and to avoid dependency on a specific API key:
        pass # Actual implementation would go here

    # --- MOCK DATA for Delhi based on your context (Severe AQI) ---
    if "delhi" in city.lower():
        # Based on current context (Nov 2025 mock)
        return {
            "city": "Delhi, India",
            "temp_c": 14,
            "condition": "Foggy/Haze",
            "aqi_status": "Severe",
            "aqi_index": 391,
            "wind_mph": 4,
            "note": "Critical air quality. Outdoor activity should be avoided."
        }
    return {
        "city": "Unknown",
        "temp_c": 25,
        "condition": "Clear",
        "aqi_status": "Good",
        "aqi_index": 50,
        "wind_mph": 5,
        "note": "Clear day. Optimal for outdoor activity."
    }

# --- MODIFIED FUNCTION: WHOOP/Biometrics Status ---
def get_whoop_status():
    """Fetches/Simulates WHOOP recovery data using environment variables."""
    recovery_percent = WHOOP_RECOVERY # Now uses global mock/env variable
    
    if recovery_percent >= 67:
        label = "Green (Optimal)"
        emoji = "💚"
        color = "Green"
    elif recovery_percent >= 34:
        label = "Yellow (Good)"
        emoji = "💛"
        color = "Yellow"
    else: 
        label = "Red (Low)"
        emoji = "❤️"
        color = "Red"
        
    return {
        "recovery": recovery_percent, 
        "label": label, 
        "emoji": emoji, 
        "Recovery Color": color,
        "strain": WHOOP_STRAIN,
        "sleep": WHOOP_SLEEP
    }

# --- NEW FUNCTION: Parent Agent's PAEI Weighting Logic (Blueprint: 15:08) ---
def get_paei_weights(whoop_status: dict, weather_status: dict) -> dict:
    """
    Determines PAEI priorities based on WHOOP Recovery and Weather Agent data.
    This is the core Parent Agent Decision Logic.
    """
    recovery_color = whoop_status['Recovery Color']
    recovery_val = whoop_status['recovery']
    aqi_status = weather_status.get('aqi_status', 'Good')
    
    # Base Weights (Standard priority)
    weights = {'Producer': 0, 'Administrator': 0, 'Entrepreneur': 0, 'Integrator': 0}
    
    # 1. Apply Recovery Weights (Blueprint Logic)
    if recovery_color == "Red":
        # Low Recovery: Integrator gets high weight, Producer gets penalty/low weight
        weights['Integrator'] += 50
        weights['Producer'] -= 20
        weights['Administrator'] += 10 # Basic maintenance is okay
    elif recovery_color == "Yellow":
        # Good Recovery: Balanced day, slight lean towards core work (P) and maintenance (A)
        weights['Producer'] += 30
        weights['Administrator'] += 20
        weights['Integrator'] += 10
    elif recovery_color == "Green":
        # Optimal Recovery: High Output, focus on P and E
        weights['Producer'] += 50
        weights['Entrepreneur'] += 40
    
    # 2. Apply Environmental Weights (Weather Agent)
    if aqi_status in ["Severe", "Very Poor"]:
        # Severe AQI (Delhi Context): PENALTY on Integrator (I) for physical outdoor tasks, 
        # but the **Integrator's overall health score** should still be prioritized by scheduling rest.
        # We boost A and E (Indoor work)
        weights['Integrator'] -= 15 # Penalty for outdoor activity
        weights['Administrator'] += 25 # Boost for indoor routine/planning
        weights['Entrepreneur'] += 15 # Boost for indoor strategy
        
    elif aqi_status in ["Good", "Moderate"] and recovery_val >= 50:
        # Good Weather + Good Recovery: Boost Integrator for outdoor activities
        weights['Integrator'] += 20
        
    # Standardize weights to a 'Focus Score' (Sum of base + penalties/bonuses)
    focus_scores = {
        k: max(0, v + 50) for k, v in weights.items() # Add base 50 to avoid negative scores
    }
    
    # Find the top 2 recommended categories
    sorted_scores = sorted(focus_scores.items(), key=lambda item: item[1], reverse=True)
    top_tasks = [k for k, v in sorted_scores[:2]]
    
    return {
        "weights": focus_scores, 
        "top_tasks": top_tasks,
        "mode_note": (
            f"**Current State:** {whoop_status['label']} Recovery ({recovery_val}%) | **AQI:** {aqi_status}. "
            f"Parent Agent recommends focusing on **{top_tasks[0]}** and **{top_tasks[1]}** tasks due to combined biometrics and environment."
        )
    }

def get_task_recommendations_based_on_recovery(recovery_color=None):
    """MODIFIED: Deprecated in favor of get_paei_weights, but kept for compatibility."""
    # This is kept for backward compatibility, but the UI will now use the new function
    if recovery_color == "Green":
        mode = "High Output Mode"
        tasks = ["Producer", "Entrepreneur"]
        note = "Focus on high-value creative and core tasks."
    elif recovery_color == "Yellow":
        mode = "Maintenance Mode"
        tasks = ["Administrator", "Producer"]
        note = "Stick to routine admin and manageable core tasks."
    else: 
        mode = "Recovery Mode"
        tasks = ["Integrator", "Administrator"]
        note = "Prioritize rest, self-care, and basic administration."
    return {"mode": mode, "tasks": tasks, "note": note}

def display_xp_growth_chart():
    # ... (function remains the same) ...
    """Generates and displays an XP growth line chart using Pandas and Streamlit."""
    if pd is None or XP_LEDGER_SHEET is None:
        st.warning("Pandas is required to show the XP Chart.")
        return
        
    try:
        # Fetch data, skip header (Row 1) and summary (Row 2)
        ledger_data = XP_LEDGER_SHEET.get_all_values()
        if len(ledger_data) < 3: 
            st.info("No sufficient XP history to display yet.")
            return

        ledger_data = ledger_data[2:] # Actual transaction rows start from index 2
        data = []
        
        for row in ledger_data:
            # Ensure the row has at least 5 columns (Total XP is at index 4)
            if len(row) < 5: continue
            
            # --- Robust Date Parsing ---
            # Date is expected in column A (index 0)
            date_str = row[0].split()[0] 
            try:
                date_only = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                try:
                    date_only = datetime.strptime(date_str, "%d/%m/%Y").date()
                except:
                    continue # Skip if date format is invalid
                
            try:
                # Total XP is in column E (index 4)
                total_xp = int(row[4]) 
            except ValueError:
                continue # Skip if XP is not a valid number

            # Aggregate XP for the same date (for cleaner chart)
            if data and data[-1]['Date'] == date_only:
                data[-1]['Total XP'] = total_xp
            else:
                data.append({'Date': date_only, 'Total XP': total_xp})

        if not data:
            st.info("No valid XP entries found for the chart.")
            return

        df = pd.DataFrame(data).set_index('Date')
        st.subheader("📈 XP Growth Over Time")
        st.line_chart(df)
        
    except Exception as e:
        st.info(f"Chart error: {e}. Requires correctly formatted date and total XP columns in the Ledger.")


# ----------------------------------------------------
# --- 3. CORE LOGIC FUNCTIONS (GEMINI, DELEGATE, MARK) ---
# ----------------------------------------------------

def classify_task_with_gemini(task_name: str) -> str:
    # ... (function remains the same) ...
    """Classifies task using Gemini API or mock logic."""
    if not GEMINI_API_KEY:
        low = task_name.lower()
        if any(w in low for w in ["call", "email", "pay", "invoice", "organize", "schedule"]):
            return "Administrator"
        if any(w in low for w in ["idea", "pitch", "plan", "strategy", "brainstorm"]):
            return "Entrepreneur"
        if any(w in low for w in ["family", "call", "meet", "dinner", "date", "gym", "meditate"]):
            return "Integrator"
        return "Producer"
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        system_instruction = (
            "You are a specialist PAEI (Producer, Administrator, Entrepreneur, Integrator) "
            "task categorization agent. Respond ONLY with the single most suitable category name from the list: Producer, Administrator, Entrepreneur, Integrator. Do not add any extra text, punctuation, or explanation."
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Classify the following task: {task_name}",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        paei_category = response.text.strip()
        valid_paei = ["PRODUCER", "ADMINISTRATOR", "ENTREPRENEUR", "INTEGRATOR"]
        if paei_category.upper() in valid_paei:
            return paei_category.title()
        else:
            return "Producer"
    except APIError:
        return "Producer" # Fallback if API key is invalid or quota exceeded
    except Exception:
        return "Producer"

def delegate_task(task_row: list, paei_category: str):
    # ... (function remains the same) ...
    """Delegates the task to the specific PAEI sheet."""
    if paei_category not in ORCHESTRATION_SHEETS:
        return
    sheet_name = ORCHESTRATION_SHEETS[paei_category]
    if SPREADSHEET is None:
        return
        
    try:
        target_sheet = SPREADSHEET.worksheet(sheet_name)
        target_sheet.append_row(task_row)
        time.sleep(0.5)
    except Exception:
        pass

def mark_task_as_completed(task_index: int, paei_category: str):
    # ... (function remains the same) ...
    """Marks a task as completed in Master and PAEI sheets."""
    if SPREADSHEET is None:
        return False
    
    # +2 because row 1 is header, +1 for 0-based index to 1-based row number
    master_row_num = task_index + 2 
    
    try:
        # 1. Update Master Tasks Sheet
        TASKS_SHEET.update_cell(master_row_num, 3, 'Completed') 
        time.sleep(0.5) 
    
        # 2. Update Orchestration Sheet 
        sheet_name = ORCHESTRATION_SHEETS.get(paei_category)
        if sheet_name:
            target_sheet = SPREADSHEET.worksheet(sheet_name)
            task_name = TASKS_SHEET.cell(master_row_num, 1).value
            
            # Find the row in the PAEI sheet based on the task name
            cell = target_sheet.find(task_name)
            
            if cell:
                # Assuming Status is in the 3rd column of the PAEI sheets
                target_sheet.update_cell(cell.row, 3, 'Completed')
                time.sleep(0.5) 
                st.success(f"✅ Task Added'{task_name}' completed and status updated in **{paei_category}** sheet!")
            
        return True
    except Exception as e:
        st.error(f"Error marking task as complete: {e}")
        return False

# --- FIXED/ADDED: update_xp_ledger function ---
def update_xp_ledger(xp_transactions: list):
    # ... (function remains the same) ...
    """
    Updates the XP Ledger with new transactions and updates the summary row (Row 2).
    :param xp_transactions: List of dicts, e.g., [{'name': 'Task A', 'category': 'Producer', 'xp': 50}]
    """
    if XP_LEDGER_SHEET is None or not xp_transactions:
        return

    try:
        USER_ROW = 2
        
        # 1. Get current status from the summary row (Row 2)
        # XP Ledger Row 2 structure: Col 2: Total XP, Col 3: Level, Col 4: Tasks Completed
        current_status = get_all_status()
        final_xp = current_status['Total XP']
        tasks_completed = current_status['Tasks Completed']
        
        # Get current PAEI XP from the summary row
        paei_cols_map = {'Producer': 7, 'Administrator': 8, 'Entrepreneur': 9, 'Integrator': 10}
        current_paei_xp = get_paei_levels() # Returns {'Producer': {'xp': 60, 'level': 1}, ...}
        
        current_paei_xp_only = {role: data['xp'] for role, data in current_paei_xp.items()}

        updates_to_ledger = []
        
        # 2. Process transactions
        for t in xp_transactions:
            xp_gained = t['xp']
            category = t['category']
            
            final_xp += xp_gained
            final_level = (final_xp // 100) + 1
            
            # Update PAEI XP totals
            if category in current_paei_xp_only:
                current_paei_xp_only[category] += xp_gained
            
            # Prepare new transaction row
            new_row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                t['name'], 
                category, 
                xp_gained, 
                final_xp, 
                final_level,
                current_paei_xp_only.get('Producer', 0),
                current_paei_xp_only.get('Administrator', 0),
                current_paei_xp_only.get('Entrepreneur', 0),
                current_paei_xp_only.get('Integrator', 0)
            ]
            updates_to_ledger.append(new_row)
            
        # 3. Append new transaction rows
        if updates_to_ledger:
            XP_LEDGER_SHEET.append_rows(updates_to_ledger)
            time.sleep(1)

            final_tasks_completed = tasks_completed + len(xp_transactions)
            
            # 4. Update Summary Row (Row 2) - Batch updates recommended, but doing sequential for simplicity and gspread's rate limits
            XP_LEDGER_SHEET.update_cell(USER_ROW, 2, final_xp) # Total XP
            time.sleep(0.2)
            XP_LEDGER_SHEET.update_cell(USER_ROW, 3, final_level) # Level
            time.sleep(0.2)
            XP_LEDGER_SHEET.update_cell(USER_ROW, 4, final_tasks_completed) # Tasks Completed
            time.sleep(0.2)

            for role, col_num in paei_cols_map.items():
                XP_LEDGER_SHEET.update_cell(USER_ROW, col_num, current_paei_xp_only[role])
                time.sleep(0.2) 
                
            st.success(f"🏆 XP Ledger Updated! Total XP: **{final_xp}** (Level **{final_level}**)")

    except Exception as e:
        st.error(f"Error updating XP Ledger: {e}")

def check_for_completed_tasks(auto_ack=True):
    # ... (function remains the same) ...
    """Scans Master Tasks for 'Completed' status and awards XP."""
    if SPREADSHEET is None or TASKS_SHEET is None:
        return "❌ Check failed: Google Spreadsheet connection is missing."
        
    xp_transactions = [] 
    updates = [] 
    
    try:
        all_master_tasks = TASKS_SHEET.get_all_values()
        
        # Start from row 2 (index 1) of the fetched list, which corresponds to row 2 in the sheet
        for i, row in enumerate(all_master_tasks[1:], start=2): 
            if len(row) < 4: continue

            task_name = row[0]
            paei_category = row[1].strip().title() if len(row) > 1 else "Unknown" 
            task_status = row[2].strip() 
            task_xp_str = row[3].strip() if len(row) > 3 else "0" 
            
            if task_status.lower() == 'completed':
                if paei_category in ORCHESTRATION_SHEETS.keys():
                    try:
                        xp_value = int(task_xp_str) 
                        if xp_value > 0:
                            xp_transactions.append({
                                'name': task_name,
                                'category': paei_category,
                                'xp': xp_value
                            })
                            # Mark this task to be acknowledged in the Master Sheet (to avoid double-counting)
                            updates.append((i, 3, 'COMPLETED')) 
                    except ValueError:
                        continue # Skip if XP is not a valid number
                else:
                    # Mark invalid PAEI tasks as acknowledged so they don't block the list
                    updates.append((i, 3, 'COMPLETED')) 

    except Exception as e:
        return f"❌ Error during Master Tasks check: {e}"
        
    if xp_transactions:
        # 1. Update the XP Ledger
        update_xp_ledger(xp_transactions)
        
        # 2. Acknowledge the tasks in the Master Sheet
        if updates:
            TASKS_SHEET.batch_update([{
                        'range': f'C{row_index}', 
                        'values': [[new_status]] 
                    } for row_index, col_index, new_status in updates])
        
        total_xp_awarded = sum(t['xp'] for t in xp_transactions)
        return f"🎊 **{total_xp_awarded} XP** awarded from Master Task List! (PAEI XP Updated)"
    else:
        return "No new completed tasks found in Master Task List."
    
def generate_weekly_report():
    # ... (function remains the same) ...
    """Generates a summary of last 7 days completed tasks and sends via Telegram."""
    if XP_LEDGER_SHEET is None:
        return "❌ Report failed: Sheet connection missing."
    
    seven_days_ago = datetime.now() - timedelta(days=7)
    tasks_completed_this_week = 0
    xp_gained_this_week = 0
    category_counts = {r: 0 for r in ORCHESTRATION_SHEETS.keys()}
    
    try:
        ledger_data = XP_LEDGER_SHEET.get_all_values()[2:] # Skip header (Row 1) and summary (Row 2)
        
        for row in ledger_data:
            if len(row) < 4: continue
            
            # Date is in Column A (index 0)
            date_str = row[0].split()[0]
            try:
                task_date = datetime.strptime(date_str, "%Y-%m-%d") 
            except ValueError:
                try:
                    task_date = datetime.strptime(date_str, "%d/%m/%Y") 
                except:
                    continue
                    
            # Check if task was completed in the last 7 days (including the start of the 7th day)
            if task_date >= seven_days_ago.replace(hour=0, minute=0, second=0, microsecond=0):
                try:
                    xp = int(row[3])
                except ValueError:
                    xp = 0
                    
                tasks_completed_this_week += 1
                xp_gained_this_week += xp
                category = row[2].strip().title()
                if category in category_counts:
                    category_counts[category] += xp
        
        report_message = (
            "📄 **WEEKLY AGENT SWARM REPORT** 🤖\n\n"
            f"**Total Tasks Completed:** {tasks_completed_this_week}\n"
            f"**Total XP Gained:** {xp_gained_this_week}\n\n"
            "**PAEI XP Breakdown (Last 7 Days):**\n"
        )
        
        for category, xp in category_counts.items():
            report_message += f" - **{category[0]} ({category}):** {xp} XP\n"
            
        report_message += "\n*Focus on the areas with lowest XP next week for balance!*"
        return report_message
        
    except Exception as e:
       return f"❌ Error generating report: {e}"

# ----------------------------------------------------
# --- 4. INPUT/OUTPUT FUNCTIONS (VOICE, TTS, ADD TASK) ---
# ----------------------------------------------------

def speak_text_autoplay(text: str):
    try:
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        st.session_state['tts_audio'] = fp.getvalue()
        st.session_state['tts_message'] = text
        
    except Exception:
        pass 


def transcribe_audio_bytes_to_text(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return ""
    
    r = sr.Recognizer()
    
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        
        final_wav_bytes = io.BytesIO()
        audio_segment.export(final_wav_bytes, format="wav")
        final_wav_bytes.seek(0)
        
        with sr.AudioFile(final_wav_bytes) as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = r.record(source) 
            
        text = r.recognize_google(audio_data)
        
        return text 
        
    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand the audio.") 
        return "" # Added return to handle flow
    except Exception: # Catch other errors like no internet, pydub issues, etc.
        return ""


def parse_input_and_add_task(task_input: str, auto_play_audio: bool = False):
    # ... (function remains the same) ...
    """Parses a task, classifies it, assigns XP, and adds it to Master/Orchestration sheets."""
    if not task_input.strip():
        st.warning("Task input cannot be empty.")
        return

    # Simple XP assignment logic
    word_count = len(task_input.split())
    xp_value = 50 if word_count > 10 else (30 if word_count > 5 else 10)
        
    st.info(f"✨ Classifying task: **{task_input}**...")
    paei_category = classify_task_with_gemini(task_input)
    
    if paei_category.title() not in ORCHESTRATION_SHEETS.keys():
        st.error(f"Could not classify task. Defaulting to 'Producer'.")
        paei_category = "Producer"

    # task_row: [Task Name, Category, Status, XP]
    task_row = [task_input, paei_category, 'Pending', str(xp_value)]
    
    try:
        # 1. Add to Master Tasks Sheet
        TASKS_SHEET.append_row(task_row)
        time.sleep(0.5) 

        # 2. Delegate to PAEI-specific Sheet
        delegate_task(task_row, paei_category)
        
        display_message = f"✅ Task added **'{task_input}'** delegated! Agent **{paei_category}** is on it. ({xp_value} XP available)" 
        st.success(display_message)
        
        telegram_message = f"**🚨 New Task Alert!**\n\n**Task:** {task_input}\n**Category:** {paei_category}\n**XP:** {xp_value}"
        send_telegram_notification(
          message=  telegram_message
        )   
        st.info("✉️ Telegram notification sent successfully.")
        email_subject = f"AGENT SWARM: New {paei_category} Task Delegated"
        email_body = f"A new task has been assigned to the {paei_category} Agent.\n\nTask: {task_input}\nXP Value: {xp_value}\n\nCheck your dashboard for details."
        
        
        send_email_notification(
         subject=email_subject,
         body=email_body
        )
        st.info("✉️ Email notification sent successfully.")
            
        if auto_play_audio:
            # FIX: Redundant clean_for_tts calls ko hata diya gaya.
            tts_raw_text = f"Task Added Category {paei_category}. {xp_value} XP."
            tts_readable_text = clean_for_tts(tts_raw_text)
            speak_text_autoplay(tts_readable_text)
        
        # Use session state to trigger input clear and rerun
        st.session_state['should_clear_input'] = True
        st.rerun()

    except Exception as e:
        st.error(f"An error occurred while adding the task to Sheets: {e}")



# ----------------------------------------------------
# --- 5. STREAMLIT UI FUNCTION ---
# ----------------------------------------------------
def streamlit_ui():
    st.set_page_config(layout="wide")
    st.title("🧠 Agent Swarm Parent Dashboard")
    
    # --- Session State Initialization ---
    if 'transcribed_text' not in st.session_state: st.session_state['transcribed_text'] = ""
    if 'last_processed_audio_id' not in st.session_state: st.session_state['last_processed_audio_id'] = None
    if 'auto_delegate' not in st.session_state: st.session_state['auto_delegate'] = False
    if 'should_clear_input' not in st.session_state: st.session_state['should_clear_input'] = False
    if 'last_auto_check' not in st.session_state: st.session_state['last_auto_check'] = 0
    if 'tts_audio' not in st.session_state: st.session_state['tts_audio'] = None
    if 'tts_message' not in st.session_state: st.session_state['tts_message'] = None
    if 'browser_agent' not in st.session_state:
        st.session_state['browser_agent'] = BrowserAgent()
    if 'search_history' not in st.session_state:
        st.session_state['search_history'] = []
    
    # --- FIXED TTS: AUDIO DISPLAY LOGIC ---
    

    if 'tts_audio' in st.session_state and st.session_state['tts_audio']:
    
     audio_bytes = st.session_state.get('tts_audio', None) 
     tts_message = st.session_state.get('tts_message', "System Response") 

     if audio_bytes:
        
        st.info(f"🔊 System spoke: *{tts_message}*")
        
        try:
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            audio_html = f"""
                <audio autoplay="true">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
            """
            
            st.markdown(audio_html, unsafe_allow_html=True)
            
        except Exception:
            pass
            
        finally:
            st.session_state['tts_audio'] = None
            st.session_state['tts_message'] = None
    # --- END AUDIO DISPLAY LOGIC ---
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("XP Ledger")
        xp_data = get_all_status()
        st.metric(label="Current XP", value=xp_data['Total XP'])
        st.metric(label="Current Level", value=xp_data['Current Level'])
        
        paei_levels = get_paei_levels()
        st.subheader("👑 PAEI Mastery")
        for role, data in paei_levels.items():
            st.markdown(f"**{role[0]} ({role}):** Lvl **{data['level']}** ({data['xp']} XP)")

        streak_status = get_streak_status()
        st.markdown(f"**🔥 Streak Status:** {streak_status['Current Streak']} days (Best: {streak_status['Max Streak']} days)")

        # --- MODIFIED: Recovery & Focus ---
        st.subheader("❤️ Contextual Focus (Parent Agent)")
        try:
            whoop_status = get_whoop_status()
            weather_status = fetch_realtime_weather(USER_LOCATION_CITY, OPEN_WEATHER_API_KEY)
            paei_weights = get_paei_weights(whoop_status, weather_status)
            
            st.markdown(f"{whoop_status['emoji']} **Recovery:** **{whoop_status['recovery']}%** ({whoop_status['label']})")
            st.markdown(f"☁️ **Weather (Delhi):** **{weather_status['temp_c']}°C** | AQI: **{weather_status['aqi_status']}**")
            
            st.info(paei_weights['mode_note'])
            
            st.markdown("---")
            st.subheader("🎯 PAEI Weighting (Focus Scores)")
            
            # Display PAEI Focus Scores
            for role, score in paei_weights['weights'].items():
                emoji = "⭐" if role in paei_weights['top_tasks'] else "🔸"
                st.markdown(f"{emoji} **{role}:** {score} Points")
                
        except Exception as e:
            st.markdown(f"*(Context loading failed: {e})*")

        st.markdown("---")
        auto_mode = st.checkbox("Auto-check Completed Tasks (every 30s)", value=False, key='auto_check_box')
        display_xp_growth_chart()

    # --- 1. Control Buttons (RESTORED) ---
    col_report, col_update, col_spacer = st.columns([3, 3, 4])
    
    with col_report:
      
        # Button to generate the weekly report and send it via Telegram (MOVED HERE)
        if st.button("Generate & Send Weekly Report", key="generate_report_btn", use_container_width=True, type="primary"):
            with st.spinner('Generating report for the last 7 days...'):
                # Call the core report generation function
                report_output = generate_weekly_report() 
                
                # Store the report in session state so it persists after page rerun
                st.session_state['weekly_report_result'] = report_output
                
                if "❌" not in report_output:
                    st.success("Report generated and sent via Telegram And Email successfully!")
                else:
                    st.error("Report generation or Telegram send failed.")
    
    with col_update:
        if st.button("Check Updates", use_container_width=True, type="secondary"):
           st.info("Checking for new system updates...")

    # --- 2. Task Delegation Input ---
    st.header("Delegate a New Task")
    col1, col2 = st.columns([3,1])
    
    if st.session_state.get('should_clear_input'):
        st.session_state['transcribed_text'] = ""
        st.session_state['should_clear_input'] = False
        st.session_state['auto_delegate'] = False
        st.rerun() 
    
    with col1:
        task_input = st.text_input(
            "delegate a task :", 
            value=st.session_state.get('transcribed_text', ''), 
            key="task_input_key"
        )
    
    with col2:
        st.write("") 
        st.write("🎤 **Voice Delegate**")
        
        if USE_MIC_RECORDER:
            audio_data = mic_recorder(
                start_prompt="🎙️ Click to Speak",
                stop_prompt="",
                format="wav",
                key='mic_recorder_key',
                use_container_width=True
            )

            if audio_data and audio_data.get('bytes'):
                audio_id = id(audio_data['bytes'])
                if st.session_state['last_processed_audio_id'] != audio_id:
                    st.session_state['last_processed_audio_id'] = audio_id
                    with st.spinner("🎧 Transcribing..."):
                        transcribed = transcribe_audio_bytes_to_text(audio_data['bytes'])
                    if transcribed and transcribed != "TRANSCRIPTION_FAILED":
                        st.session_state['transcribed_text'] = transcribed
                        st.session_state['auto_delegate'] = True
                        st.rerun() 
                    else:
                        st.session_state['transcribed_text'] = ""
                        st.error("❌ **Transcription Failed!** Try again.")
            
        else:
             st.warning("🚨 Mic component not installed.")
            
    if st.session_state.get('auto_delegate'):
        if st.session_state.get('transcribed_text') and st.session_state.get('transcribed_text') != "TRANSCRIPTION_FAILED":
            parse_input_and_add_task(st.session_state['transcribed_text'], auto_play_audio=True)
            st.session_state['transcribed_text'] = ""

    if st.button("🚀 Delegate Task", type="primary", use_container_width=True):
        if task_input and task_input.strip():
            parse_input_and_add_task(task_input, auto_play_audio=True)
        else:
            st.warning("⚠️ Please enter a task before delegating.")
            st.rerun()

    st.markdown("---")
    st.subheader("🌐 Browser/Research Agent")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "🔍 Research Query:", 
            placeholder="e.g., best productivity tools 2024",
            key="browser_search_input"  
 )
    with col2:
        agent_filter = st.selectbox(
            "Agent Type:", 
            ["All", "Producer", "Administrator", "Entrepreneur", "Integrator"],
            key="browser_agent_filter"
        )
    if st.button("🔎 Search Web", type="primary", key="browser_search_btn"):
        if search_query:
            with st.spinner("🔍 Searching the web..."):
                try:
                    if agent_filter == "All":
                        results = st.session_state.browser_agent.search_web(search_query, num_results=5)
                    else:
                        results = st.session_state.browser_agent.research_for_agent(search_query, agent_filter)
                    
                    if results:
                        st.success(f"✅ Found {len(results)} results!")
                        
                        for result in results:
                          with st.expander(f"🔗 {result['rank']}. {result['title'][:80]}..."):
                                st.write(f"**URL:** {result['url']}")
                                st.caption(f"**Snippet:** {result['snippet']}")
                                
                                if st.button(f"➕ Create Task from Result", key=f"task_create_{result['rank']}"):
                                    task_from_search = f"Research: {result['title'][:50]}"
                                    parse_input_and_add_task(task_from_search, auto_play_audio=False)
                        st.session_state.search_history.append({
                            'query': search_query,
                            'agent': agent_filter,
                            'results': len(results),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
                    else:
                        st.warning("⚠️ No results found. Try different keywords.")
                        
                except Exception as e:
                    st.error(f"Search error: {e}")
    else:
            st.warning("⚠️ Please enter a search query!")

    # Show search history
    if st.session_state.search_history:
        with st.expander("📜 Recent Searches"):
            for hist in st.session_state.search_history[-5:]:
                st.text(f"🔍 {hist['timestamp']} - {hist['query']} ({hist['agent']}) - {hist['results']} results")
    
    # ====================================================================
    # --- END BROWSER AGENT SECTION ---
    # ====================================================================

    # --- 3. Master Tasks Display (Corrected Headers and Logic) ---

    if TASKS_SHEET is not None:
        try:
            st.markdown("---")
            
            master_tasks_data = TASKS_SHEET.get_all_values()
            
            if pd is None:
                st.warning("Pandas is required to display the task breakdown chart.")
                return 

            if len(master_tasks_data) > 1:
                columns = master_tasks_data[0]
                df_tasks = pd.DataFrame(master_tasks_data[1:], columns=columns)
                
                category_column = None
                
                if 'Category' in df_tasks.columns:
                    category_column = 'Category'
                elif 'PAEI' in df_tasks.columns:
                    df_tasks.rename(columns={'PAEI': 'Category'}, inplace=True)
                    category_column = 'Category'
                
                if category_column is None:
                    st.error("❌ Google Sheet Error: Required column header ('Category' or 'PAEI') not found.")
                    return

                st.markdown("### 📋 Pending Tasks & Action")
                df_pending_tasks = df_tasks[df_tasks['Status'].astype(str).str.strip().str.upper() == 'PENDING'].copy()
                df_pending_tasks['Sheet Row Index'] = df_pending_tasks.index + 2 
                
                if not df_pending_tasks.empty:
                    for index, row in df_pending_tasks.iterrows():
                        task_name = row['Task Name']
                        sheet_row_index = row['Sheet Row Index']
                        
                        unique_id_part = row.get('Timestamp', str(sheet_row_index))
                        button_key = f"complete_btn_{sheet_row_index}_{unique_id_part}" 

                        col1, col2, col3 = st.columns([5, 2, 2])
                        
                        with col1:
                            st.markdown(f"**{task_name}**")
                        with col2:
                            xp_value = row.get('XP Value', 'N/A')
                            st.markdown(f"*{row['Category']} ({xp_value} XP)*")
                        with col3:
                            if st.button(f"✅ Complete", key=button_key):
                                status_col_index = df_tasks.columns.get_loc('Status') + 1
                                TASKS_SHEET.update_cell(sheet_row_index, status_col_index, 'COMPLETED')
                                time.sleep(1)
                                response = check_for_completed_tasks(auto_ack=True)
                                st.success(f"Task '{task_name}' marked as complete! Refreshing...")
                                st.rerun()
                                
                else:
                    st.info("🎉 All pending tasks are cleared!")


                if category_column and 'Status' in df_tasks.columns and not df_pending_tasks.empty:
                    st.markdown("---")
                    st.subheader("📊 Pending Task Breakdown (PAEI)") 
                    
                    category_counts = df_pending_tasks[category_column].value_counts().reset_index()
                    category_counts.columns = ['Category', 'Task Count']
                    
                    st.bar_chart(category_counts.set_index('Category'))
                
            else:
                st.info("No tasks found in the Master Task Sheet.")

        except Exception as e:
            st.error(f"Error loading Master Tasks: {e}")
    
if __name__ == '__main__':
    if SHEET_CONNECTION_ERROR:
        st.error(f"Google Sheets Connection Error: {SHEET_CONNECTION_ERROR}")
        st.warning("Running in Mock Data Mode. Tasks will not be saved or fetched live.")
    
    streamlit_ui()