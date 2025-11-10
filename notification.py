import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import streamlit as st # Used for Streamlit secrets access and logging/messages
import os
from dotenv import load_dotenv
load_dotenv()

# --- Global Configuration (Load from Environment) ---
# NOTE: These variables must be set in your .env file or Streamlit secrets
# The correct key names are used here, NOT the values.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") 
# --------------------------------------------------------------------------

def clean_for_tts(text):
    """
    Removes common Markdown characters and emojis from text before sending to TTS.
    This ensures clean, readable text is processed by a Text-to-Speech service.
    """
    # Remove common Markdown symbols
    cleaned_text = re.sub(r'[*\-_#\[\]()]', '', text)
    # Remove all non-alphanumeric characters except spaces, periods, commas, question marks, and exclamation marks.
    cleaned_text = re.sub(r'[^\w\s.,?!]', '', cleaned_text)
    # Collapse multiple spaces into single spaces
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

# --------------------------------------------------------------------------

def send_telegram_notification(message):
    """Sends a message to the specified Telegram chat ID."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # Configuration is missing, skip notification silently.
        return

    try:
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        # Display an informational message in the Streamlit app
        
        
        # Send the POST request to the Telegram API
        requests.post(telegram_url, data=payload, timeout=5)
    
    except requests.exceptions.RequestException as e:
        # Handle connection errors or timeouts silently to not interrupt the main application flow
        pass

# --------------------------------------------------------------------------

def send_email_notification(subject: str, body: str, receiver: str = RECEIVER_EMAIL):
    """
    Sends an email notification via SMTP using SSL on port 465.
    Requires SENDER_EMAIL and an App Password (EMAIL_PASSWORD) for Gmail.
    """
    
    # Guard clause to ensure all necessary credentials are available
    if not SENDER_EMAIL or not EMAIL_PASSWORD or not receiver:
        # Credentials are incomplete; cannot send email.
        return False
    
    try:
        # Create the email content
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = SENDER_EMAIL
        message["To"] = receiver
        
        # Convert the body text to HTML, preserving newlines as <br>
        html = f"<html><body><p>{body.replace('\n', '<br>')}</p></body></html>"
        message.attach(MIMEText(html, "html"))
        
        # Connect to Gmail SMTP server using SSL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            # Log in using the sender's email and App Password
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            # Send the email
            server.sendmail(SENDER_EMAIL, receiver, message.as_string())
        
        return True
        
    except smtplib.SMTPAuthenticationError as auth_err:
        # Specific error for incorrect login credentials (SENDER_EMAIL or App Password)
        return False
    except Exception as e:
        # Catch other potential errors (network, server issues)
        return False