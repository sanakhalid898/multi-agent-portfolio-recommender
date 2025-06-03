import streamlit as st
from dotenv import load_dotenv
import os
import base64
import tempfile
import requests
from utils.logger import logger

#API Configs
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key is empty")
        raise ValueError("OpenAI API key is not set")
    logger.info("OpenAI API key loaded successfully")
except Exception as e:
    logger.error(f"Error loading OpenAI API key: {str(e)}")
    raise

GEMINI_API_KEY = "AIzaSyDrIw0IQZnPOc5w2db5MG97wvNrpbVFal8"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]
FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
GNEWS_API_KEY = st.secrets["GNEWS_API_KEY"]
ALPHA_VANTAGE_API_KEY = "V70BZ4ORJS6YRUMI"
FRED_API_KEY = "23395a74b6ef0e6647cf8f647c755ed4"

#Database Configs
AZURE_DATABASE=st.secrets["database"]["AZURE_DATABASE"]
AZURE_HOSTNAME=st.secrets["database"]["AZURE_HOSTNAME"]
AZURE_PASSWORD=st.secrets["database"]["AZURE_PASSWORD"]
AZURE_USER=st.secrets["database"]["AZURE_USER"]
AZURE_PORT=st.secrets["database"]["AZURE_PORT"]
# AZURE_SSL_CA=st.secrets["database"]["AZURE_SSL_CA"]

cert_base64 = st.secrets["database"]["AZURE_CERT"]
AZURE_SSL = base64.b64decode(cert_base64)

with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as tmp_cert_file:
    tmp_cert_file.write(AZURE_SSL)
    AZURE_SSL_CA = tmp_cert_file.name


load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
# FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
# MYSQL_HOST = os.getenv("MYSQL_HOST")
# MYSQL_USER = os.getenv("MYSQL_USER")
# MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
# MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
# GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# AZURE_DATABASE=os.getenv("AZURE_DATABASE")
# AZURE_HOSTNAME=os.getenv("AZURE_HOSTNAME")
# AZURE_PASSWORD=os.getenv("AZURE_PASSWORD")
# AZURE_USER=os.getenv("AZURE_USER")
# AZURE_PORT=os.getenv("AZURE_PORT")
# AZURE_SSL_CA=os.getenv("AZURE_SSL_CA")
# st.write(f"Using DB user: {FINNHUB_API_KEY}")
# st.write(f"Using DB user: {GROQ_API_KEY}")
# st.write(f"Using DB user: {GNEWS_API_KEY}")
# st.write(f"Using DB user: {NEWSAPI_KEY}")