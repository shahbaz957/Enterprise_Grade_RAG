# scripts/check_logfire.py
import logfire

logfire.configure()  # auto-discovers .logfire/logfire_credentials.json
logfire.info("Logfire connectivity check from Enterprise RAG")
print("Sent — check your Logfire dashboard for this log line.")