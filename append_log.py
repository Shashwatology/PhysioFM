import sys
import time

log_path = r"C:\Users\MNIT USER\.gemini\antigravity-ide\brain\c0100c4e-932e-4971-9545-c6717d1e59d7\.system_generated\tasks\task-546.log"
try:
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
except Exception as e:
    log_content = str(e)

with open(r"f:\INTERN DATA\Shashwat\PhysioFM_Codebase\installation_log.md", 'a', encoding='utf-8') as f:
    f.write("\n## Phase 0 - Requirements Installation (Continued)\n\n```\n")
    f.write(log_content)
    f.write("\n```\n")
