"""Gate 2 human-confirmation emails — SecureEye Crisis Command."""
from __future__ import annotations

import logging
import os
import smtplib
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# Explicit path so dotenv finds .env regardless of working directory
_ENV_FILE = Path(__file__).resolve().parent / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=True)
except ImportError:
    pass

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("CRISIS_CONFIRM_URL", "http://localhost:8000")


def _creds() -> tuple[str, str]:
    """Read credentials fresh each call so dotenv/env changes are always picked up."""
    return (
        os.getenv("CRISIS_GMAIL_USER", "").strip(),
        os.getenv("CRISIS_GMAIL_PASSWORD", "").strip(),
    )


def _fire_to() -> str:
    return os.getenv("CRISIS_FIRE_AUTHORITY_EMAIL", "").strip()


def _fall_to() -> str:
    return os.getenv("CRISIS_FALL_AUTHORITY_EMAIL", "").strip()


def _send(to: str, subject: str, body_html: str,
          attachments: list[tuple[str, bytes, str]] | None = None) -> tuple[bool, str]:
    """Send via Gmail SMTP. Returns (success, error_detail)."""
    gmail_user, gmail_pass = _creds()

    if not gmail_user:
        return False, "CRISIS_GMAIL_USER not set in .env"
    if not gmail_pass:
        return False, "CRISIS_GMAIL_PASSWORD not set in .env"
    if not to:
        return False, "Recipient email not set in .env"

    msg = MIMEMultipart("mixed")
    msg["From"]    = gmail_user
    msg["To"]      = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body_html, "html"))

    for fname, data, mime in (attachments or []):
        part = MIMEBase(*mime.split("/", 1))
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
        msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as srv:
            srv.login(gmail_user, gmail_pass)
            srv.sendmail(gmail_user, [to], msg.as_string())
        logger.info("Gate 2 email sent → %s: %s", to, subject)
        return True, "ok"
    except smtplib.SMTPAuthenticationError:
        return False, "SMTP auth failed — check app password in .env"
    except smtplib.SMTPException as exc:
        return False, f"SMTP error: {exc}"
    except Exception as exc:
        return False, f"Unexpected error: {exc}"


def test_email() -> tuple[bool, str]:
    """Send a quick test email to both configured recipients. Returns (ok, detail)."""
    u, p = _creds()
    fire_to = _fire_to()
    fall_to = _fall_to()
    details = [
        f"CRISIS_GMAIL_USER={u or '(empty)'}",
        f"CRISIS_GMAIL_PASSWORD={'*set*' if p else '(empty)'}",
        f"CRISIS_FIRE_AUTHORITY_EMAIL={fire_to or '(empty)'}",
        f"CRISIS_FALL_AUTHORITY_EMAIL={fall_to or '(empty)'}",
        f".env file found: {_ENV_FILE.exists()}",
    ]
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    body = f"""
<html><body style="font-family:sans-serif;background:#0f1724;color:#e8eef9;padding:24px;max-width:500px">
  <h2 style="color:#3BF0A4">✓ SecureEye Gate 2 — Test Email</h2>
  <p style="color:#94a3b8">If you received this, the email pipeline is working correctly.</p>
  <p style="color:#64748b;font-size:12px">{ts}</p>
</body></html>"""

    results = []
    for to in {fire_to, fall_to} - {""}:
        ok, err = _send(to, "[SecureEye] Gate 2 Test Email", body)
        results.append(f"{to}: {'OK' if ok else err}")

    detail = " | ".join(details) + " || Results: " + ("; ".join(results) if results else "no recipients configured")
    return all("OK" in r for r in results) if results else False, detail


def send_fire_review(
    event_id: str,
    camera_id: str,
    confidence: float,
    severity: str,
    clip_bytes: bytes | None = None,
) -> tuple[bool, str]:
    base     = BASE_URL
    dispatch = f"{base}/gate2/response?id={event_id}&action=dispatch_fire"
    onsite   = f"{base}/gate2/response?id={event_id}&action=onsite_team"
    dismiss  = f"{base}/gate2/response?id={event_id}&action=false_alarm"
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    subject = f"[GATE 2 REVIEW] Fire Detected — {camera_id} | {confidence:.0%} confidence"
    body = f"""
<html><body style="font-family:sans-serif;background:#0f1724;color:#e8eef9;padding:24px;max-width:620px;margin:0 auto">
<div style="border-left:4px solid #ff6600;padding-left:16px;margin-bottom:24px">
  <h2 style="color:#ff6600;margin:0 0 4px">🔥 Fire Event — Gate 2 Human Review</h2>
  <p style="color:#94a3b8;margin:0">Gate 1 AI analysis passed. Your confirmation is required.</p>
</div>
<table style="width:100%;border-collapse:collapse;margin-bottom:24px;background:#131d2e;border-radius:8px;overflow:hidden">
  <tr style="border-bottom:1px solid #1e2d3d"><td style="padding:10px 14px;color:#64748b;width:140px">Camera</td><td style="padding:10px 14px">{camera_id}</td></tr>
  <tr style="border-bottom:1px solid #1e2d3d"><td style="padding:10px 14px;color:#64748b">Confidence</td><td style="padding:10px 14px;color:#ff8800;font-weight:bold">{confidence:.1%}</td></tr>
  <tr style="border-bottom:1px solid #1e2d3d"><td style="padding:10px 14px;color:#64748b">Severity</td><td style="padding:10px 14px;color:#ff4444">{severity}</td></tr>
  <tr style="border-bottom:1px solid #1e2d3d"><td style="padding:10px 14px;color:#64748b">Event ID</td><td style="padding:10px 14px;font-family:monospace;font-size:12px">{event_id}</td></tr>
  <tr><td style="padding:10px 14px;color:#64748b">Detected At</td><td style="padding:10px 14px">{ts}</td></tr>
</table>
<p style="color:#94a3b8;font-size:13px;margin-bottom:20px">The 7-second event clip is attached. Review it and select one action:</p>
<div style="display:flex;flex-direction:column;gap:10px">
  <a href="{dispatch}" style="display:inline-block;background:#dc2626;color:white;padding:13px 28px;border-radius:6px;text-decoration:none;font-weight:bold;font-size:15px">🚒 Confirm Emergency — Dispatch Fire Engine</a>
  <a href="{onsite}"  style="display:inline-block;background:#d97706;color:white;padding:13px 28px;border-radius:6px;text-decoration:none;font-weight:bold;font-size:15px">🏃 On-site Team Will Handle</a>
  <a href="{dismiss}" style="display:inline-block;background:#334155;color:white;padding:13px 28px;border-radius:6px;text-decoration:none;font-weight:bold;font-size:15px">✗ Mark as False Alarm</a>
</div>
<p style="color:#334155;font-size:11px;border-top:1px solid #1e2d3d;padding-top:16px;margin-top:32px">
  SecureEye Crisis Command &nbsp;·&nbsp; Gate 2 Human Confirmation &nbsp;·&nbsp; {ts}
</p>
</body></html>
"""
    attachments = [(f"fire_clip_{event_id}.mp4", clip_bytes, "video/mp4")] if clip_bytes else []
    return _send(_fire_to(), subject, body, attachments)


def send_fall_review(
    event_id: str,
    camera_id: str,
    confidence: float,
    clip_bytes: bytes | None = None,
) -> tuple[bool, str]:
    base    = BASE_URL
    confirm = f"{base}/gate2/response?id={event_id}&action=dispatch_ambulance"
    onsite  = f"{base}/gate2/response?id={event_id}&action=onsite_team"
    dismiss = f"{base}/gate2/response?id={event_id}&action=false_alarm"
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    subject = f"[GATE 2 REVIEW] Fall Detected — {camera_id} | {confidence:.0%} confidence"
    body = f"""
<html><body style="font-family:sans-serif;background:#0f1724;color:#e8eef9;padding:24px;max-width:620px;margin:0 auto">
<div style="border-left:4px solid #5AA9FF;padding-left:16px;margin-bottom:24px">
  <h2 style="color:#5AA9FF;margin:0 0 4px">🚑 Fall Event — Gate 2 Human Review</h2>
  <p style="color:#94a3b8;margin:0">AI monitored for 3 seconds — person did not appear to recover.</p>
</div>
<table style="width:100%;border-collapse:collapse;margin-bottom:24px;background:#131d2e;border-radius:8px;overflow:hidden">
  <tr style="border-bottom:1px solid #1e2d3d"><td style="padding:10px 14px;color:#64748b;width:140px">Camera</td><td style="padding:10px 14px">{camera_id}</td></tr>
  <tr style="border-bottom:1px solid #1e2d3d"><td style="padding:10px 14px;color:#64748b">Confidence</td><td style="padding:10px 14px;color:#5AA9FF;font-weight:bold">{confidence:.1%}</td></tr>
  <tr style="border-bottom:1px solid #1e2d3d"><td style="padding:10px 14px;color:#64748b">Event ID</td><td style="padding:10px 14px;font-family:monospace;font-size:12px">{event_id}</td></tr>
  <tr><td style="padding:10px 14px;color:#64748b">Detected At</td><td style="padding:10px 14px">{ts}</td></tr>
</table>
<p style="color:#94a3b8;font-size:13px;margin-bottom:20px">The 7-second event clip is attached. Review it and select one action:</p>
<div style="display:flex;flex-direction:column;gap:10px">
  <a href="{confirm}" style="display:inline-block;background:#2563eb;color:white;padding:13px 28px;border-radius:6px;text-decoration:none;font-weight:bold;font-size:15px">🚑 Confirm Emergency — Dispatch Ambulance</a>
  <a href="{onsite}"  style="display:inline-block;background:#d97706;color:white;padding:13px 28px;border-radius:6px;text-decoration:none;font-weight:bold;font-size:15px">🏃 On-site Team Will Handle</a>
  <a href="{dismiss}" style="display:inline-block;background:#334155;color:white;padding:13px 28px;border-radius:6px;text-decoration:none;font-weight:bold;font-size:15px">✗ Mark as False Alarm</a>
</div>
<p style="color:#334155;font-size:11px;border-top:1px solid #1e2d3d;padding-top:16px;margin-top:32px">
  SecureEye Crisis Command &nbsp;·&nbsp; Gate 2 Human Confirmation &nbsp;·&nbsp; {ts}
</p>
</body></html>
"""
    attachments = [(f"fall_clip_{event_id}.mp4", clip_bytes, "video/mp4")] if clip_bytes else []
    return _send(_fall_to(), subject, body, attachments)
