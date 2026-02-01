import logging
import os
from typing import Any, Optional

import requests
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)


def send_line_notification(message: str, token: Optional[str] = None):
    """
    Sends a notification to LINE via the Messaging API (Broadcast).
    Requires a Channel Access Token from a LINE Official Account.
    Can be passed as an argument or set via LINE_CHANNEL_ACCESS_TOKEN env var.
    """
    # Prefer argument, fallback to environment variable
    raw_token = token or os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

    if not raw_token:
        logger.warning("LINE Channel Access Token is missing. Notification not sent.")
        return

    # Clean the token (strip whitespace and potential surrounding quotes)
    line_token = raw_token.strip().strip('"').strip("'")

    # Debug info (Safe preview)
    token_preview = f"{line_token[:5]}...{line_token[-5:]}" if len(line_token) > 10 else "too short"
    logger.info(
        f"Attempting LINE notification. Token length: {len(line_token)}, Preview: {token_preview}"
    )

    # New Messaging API endpoint for broadcasting
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {line_token}"}

    # Messaging API requires a specific JSON format
    payload = {"messages": [{"type": "text", "text": message}]}

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            logger.info("LINE Messaging API notification sent successfully.")
        else:
            logger.error(
                f"Failed to send LINE notification: {response.status_code} - {response.text}"
            )
    except Exception as e:
        logger.error(f"Error sending LINE notification: {e}")


def notify_job_status(
    status: str,
    task_name: str,
    fm_name: str,
    prep_name: str,
    score_or_error: Any,
    file_id: str,
    notifications_cfg: Any,
):
    """
    High-level notification helper that handles message formatting and sweep progress.
    """
    if not notifications_cfg or not getattr(notifications_cfg, "enabled", False):
        return

    # 1. Try to get job info for sweep progress from Hydra
    job_info = ""
    try:
        hydra_cfg = HydraConfig.get()
        job_cfg = getattr(hydra_cfg, "job", None)
        if job_cfg:
            job_num = job_cfg.num + 1
            multirun_cfg = getattr(hydra_cfg, "multirun", None)
            total = getattr(multirun_cfg, "total", None) if multirun_cfg else None

            if total:
                job_info = f"Job {job_num} of {total}\n"
            else:
                job_info = f"Job #{job_num}\n"
    except Exception:
        # Silently fail if HydraConfig isn't initialized/available in this context
        pass

    # 2. Format Message
    title = getattr(notifications_cfg, "title", "QCNN Job")
    if status.lower() == "success":
        icon = "✅"
        label = "Finished"
        val_label = "Score"
    else:
        icon = "❌"
        label = "Failed"
        val_label = "Error"
        # Truncate error if it's too long
        score_or_error = (
            str(score_or_error)[:100] + "..." if len(str(score_or_error)) > 100 else score_or_error
        )

    message = (
        f"\n[{title}] {icon} {label}!\n"
        f"{job_info}"
        f"Task: {task_name}\n"
        f"Config: {fm_name} | {prep_name}\n"
        f"{val_label}: {score_or_error}\n"
        f"ID: {file_id}"
    )

    # 3. Send
    token = getattr(notifications_cfg, "line_token", None)
    send_line_notification(message, token)
