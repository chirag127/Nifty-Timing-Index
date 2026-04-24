"""Multi-Provider Email Sender with Fallback Chain.

Supports 4+ email providers. Tries providers in order, falling back to the next on failure.
All providers configurable via .env with enable/disable toggles.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import httpx

from nti.config.settings import settings

logger = logging.getLogger(__name__)


class MultiEmailSender:
    """Email sender that tries multiple providers with fallback.

    Usage:
        sender = MultiEmailSender()
        sender.send_alert("Subject", "<h1>HTML body</h1>")
    """

    def send_alert(self, subject: str, html_body: str) -> bool:
        """Send an email alert using the first available provider.

        Returns True if email was sent successfully, False otherwise.
        """
        if not settings.enable_email:
            logger.info("Email disabled via NTI_ENABLE_EMAIL=false")
            return False

        if not settings.alert_email_to:
            logger.warning("No ALERT_EMAIL_TO configured")
            return False

        for provider in settings.get_enabled_email_providers():
            try:
                success = self._send_via_provider(provider, subject, html_body)
                if success:
                    logger.info(f"Email sent via {provider.name}")
                    return True
            except Exception as e:
                logger.warning(f"Email provider {provider.name} failed: {e}. Trying next...")
                continue

        logger.error("All email providers failed")
        return False

    def _send_via_provider(self, provider, subject: str, html_body: str) -> bool:
        """Send email via a specific provider."""
        match provider.name:
            case "gmail":
                return self._send_gmail(provider, subject, html_body)
            case "resend":
                return self._send_resend(provider, subject, html_body)
            case "brevo":
                return self._send_brevo(provider, subject, html_body)
            case "sendgrid":
                return self._send_sendgrid(provider, subject, html_body)
            case _:
                logger.warning(f"Unknown email provider: {provider.name}")
                return False

    def _send_gmail(self, provider, subject: str, html_body: str) -> bool:
        """Send via Gmail SMTP (free, 500 emails/day)."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = provider.from_address
        msg["To"] = settings.alert_email_to
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(provider.smtp_host, provider.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(provider.from_address, provider.api_key)  # api_key = app password
            server.sendmail(provider.from_address, settings.alert_email_to, msg.as_string())

        return True

    def _send_resend(self, provider, subject: str, html_body: str) -> bool:
        """Send via Resend API (free, 100 emails/day)."""
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{provider.base_url}/emails",
                headers={
                    "Authorization": f"Bearer {provider.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": provider.from_address,
                    "to": [settings.alert_email_to],
                    "subject": subject,
                    "html": html_body,
                },
            )
            resp.raise_for_status()
        return True

    def _send_brevo(self, provider, subject: str, html_body: str) -> bool:
        """Send via Brevo/Sendinblue API (free, 300 emails/day)."""
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{provider.base_url}/smtp/email",
                headers={
                    "api-key": provider.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "sender": {"email": provider.from_address},
                    "to": [{"email": settings.alert_email_to}],
                    "subject": subject,
                    "htmlContent": html_body,
                },
            )
            resp.raise_for_status()
        return True

    def _send_sendgrid(self, provider, subject: str, html_body: str) -> bool:
        """Send via SendGrid API (free, 100 emails/day)."""
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{provider.base_url}/mail/send",
                headers={
                    "Authorization": f"Bearer {provider.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "personalizations": [{"to": [{"email": settings.alert_email_to}]}],
                    "from": {"email": provider.from_address},
                    "subject": subject,
                    "content": [{"type": "text/html", "value": html_body}],
                },
            )
            resp.raise_for_status()
        return True


# Singleton
email_sender = MultiEmailSender()


def send_zone_change_alert(
    from_zone: str,
    to_zone: str,
    score: float,
    nifty_price: float,
    drivers: list[str],
) -> bool:
    """Send a zone change alert email."""
    subject = f"⚠️ NTI ZONE CHANGE: {from_zone} → {to_zone} | Nifty: {nifty_price:,.0f}"

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #e74c3c;">🔴 NTI Zone Change Alert</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 8px;"><strong>Previous Zone:</strong></td><td>{from_zone}</td></tr>
            <tr><td style="padding: 8px;"><strong>New Zone:</strong></td><td style="color: #e74c3c; font-weight: bold;">{to_zone}</td></tr>
            <tr><td style="padding: 8px;"><strong>NTI Score:</strong></td><td>{score:.1f}/100</td></tr>
            <tr><td style="padding: 8px;"><strong>Nifty 50:</strong></td><td>{nifty_price:,.0f}</td></tr>
            <tr><td style="padding: 8px;"><strong>Top Drivers:</strong></td><td>{', '.join(drivers)}</td></tr>
        </table>
        <p style="color: #666; font-size: 12px; margin-top: 20px;">
            This is not investment advice. NTI is a personal market analysis tool.
        </p>
    </div>
    """
    return email_sender.send_alert(subject, html)


def send_big_move_alert(
    prev_score: float,
    curr_score: float,
    nifty_price: float,
) -> bool:
    """Send a big move alert email."""
    diff = curr_score - prev_score
    direction = "↑" if diff > 0 else "↓"
    subject = f"📈 NTI BIG MOVE: Score {prev_score:.0f} → {curr_score:.0f} ({direction}{abs(diff):.0f} pts)"

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2>📈 NTI Big Move Alert</h2>
        <p>NTI score moved <strong>{direction}{abs(diff):.1f} points</strong> in the last hour.</p>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 8px;">Previous Score:</td><td>{prev_score:.1f}</td></tr>
            <tr><td style="padding: 8px;">Current Score:</td><td>{curr_score:.1f}</td></tr>
            <tr><td style="padding: 8px;">Nifty 50:</td><td>{nifty_price:,.0f}</td></tr>
        </table>
    </div>
    """
    return email_sender.send_alert(subject, html)
