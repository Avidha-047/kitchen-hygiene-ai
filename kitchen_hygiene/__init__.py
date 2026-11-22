"""AI-powered kitchen hygiene monitoring: detects hygiene violations in CCTV video and explains each alert."""
from .config import Config, SEVERITY
from .engine import HygieneEngine, Finding, Event
from .bayes import p_soap

__all__ = ["Config", "SEVERITY", "HygieneEngine", "Finding", "Event", "p_soap"]
