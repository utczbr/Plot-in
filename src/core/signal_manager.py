"""
Signal Connection Manager - Track Signal Connections for Guaranteed Cleanup

This module provides tracking of Qt signal connections to prevent memory leaks
from lambda closures and orphaned connections (identified in main_modern.py).

Author: Chart Analysis Tool Team
Date: November 24, 2025
"""

from PyQt6.QtCore import QObject
from typing import List, Tuple, Any, Callable
import logging

logger = logging.getLogger(__name__)


class SignalConnectionManager(QObject):
    """
    Track signal connections for guaranteed cleanup.
    
    Prevents memory leaks from:
    - Lambda closures capturing 'self' (circular references)
    - Orphaned connections when widgets are deleted
    - Signal connections not disconnected on cleanup
    
    Example:
        >>> manager = SignalConnectionManager()
        >>> manager.connect(button.clicked, self.on_click)
        >>> manager.connect(slider.valueChanged, self.on_value_changed)
        >>> # On cleanup:
        >>> manager.disconnect_all()
    """
    
    def __init__(self):
        """Initialize signal connection manager."""
        super().__init__()
        self._connections: List[Tuple[Any, Callable]] = []
        self._blocked = False
        logger.debug("SignalConnectionManager initialized")
    
    def connect(self, signal, slot: Callable) -> Any:
        """
        Connect signal to slot and track for cleanup.
        
        Args:
            signal: Qt signal to connect
            slot: Callable slot (method or function)
        
        Returns:
            The signal (for chaining)
        
        Example:
            >>> manager.connect(button.clicked, self.on_button_click)
            >>> manager.connect(checkbox.stateChanged, lambda state: print(state))
        """
        signal.connect(slot)
        self._connections.append((signal, slot))
        logger.debug(f"Connected signal (total: {len(self._connections)})")
        return signal
    
    def disconnect(self, signal, slot: Callable) -> bool:
        """
        Disconnect specific signal-slot pair.
        
        Args:
            signal: Qt signal
            slot: Callable slot
        
        Returns:
            True if disconnected, False if not found
        """
        try:
            signal.disconnect(slot)
            self._connections.remove((signal, slot))
            logger.debug(f"Disconnected signal (remaining: {len(self._connections)})")
            return True
        except (TypeError, RuntimeError, ValueError):
            logger.warning("Signal already disconnected or not found")
            return False
    
    def disconnect_all(self) -> int:
        """
        Disconnect all tracked signals.
        
        Call this in closeEvent() or widget cleanup to prevent memory leaks.
        
        Returns:
            Number of connections disconnected
        
        Example:
            >>> def closeEvent(self, event):
            ...     self.signal_manager.disconnect_all()
            ...     event.accept()
        """
        disconnected = 0
        for signal, slot in self._connections:
            try:
                signal.disconnect(slot)
                disconnected += 1
            except (TypeError, RuntimeError):
                # Already disconnected or object deleted
                pass
        
        self._connections.clear()
        logger.info(f"Disconnected {disconnected} signal connections")
        return disconnected
    
    def block_all(self) -> None:
        """
        Block all tracked signals (for batch updates).
        
        Use this to prevent signals from firing during bulk operations.
        
        Example:
            >>> manager.block_all()
            >>> for i in range(100):
            ...     slider.setValue(i)  # No signals fired
            >>> manager.unblock_all()  # Signals resume
        """
        if self._blocked:
            return
        
        for signal, _ in self._connections:
            # Get the sender QObject if available
            sender = getattr(signal, '__self__', None)
            if sender and hasattr(sender, 'blockSignals'):
                sender.blockSignals(True)
        
        self._blocked = True
        logger.debug("All signals blocked")
    
    def unblock_all(self) -> None:
        """
        Unblock all tracked signals.
        
        Resumes signal emission after block_all().
        """
        if not self._blocked:
            return
        
        for signal, _ in self._connections:
            sender = getattr(signal, '__self__', None)
            if sender and hasattr(sender, 'blockSignals'):
                sender.blockSignals(False)
        
        self._blocked = False
        logger.debug("All signals unblocked")
    
    def get_connection_count(self) -> int:
        """Get number of tracked connections."""
        return len(self._connections)
    
    def clear(self) -> None:
        """
        Clear all tracked connections without disconnecting.
        
        Use this if signals were already disconnected externally.
        """
        self._connections.clear()
        logger.debug("Connection tracking cleared")


def safe_connect(signal, slot: Callable, manager: SignalConnectionManager = None) -> None:
    """
    Helper function for safe signal connection with optional tracking.
    
    Args:
        signal: Qt signal
        slot: Callable slot
        manager: Optional SignalConnectionManager for tracking
    
    Example:
        >>> safe_connect(button.clicked, on_click, manager)
        >>> # Or without manager:
        >>> safe_connect(button.clicked, on_click)
    """
    if manager:
        manager.connect(signal, slot)
    else:
        signal.connect(slot)
