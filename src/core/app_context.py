"""
Application Context - Facade for accessing application services
"""

from typing import Optional, Any, Dict
from pathlib import Path
import logging

from services.service_container import ServiceContainer, create_service_container

logger = logging.getLogger(__name__)


class ApplicationContext:
    """
    Application-wide context providing unified access to all services.
    Acts as a facade over the service container.
    """
    
    _instance: Optional['ApplicationContext'] = None
    _container: Optional[ServiceContainer] = None
    
    def __init__(self, container: Optional[ServiceContainer] = None):
        """
        Initialize application context.
        
        Args:
            container: Optional pre-configured service container
        """
        if container is None:
            container = create_service_container()
        self._container = container
        ApplicationContext._instance = self
    
    @classmethod
    def get_instance(cls) -> 'ApplicationContext':
        """Get singleton instance of application context."""
        if cls._instance is None:
            cls._instance = ApplicationContext()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance (for testing)."""
        if cls._instance and cls._instance._container:
            cls._instance._container.reset()
        cls._instance = None
    
    # Core Services
    
    @property
    def data_manager(self):
        """Get data manager service."""
        return self._container.get('data_manager')
    
    @property
    def image_manager(self):
        """Get image manager service."""
        return self._container.get('image_manager')
    
    @property
    def analysis_manager(self):
        """Get analysis manager service."""
        return self._container.get('analysis_manager')
    
    @property
    def export_manager(self):
        """Get export manager service."""
        return self._container.get('export_manager')
    
    @property
    def thread_safety(self):
        """Get thread safety manager service."""
        return self._container.get('thread_safety')
    
    @property
    def state_manager(self):
        """Get state manager service."""
        return self._container.get('state_manager')
    
    # Backend Services
    
    @property
    def model_manager(self):
        """Get model manager service."""
        return self._container.get('model_manager')
    
    @property
    def orchestrator(self):
        """Get chart analysis orchestrator."""
        return self._container.get('orchestrator')
    
    @property
    def calibration_service(self):
        """Get calibration service (creates new instance each time)."""
        return self._container.get('calibration_service')
    
    # UI Services
    
    @property
    def visualization_service(self):
        """Get visualization service."""
        return self._container.get('visualization_service')
    
    @property
    def app_config(self):
        """Get application configuration."""
        return self._container.get('app_config')
    
    # Configuration
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """Get configuration value."""
        return self._container.get_config(key, default)
    
    def set_config(self, config: Dict[str, Any]):
        """Set application configuration."""
        self._container.set_config(config)
    
    # Service Access
    
    def get_service(self, name: str) -> Any:
        """Get any service by name."""
        return self._container.get(name)
    
    def has_service(self, name: str) -> bool:
        """Check if service is available."""
        return self._container.has(name)