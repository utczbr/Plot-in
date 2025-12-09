"""
Service Container - Dependency Injection Container for Chart Analysis Application
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
import logging
import easyocr

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Dependency injection container for managing service lifecycle and dependencies.
    Implements the Service Locator pattern with lazy initialization.
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
    
    def register(self, name: str, factory: Callable, singleton: bool = True):
        """
        Register a service factory.
        
        Args:
            name: Service identifier
            factory: Callable that creates the service instance
            singleton: If True, only one instance is created
        """
        self._factories[name] = factory
        if singleton:
            self._singletons[name] = None
        logger.debug(f"Registered service: {name} (singleton={singleton})")
    
    def register_instance(self, name: str, instance: Any):
        """Register a pre-existing service instance."""
        self._services[name] = instance
        logger.debug(f"Registered instance: {name}")
    
    def get(self, name: str) -> Any:
        """
        Retrieve a service by name.
        
        Args:
            name: Service identifier
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service is not registered
        """
        # Check if already instantiated
        if name in self._services:
            return self._services[name]
        
        # Check if singleton already exists
        if name in self._singletons and self._singletons[name] is not None:
            return self._singletons[name]
        
        # Create new instance from factory
        if name not in self._factories:
            raise KeyError(f"Service '{name}' not registered")
        
        factory = self._factories[name]
        instance = factory(self)  # Pass container to factory for dependency resolution
        
        # Store singleton or transient instance
        if name in self._singletons:
            self._singletons[name] = instance
            self._services[name] = instance
        
        return instance
    
    def set_config(self, config: Dict[str, Any]):
        """Set application configuration."""
        self._config = config
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """Get configuration value."""
        if key is None:
            return self._config
        return self._config.get(key, default)
    
    def has(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self._factories or name in self._services
    
    def reset(self):
        """Reset all services (useful for testing)."""
        self._services.clear()
        self._singletons = {k: None for k in self._singletons.keys()}
        logger.debug("Service container reset")


def create_service_container(config_path: Optional[Path] = None) -> ServiceContainer:
    """
    Factory function to create and configure the service container.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured ServiceContainer instance
    """
    container = ServiceContainer()
    
    # Load configuration
    if config_path and config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        container.set_config(config)
    
    # Register core services
    _register_core_services(container)
    _register_backend_services(container)
    _register_ui_services(container)
    
    return container


def _register_core_services(container: ServiceContainer):
    """Register core data management services."""
    from core.data_manager import DataManager
    from core.image_manager import ImageManager
    from core.analysis_manager import AnalysisManager
    from core.export_manager import ExportManager
    from core.thread_safety import ThreadSafetyManager
    from core.app_state import StateManager
    
    container.register(
        'data_manager',
        lambda c: DataManager(),
        singleton=True
    )
    
    container.register(
        'image_manager',
        lambda c: ImageManager(c.get('data_manager')),
        singleton=True
    )
    
    container.register(
        'analysis_manager',
        lambda c: AnalysisManager(
            c.get('data_manager'),
            c.get('image_manager')
        ),
        singleton=True
    )
    
    container.register(
        'export_manager',
        lambda c: ExportManager(),
        singleton=True
    )
    
    # NEW: Thread safety and state management
    container.register(
        'thread_safety',
        lambda c: ThreadSafetyManager(max_concurrent_analyses=4),
        singleton=True
    )
    
    container.register(
        'state_manager',
        lambda c: StateManager(),
        singleton=True
    )


def _register_backend_services(container: ServiceContainer):
    """Register backend analysis services."""
    from core.model_manager import ModelManager
    from calibration.calibration_factory import CalibrationFactory
    from services.orientation_service import OrientationService
    from services.orientation_detection_service import OrientationDetectionService
    from services.dual_axis_service import DualAxisDetectionService
    from services.meta_clustering_service import MetaClusteringService
    from ChartAnalysisOrchestrator import ChartAnalysisOrchestrator
    
    container.register(
        'model_manager',
        lambda c: ModelManager(),
        singleton=True
    )
    
    container.register(
        'calibration_service',
        lambda c: CalibrationFactory.create(
            c.get_config('calibration_method', 'PROSAC')
        ),
        singleton=False  # Create new instances as needed
    )
    
    container.register(
        'orientation_service',
        lambda c: OrientationService(),
        singleton=True
    )
    
    container.register(
        'orientation_detector',
        lambda c: OrientationDetectionService(),
        singleton=True
    )
    
    container.register(
        'dual_axis_service',
        lambda c: DualAxisDetectionService(),
        singleton=True
    )
    
    container.register(
        'meta_clustering_service',
        lambda c: MetaClusteringService(),
        singleton=True
    )
    
    container.register(
        'orchestrator',
        lambda c: ChartAnalysisOrchestrator(
            calibration_service=c.get('calibration_service'),
            logger=logging.getLogger('Orchestrator')
        ),
        singleton=True
    )

    container.register(
        'easyocr_reader',
        lambda c: easyocr.Reader(['en'], gpu=c.get_config('ocr', {}).get('use_gpu', False)),
        singleton=True
    )


def _register_ui_services(container: ServiceContainer):
    """Register UI-related services."""
    from visual.visualization_service import VisualizationService
    from visual.config_manager import AppConfig
    
    container.register(
        'visualization_service',
        lambda c: VisualizationService,  # Static class, no instantiation needed
        singleton=True
    )
    
    container.register(
        'app_config',
        lambda c: AppConfig(),
        singleton=True
    )