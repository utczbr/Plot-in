from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

class BasePipeline(ABC):
    """
    Abstract base class for analysis pipelines.
    Defines the standard interface for executing analysis tasks.
    """
    
    def __init__(self, context: Optional[Any] = None):
        """
        Initialize the pipeline.
        
        Args:
            context: Optional application context or configuration object
        """
        self.context = context
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Any:
        """
        Execute the pipeline on a single input.
        
        Args:
            input_data: The input to process (e.g., file path, image array)
            **kwargs: Additional execution parameters
            
        Returns:
            The processing result
        """
        pass

