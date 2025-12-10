
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
from pathlib import Path

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

    def run_batch(self, inputs: List[Any], **kwargs) -> List[Any]:
        """
        Execute the pipeline on a batch of inputs.
        
        Args:
            inputs: List of inputs to process
            **kwargs: Additional execution parameters
            
        Returns:
            List of processing results
        """
        results = []
        for item in inputs:
            try:
                result = self.run(item, **kwargs)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing item {item}: {e}")
        return results
