"""
Metrics calculation algorithms for RMCP
Implements EMA (Exponential Moving Average) and P-Square algorithms for dynamic metrics
"""

import math
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MetricSnapshot:
    """Snapshot of a metric at a point in time"""
    value: float
    timestamp: float
    sample_count: int


class EMACalculator:
    """
    Exponential Moving Average calculator
    
    Used for calculating smooth, responsive averages that give more weight
    to recent observations while still considering historical data.
    """
    
    def __init__(self, alpha: float = 0.1, initial_value: Optional[float] = None):
        """
        Initialize EMA calculator
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
                  - Lower values = more smoothing, slower response
                  - Higher values = less smoothing, faster response
            initial_value: Initial value for the EMA
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        self.alpha = alpha
        self.ema_value = initial_value
        self.sample_count = 0
        self.last_update_time = None
    
    def update(self, value: float, timestamp: Optional[float] = None) -> float:
        """
        Update EMA with new value
        
        Args:
            value: New observation value
            timestamp: Timestamp of the observation (optional)
            
        Returns:
            Updated EMA value
        """
        if timestamp is None:
            import time
            timestamp = time.time()
        
        if self.ema_value is None:
            # First observation
            self.ema_value = value
        else:
            # EMA formula: EMA_new = alpha * value + (1 - alpha) * EMA_old
            self.ema_value = self.alpha * value + (1 - self.alpha) * self.ema_value
        
        self.sample_count += 1
        self.last_update_time = timestamp
        
        return self.ema_value
    
    def get_value(self) -> Optional[float]:
        """Get current EMA value"""
        return self.ema_value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get EMA statistics"""
        return {
            "ema_value": self.ema_value,
            "alpha": self.alpha,
            "sample_count": self.sample_count,
            "last_update_time": self.last_update_time
        }
    
    def reset(self, initial_value: Optional[float] = None) -> None:
        """Reset EMA calculator"""
        self.ema_value = initial_value
        self.sample_count = 0
        self.last_update_time = None


class PSquareCalculator:
    """
    P-Square algorithm for calculating percentiles
    
    P-Square is an online algorithm for calculating percentiles without
    storing all observations. It maintains a small number of markers
    that approximate the desired percentile.
    """
    
    def __init__(self, percentile: float = 95.0):
        """
        Initialize P-Square calculator
        
        Args:
            percentile: Target percentile (0-100)
        """
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        self.percentile = percentile
        self.p = percentile / 100.0
        
        # P-Square markers (5 markers for percentiles)
        self.markers = [0.0] * 5
        self.marker_positions = [0, 1, 2, 3, 4]
        self.marker_desired_positions = [0, 1, 2, 3, 4]
        
        # Sample count
        self.sample_count = 0
        self.last_update_time = None
        
        # Initialization flag
        self.initialized = False
    
    def update(self, value: float, timestamp: Optional[float] = None) -> float:
        """
        Update P-Square with new value
        
        Args:
            value: New observation value
            timestamp: Timestamp of the observation (optional)
            
        Returns:
            Current percentile estimate
        """
        if timestamp is None:
            import time
            timestamp = time.time()
        
        self.sample_count += 1
        self.last_update_time = timestamp
        
        if not self.initialized:
            # Initialize with first 5 values
            if self.sample_count <= 5:
                self.markers[self.sample_count - 1] = value
                if self.sample_count == 5:
                    self._initialize_markers()
                    self.initialized = True
            return value
        
        # Find the appropriate marker to update
        marker_index = self._find_marker_index(value)
        
        # Update marker positions
        for i in range(marker_index + 1, 5):
            self.marker_positions[i] += 1
        
        # Update desired positions
        for i in range(5):
            self.marker_desired_positions[i] = self._calculate_desired_position(i)
        
        # Update the marker value
        self._update_marker_value(marker_index, value)
        
        # Adjust markers if needed
        self._adjust_markers()
        
        # Return the percentile estimate (marker 2 for 95th percentile)
        return self.markers[2]
    
    def _initialize_markers(self) -> None:
        """Initialize P-Square markers after first 5 observations"""
        # Sort the initial values
        sorted_values = sorted(self.markers)
        
        # Assign sorted values to markers
        for i in range(5):
            self.markers[i] = sorted_values[i]
        
        # Calculate desired positions
        for i in range(5):
            self.marker_desired_positions[i] = self._calculate_desired_position(i)
    
    def _find_marker_index(self, value: float) -> int:
        """Find the appropriate marker index for a new value"""
        if value < self.markers[0]:
            return 0
        elif value >= self.markers[4]:
            return 4
        else:
            for i in range(4):
                if self.markers[i] <= value < self.markers[i + 1]:
                    return i
        return 2  # Default to middle marker
    
    def _calculate_desired_position(self, marker_index: int) -> float:
        """Calculate desired position for a marker"""
        if marker_index == 0:
            return 1
        elif marker_index == 4:
            return self.sample_count
        else:
            return 1 + marker_index * (self.sample_count - 1) * self.p
    
    def _update_marker_value(self, marker_index: int, value: float) -> None:
        """Update marker value using parabolic interpolation"""
        if marker_index == 0 or marker_index == 4:
            self.markers[marker_index] = value
        else:
            # Parabolic interpolation
            n = self.sample_count
            q = self.p
            
            # Calculate interpolation parameters
            d = self.marker_desired_positions[marker_index] - self.marker_positions[marker_index]
            
            if abs(d) >= 1:
                # Use parabolic interpolation
                q1 = self.markers[marker_index - 1]
                q2 = self.markers[marker_index]
                q3 = self.markers[marker_index + 1]
                
                n1 = self.marker_positions[marker_index - 1]
                n2 = self.marker_positions[marker_index]
                n3 = self.marker_positions[marker_index + 1]
                
                # Parabolic interpolation formula
                numerator = d * ((n2 - n1 + d) * (q3 - q2) / (n3 - n2) + (n3 - n2 - d) * (q2 - q1) / (n2 - n1))
                denominator = n3 - n1
                
                if denominator != 0:
                    self.markers[marker_index] = q2 + numerator / denominator
                else:
                    self.markers[marker_index] = value
            else:
                self.markers[marker_index] = value
    
    def _adjust_markers(self) -> None:
        """Adjust marker positions if they deviate too much from desired positions"""
        for i in range(1, 4):  # Don't adjust boundary markers
            desired = self.marker_desired_positions[i]
            actual = self.marker_positions[i]
            
            if abs(desired - actual) >= 1:
                # Adjust marker position
                if desired > actual:
                    self.marker_positions[i] += 1
                else:
                    self.marker_positions[i] -= 1
    
    def get_percentile(self) -> Optional[float]:
        """Get current percentile estimate"""
        if not self.initialized:
            return None
        return self.markers[2]  # Middle marker for 95th percentile
    
    def get_stats(self) -> Dict[str, Any]:
        """Get P-Square statistics"""
        return {
            "percentile": self.percentile,
            "current_estimate": self.get_percentile(),
            "sample_count": self.sample_count,
            "markers": self.markers.copy(),
            "marker_positions": self.marker_positions.copy(),
            "marker_desired_positions": self.marker_desired_positions.copy(),
            "initialized": self.initialized,
            "last_update_time": self.last_update_time
        }
    
    def reset(self) -> None:
        """Reset P-Square calculator"""
        self.markers = [0.0] * 5
        self.marker_positions = [0, 1, 2, 3, 4]
        self.marker_desired_positions = [0, 1, 2, 3, 4]
        self.sample_count = 0
        self.last_update_time = None
        self.initialized = False


class MetricsAggregator:
    """
    Aggregator for multiple metrics calculations
    
    Combines EMA and P-Square calculators for comprehensive metrics tracking
    """
    
    def __init__(self, alpha: float = 0.1, percentile: float = 95.0):
        """
        Initialize metrics aggregator
        
        Args:
            alpha: EMA smoothing factor
            percentile: P-Square target percentile
        """
        self.success_rate_ema = EMACalculator(alpha=alpha, initial_value=0.95)
        self.latency_ema = EMACalculator(alpha=alpha)
        self.latency_p95 = PSquareCalculator(percentile=percentile)
        self.cost_ema = EMACalculator(alpha=alpha, initial_value=0.0)
        
        self.sample_count = 0
        self.last_update_time = None
    
    def update(
        self, 
        success: bool, 
        latency_ms: float, 
        cost: float = 0.0,
        timestamp: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update all metrics with new observation
        
        Args:
            success: Whether the operation was successful
            latency_ms: Operation latency in milliseconds
            cost: Operation cost
            timestamp: Timestamp of the observation
            
        Returns:
            Dictionary with updated metrics
        """
        if timestamp is None:
            import time
            timestamp = time.time()
        
        # Update success rate (convert boolean to 0/1)
        success_value = 1.0 if success else 0.0
        success_rate = self.success_rate_ema.update(success_value, timestamp)
        
        # Update latency metrics
        latency_avg = self.latency_ema.update(latency_ms, timestamp)
        latency_p95 = self.latency_p95.update(latency_ms, timestamp)
        
        # Update cost
        cost_avg = self.cost_ema.update(cost, timestamp)
        
        self.sample_count += 1
        self.last_update_time = timestamp
        
        return {
            "success_rate": success_rate,
            "latency_avg_ms": latency_avg,
            "latency_p95_ms": latency_p95,
            "cost_avg": cost_avg,
            "sample_count": self.sample_count
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "success_rate": self.success_rate_ema.get_value(),
            "latency_avg_ms": self.latency_ema.get_value(),
            "latency_p95_ms": self.latency_p95.get_percentile(),
            "cost_avg": self.cost_ema.get_value(),
            "sample_count": self.sample_count,
            "last_update_time": self.last_update_time
        }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for all metrics"""
        return {
            "success_rate": self.success_rate_ema.get_stats(),
            "latency_avg": self.latency_ema.get_stats(),
            "latency_p95": self.latency_p95.get_stats(),
            "cost_avg": self.cost_ema.get_stats(),
            "aggregate": {
                "sample_count": self.sample_count,
                "last_update_time": self.last_update_time
            }
        }
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.success_rate_ema.reset(initial_value=0.95)
        self.latency_ema.reset()
        self.latency_p95.reset()
        self.cost_ema.reset(initial_value=0.0)
        self.sample_count = 0
        self.last_update_time = None

