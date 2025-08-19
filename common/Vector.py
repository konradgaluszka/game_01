"""
2D Vector Mathematics Utility

This module provides a simple 2D vector class for handling positions, velocities,
and directions in the soccer game. It implements common vector operations needed
for physics calculations, collision detection, and game logic.

**Key Operations**:
- Vector arithmetic (addition, subtraction, scalar multiplication)
- Magnitude calculation (length, length_squared for performance)
- Normalization for unit vectors and direction calculations
- Dot product for angle and projection calculations

**Usage**: Used throughout the codebase for player positions, ball movement,
goal calculations, and physics vector operations.
"""

import math
from dataclasses import dataclass


@dataclass
class Vector:
    """
    2D vector class for position, velocity, and direction calculations.
    
    **Purpose**: Provide vector mathematics for 2D soccer game physics
    
    **Key Features**:
    1. **Arithmetic Operations**: Add, subtract, multiply vectors and scalars
    2. **Magnitude Calculations**: Length and squared length for distance
    3. **Normalization**: Convert to unit vector for direction calculations  
    4. **Dot Product**: For angle calculations and vector projections
    5. **Clean Interface**: Operator overloading for intuitive vector math
    
    **Common Usage Patterns**:
    - Positions: Vector(player.x, player.y)
    - Velocities: Vector(vel_x, vel_y) 
    - Directions: (target - position).normalize()
    - Distances: (pos1 - pos2).length()
    """
    x: float
    y: float

    def __add__(self, other: "Vector") -> "Vector":
        """Add two vectors component-wise. Used for position + displacement."""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector") -> "Vector":
        """Subtract vectors to get displacement from other to self."""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector":
        """Scale vector by multiplying each component by scalar."""
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector":
        """Allow scalar * vector syntax (e.g., 2 * Vector(1,1))."""
        return self.__mul__(scalar)

    def length(self) -> float:
        """Calculate Euclidean distance/magnitude of vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def length_squared(self) -> float:
        """
        Calculate squared magnitude without sqrt for performance.
        Useful for distance comparisons where exact value isn't needed.
        """
        return self.x ** 2 + self.y ** 2

    def normalize(self) -> "Vector":
        """
        Return unit vector (length 1) in same direction.
        Returns zero vector if original length is zero.
        Used for direction calculations and movement vectors.
        """
        l = self.length()
        if l == 0:
            return Vector(0, 0)
        return Vector(self.x / l, self.y / l)

    def dot(self, other: "Vector") -> float:
        """
        Calculate dot product with another vector.
        Used for angle calculations and vector projections.
        Returns: |a| * |b| * cos(angle_between_vectors)
        """
        return self.x * other.x + self.y * other.y

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"