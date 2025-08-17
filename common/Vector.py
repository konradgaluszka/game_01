import math
from dataclasses import dataclass


@dataclass
class Vector:
    x: float
    y: float

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector":
        """Multiply by a scalar."""
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector":
        """Allow scalar * Vector as well."""
        return self.__mul__(scalar)

    def length(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)


    def length_squared(self) -> float:
        return self.x ** 2 + self.y ** 2

    def normalize(self) -> "Vector":
        l = self.length()
        if l == 0:
            return Vector(0, 0)
        return Vector(self.x / l, self.y / l)

    def dot(self, other: "Vector") -> float:
        return self.x * other.x + self.y * other.y

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y})"