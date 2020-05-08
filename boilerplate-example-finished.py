#!/usr/bin/env python


class MyClass():
    def __init__(self, a, b):
        self._a = a
        self._b = b

from dataclasses import dataclass

@dataclass
class MyDataClass():
    a: int
    b: float
