from __future__ import annotations

class ShapeNode:
    id: int  # ID único del nodo (usado como clave primaria en la DB)
    next_id: int  # ID del siguiente nodo (clave foránea a otro ColorNode)
    prev_id: int  # ID del nodo anterior (clave foránea a otro ColorNode)
    next_i: int
    next_j: int
    prev_i: int
    prev_j: int