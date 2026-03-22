"""
local_zep ontology stubs
Compatible with the zep_cloud.external_clients.ontology interface.

graph_builder.py uses type(name, (EntityModel,), attrs) to dynamically create subclasses;
these classes only need to exist — no business logic is required.
"""


class EntityModel:
    """Entity base class (stub)."""
    pass


class EntityText:
    """Entity text field type (stub)."""
    pass


class EdgeModel:
    """Edge base class (stub)."""
    pass
