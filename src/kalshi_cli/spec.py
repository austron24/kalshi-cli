"""OpenAPI specification utilities."""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import yaml


SPEC_PATH = Path(__file__).parent / "openapi.yaml"


@dataclass
class EndpointInfo:
    """Information about an API endpoint."""

    path: str
    method: str
    operation_id: str
    summary: str
    description: str
    tags: list[str] = field(default_factory=list)
    parameters: list[dict] = field(default_factory=list)
    request_body: Optional[dict] = None
    responses: dict = field(default_factory=dict)
    requires_auth: bool = False


def load_spec() -> dict:
    """Load the OpenAPI specification.

    Returns:
        Parsed OpenAPI spec dictionary

    Raises:
        FileNotFoundError: If spec file doesn't exist
    """
    if not SPEC_PATH.exists():
        raise FileNotFoundError(f"OpenAPI spec not found at {SPEC_PATH}")
    with open(SPEC_PATH) as f:
        return yaml.safe_load(f)


def get_endpoints(spec: Optional[dict] = None) -> list[EndpointInfo]:
    """Extract all endpoints from the spec.

    Args:
        spec: Pre-loaded spec, or None to load from file

    Returns:
        List of EndpointInfo objects
    """
    if spec is None:
        spec = load_spec()

    endpoints = []
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if method not in ["get", "post", "put", "delete", "patch"]:
                continue

            endpoints.append(
                EndpointInfo(
                    path=path,
                    method=method.upper(),
                    operation_id=details.get("operationId", ""),
                    summary=details.get("summary", ""),
                    description=details.get("description", ""),
                    tags=details.get("tags", []),
                    parameters=details.get("parameters", []),
                    request_body=details.get("requestBody"),
                    responses=details.get("responses", {}),
                    requires_auth=bool(details.get("security", [])),
                )
            )

    return endpoints


def get_endpoint(
    operation_id: str, spec: Optional[dict] = None
) -> Optional[EndpointInfo]:
    """Get a specific endpoint by operation ID.

    Args:
        operation_id: The operationId to search for (case-insensitive)
        spec: Pre-loaded spec, or None to load from file

    Returns:
        EndpointInfo if found, None otherwise
    """
    endpoints = get_endpoints(spec)
    for ep in endpoints:
        if ep.operation_id.lower() == operation_id.lower():
            return ep
    return None


def get_schemas(spec: Optional[dict] = None) -> dict[str, dict]:
    """Get all schema definitions.

    Args:
        spec: Pre-loaded spec, or None to load from file

    Returns:
        Dictionary of schema name to schema definition
    """
    if spec is None:
        spec = load_spec()
    return spec.get("components", {}).get("schemas", {})


def get_schema(name: str, spec: Optional[dict] = None) -> Optional[dict]:
    """Get a specific schema by name (case-insensitive).

    Args:
        name: Schema name to search for
        spec: Pre-loaded spec, or None to load from file

    Returns:
        Schema definition if found, None otherwise
    """
    schemas = get_schemas(spec)
    for schema_name, schema_def in schemas.items():
        if schema_name.lower() == name.lower():
            return schema_def
    return None


def search_spec(query: str, spec: Optional[dict] = None) -> dict:
    """Search endpoints and schemas by query string.

    Args:
        query: Search query (case-insensitive)
        spec: Pre-loaded spec, or None to load from file

    Returns:
        {"endpoints": [...], "schemas": [...]}
    """
    if spec is None:
        spec = load_spec()

    query_lower = query.lower()

    matching_endpoints = []
    for ep in get_endpoints(spec):
        if (
            query_lower in ep.operation_id.lower()
            or query_lower in ep.path.lower()
            or query_lower in ep.description.lower()
            or query_lower in ep.summary.lower()
        ):
            matching_endpoints.append(ep)

    matching_schemas = [
        name for name in get_schemas(spec) if query_lower in name.lower()
    ]

    return {
        "endpoints": matching_endpoints,
        "schemas": matching_schemas,
    }


def get_tags(spec: Optional[dict] = None) -> dict[str, int]:
    """Get all tags with their endpoint counts.

    Args:
        spec: Pre-loaded spec, or None to load from file

    Returns:
        Dictionary of tag name to endpoint count
    """
    endpoints = get_endpoints(spec)
    tag_counts: dict[str, int] = {}

    for ep in endpoints:
        for tag in ep.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return dict(sorted(tag_counts.items()))


def get_endpoints_by_tag(tag: str, spec: Optional[dict] = None) -> list[EndpointInfo]:
    """Get all endpoints with a specific tag.

    Args:
        tag: Tag to filter by (case-insensitive)
        spec: Pre-loaded spec, or None to load from file

    Returns:
        List of matching endpoints
    """
    endpoints = get_endpoints(spec)
    tag_lower = tag.lower()
    return [ep for ep in endpoints if any(t.lower() == tag_lower for t in ep.tags)]
