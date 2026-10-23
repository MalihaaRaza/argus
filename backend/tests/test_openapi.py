"""Error responses are documented in the OpenAPI schema (no database needed)."""

from typing import Any

from argus.api import app


def response_schema(path: str, code: str) -> dict[str, Any]:
    response = app.openapi()["paths"][path]["get"]["responses"][code]
    return response["content"]["application/json"]["schema"]


def test_get_run_documents_404() -> None:
    schema = response_schema("/runs/{run_id}", "404")
    assert schema == {"$ref": "#/components/schemas/ErrorOut"}


def test_health_documents_200_and_503() -> None:
    for code in ("200", "503"):
        schema = response_schema("/health", code)
        assert schema == {"$ref": "#/components/schemas/HealthOut"}


def test_error_schemas_describe_the_bodies() -> None:
    schemas = app.openapi()["components"]["schemas"]
    assert schemas["ErrorOut"]["required"] == ["detail"]
    assert schemas["HealthOut"]["properties"]["database"]["enum"] == [
        "ok",
        "unreachable",
    ]
