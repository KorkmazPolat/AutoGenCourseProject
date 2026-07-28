from __future__ import annotations

from typing import Any

from fastapi.responses import HTMLResponse


def enable_legacy_template_response(templates: Any) -> Any:
    """Allow TemplateResponse(name, context) with newer Starlette versions."""
    starlette_template_response = templates.TemplateResponse

    def template_response_compat(*args: Any, **kwargs: Any) -> HTMLResponse:
        if args and isinstance(args[0], str):
            if len(args) < 2 or not isinstance(args[1], dict):
                return starlette_template_response(*args, **kwargs)
            context = args[1]
            request = context.get("request")
            if request is None:
                raise ValueError("Template context must include a request object")
            return starlette_template_response(request, args[0], context, *args[2:], **kwargs)
        return starlette_template_response(*args, **kwargs)

    templates.TemplateResponse = template_response_compat
    return templates
