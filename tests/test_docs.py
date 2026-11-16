"""The docs are part of the product: generated pages must match the code and links must resolve."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="docs/ is not part of the installed package")


def _generator():
    spec = importlib.util.spec_from_file_location("build_api_docs", ROOT / "tools" / "build_api_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_pages_are_up_to_date():
    gen = _generator()
    assert (DOCS / "models.md").read_text(encoding="utf-8") == gen.models_page(), \
        "docs/models.md is stale: run  python tools/build_api_docs.py"
    assert (DOCS / "api.md").read_text(encoding="utf-8") == gen.api_page(), \
        "docs/api.md is stale: run  python tools/build_api_docs.py"


def test_every_model_key_is_documented():
    import multimodel_analysis as mm
    text = (DOCS / "models.md").read_text(encoding="utf-8")
    for cls in (mm.MultiModelClassifier, mm.MultiModelRegressor, mm.MultiModelClusterer,
                mm.MultiModelAnomalyDetector, mm.MultiModelReducer):
        for key in cls.available_models():
            assert f"`{key}`" in text, f"{cls.__name__}: model '{key}' missing from docs/models.md"


_LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize("page", sorted([ROOT / "readme.md", *DOCS.glob("*.md")]), ids=lambda p: p.name)
def test_relative_links_exist(page):
    for target in _LINK.findall(page.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (page.parent / target).exists(), f"{page.name}: broken link -> {target}"
