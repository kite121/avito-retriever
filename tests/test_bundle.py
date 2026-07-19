import json
import zipfile

from avito_retriever.experiments.bundle import build_analysis_bundle


def test_analysis_bundle_includes_results_but_not_raw_data(tmp_path) -> None:
    (tmp_path / "artifacts/notebook_results/05_final").mkdir(parents=True)
    (tmp_path / "artifacts/notebook_results/05_final/results.csv").write_text("x\n1\n")
    (tmp_path / "data/candidate_data").mkdir(parents=True)
    (tmp_path / "data/candidate_data/articles.f").write_bytes(b"private bundle payload")

    bundle, _ = build_analysis_bundle(tmp_path)
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert "artifacts/notebook_results/05_final/results.csv" in names
        assert "data/candidate_data/articles.f" not in names
        manifest = json.loads(archive.read("analysis_bundle_manifest.json"))
        assert manifest["file_count"] == 1
