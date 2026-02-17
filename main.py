import argparse
import os
import runpy
import sys
from pathlib import Path

from data_curation.prepare_data import download_and_extract_files, merge_jsons
from data_curation.export import cvm2csv
from preprocessing.fda_pipeline import run_fda_cleaning
from preprocessing.pubchem_fda_pipeline import run_pipeline
from predictive_modeling.ml_pipeline import ml_pipeline


def _ensure_project_root() -> Path:
	base_dir = Path(__file__).resolve().parent
	os.chdir(base_dir)
	if str(base_dir) not in sys.path:
		sys.path.insert(0, str(base_dir))
	return base_dir


def _has_any_subdir(path: Path) -> bool:
	if not path.exists():
		return False
	return any(p.is_dir() for p in path.iterdir())


def _should_run(outputs: list[Path], force: bool) -> bool:
	if force:
		return True
	return any(not p.exists() for p in outputs)


def _run_llm_pipeline(base_dir: Path) -> None:
	llm_script = base_dir / "predictive_modeling" / "LLM" / "few_shot.py"
	if not llm_script.exists():
		raise FileNotFoundError(f"LLM script not found: {llm_script}")
	print(f"[run] Running LLM script: {llm_script}")
	runpy.run_path(str(llm_script), run_name="__main__")


def run_all(
	force: bool,
	skip_download: bool,
	skip_pubchem: bool,
	skip_ml: bool,
	mode: str,
) -> None:
	base_dir = _ensure_project_root()
	data_dir = base_dir / "data"
	json_dir = data_dir / "json_files"

	print("=== CVM_FDA pipeline start ===")

	# Step 1: Download and extract FDA CVM JSONs
	if skip_download:
		print("[skip] Download step was skipped.")
	else:
		if _has_any_subdir(json_dir) and not force:
			print("[skip] JSON files already exist; use --force to re-download.")
		else:
			json_dir.mkdir(parents=True, exist_ok=True)
			print("[run] Downloading FDA CVM JSON files...")
			download_and_extract_files(save_path=str(json_dir) + "/")

	# Step 2: Merge JSON files into a single file
	fda_all_data_json = data_dir / "all_data.json"
	if _has_any_subdir(json_dir):
		if _should_run([fda_all_data_json], force=force):
			print("[run] Merging JSON files...")
			merge_jsons(json_dir=str(json_dir) + "/", save_path=str(fda_all_data_json))
		else:
			print("[skip] all_data.json already exists; use --force to regenerate.")
	else:
		print("[skip] No JSON files found; merge_jsons skipped.")

	# Step 3: Export merged CSVs
	csv_outputs = [
		data_dir / "all.csv",
		data_dir / "drugs.csv",
		data_dir / "reactions.csv",
		data_dir / "outcomes.csv",
	]
	if _should_run(csv_outputs, force=force):
		print("[run] Exporting CSV files from JSON...")
		cvm2csv(json_dir=str(json_dir), save_dir=str(data_dir))
	else:
		print("[skip] CSV outputs already exist; use --force to regenerate.")

	# Step 4: FDA preprocessing + cleaning
	npz_output = data_dir / "np_analysis_cleaned.npz"
	if _should_run([npz_output], force=force):
		print("[run] Running FDA preprocessing pipeline...")
		run_fda_cleaning(raw_dir=str(data_dir), out_npz=str(npz_output))
	else:
		print("[skip] Cleaned NPZ already exists; use --force to regenerate.")

	# Step 5: PubChem enrichment
	if skip_pubchem:
		print("[skip] PubChem enrichment step was skipped.")
	else:
		pubchem_outputs = [
			data_dir / "expanded_cleaned_data.csv",
			data_dir / "pubchem_compounds.csv",
		]
		if _should_run(pubchem_outputs, force=force):
			print("[run] Running PubChem enrichment pipeline...")
			run_pipeline(
				npz_path=str(npz_output),
				conversion_json=str(data_dir / "drugs_conversion_backup.json"),
				pugview_dir=str(data_dir / "pugviews_json"),
				out_csv=str(data_dir / "expanded_cleaned_data.csv"),
				pubchem_csv=str(data_dir / "pubchem_compounds.csv"),
			)
		else:
			print("[skip] PubChem outputs already exist; use --force to regenerate.")

	# Step 6: Modeling pipeline (ML or LLM)
	if skip_ml:
		print("[skip] Modeling step was skipped.")
	else:
		if mode == "ml":
			print("[run] Running ML pipeline...")
			ml_pipeline()
		elif mode == "llm":
			_run_llm_pipeline(base_dir)
		else:
			raise ValueError(f"Unsupported mode: {mode}")

	print("=== CVM_FDA pipeline complete ===")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run the CVM_FDA project pipeline in order."
	)
	parser.add_argument(
		"--mode",
		choices=["ml", "llm"],
		default="ml",
		help="Choose modeling pipeline: ml or llm.",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Re-run steps even if outputs already exist.",
	)
	parser.add_argument(
		"--skip-download",
		action="store_true",
		help="Skip downloading FDA CVM JSON files.",
	)
	parser.add_argument(
		"--skip-pubchem",
		action="store_true",
		help="Skip PubChem enrichment step.",
	)
	parser.add_argument(
		"--skip-ml",
		action="store_true",
		help="Skip the modeling step (ML or LLM).",
	)
	return parser


if __name__ == "__main__":
	args = build_arg_parser().parse_args()
	run_all(
		force=args.force,
		skip_download=args.skip_download,
		skip_pubchem=args.skip_pubchem,
		skip_ml=args.skip_ml,
		mode=args.mode,
	)
