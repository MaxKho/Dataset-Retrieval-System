import subprocess
import sys
import os

def run_script(script_path, success_msg):
    print(f"Running: {script_path}")
    result = subprocess.run(["python", script_path])
    if result.returncode == 0:
        print(success_msg)
    else:
        print(f"Error running {script_path}. Exiting.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Starting the pipeline with working directory:", os.getcwd())

    # 1. Document creation
    run_script("document_creation_module/document_creation.py", "1) Document creation successful.")

    # 2. Query generation (queries & feature summaries)
    run_script("query_generation_module/generate_queries_feature_summaries_and_query_document_pairs.py", "2) Query generation successful.")

    # 3. Generate hard negatives
    run_script("training_set_generation_module/generate_hard_negatives.py", "3) Hard negatives generated successfully.")

    # 4. Combine data sources
    run_script("training_set_generation_module/combine_data_sources.py", "4) Data sources combined successfully.")

    # 5. Split data
    run_script("training_set_generation_module/split_data.py", "5) Data split successful.")

    # 6. Fine-tuning
    run_script("fine_tuning_module/fine_tune_spectar2_partial.py", "6) Fine-tuning successful.")

    # 7. Scoring
    run_script("scoring_function_module/scoring_function_partial.py", "7) Scoring complete.")

    # 8. Evaluation
    run_script("evaluation_module/evaluation.py", "8) Evaluation successful.")

    print("Pipeline finished successfully.")