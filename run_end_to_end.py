import os
import sys

# Ensure the root of the project is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.business_entity_resolution.pipeline import InferencePipeline

def main():
    print("Starting End-to-End Pipeline...")
    pipeline = InferencePipeline(
        test_dir='dataset/test',
        output_dir='output',
        models_dir='models',
        exp_dir='experiments'
    )
    
    summary = pipeline.run_full_pipeline()
    print("Pipeline finished successfully!")
    print("Summary:", summary)

if __name__ == "__main__":
    main()
