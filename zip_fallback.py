import os
import shutil

print("Creating strict structure for fallback zip...")
os.makedirs('temp_zip/output', exist_ok=True)

print("Copying fallback files...")
shutil.copy('fallback_output/matching_results.tsv', 'temp_zip/output/matching_results.tsv')
shutil.copy('fallback_output/candidate_pairs.tsv', 'temp_zip/output/candidate_pairs.tsv')

print("Copying code...")
if os.path.exists('temp_zip/code'):
    shutil.rmtree('temp_zip/code')
shutil.copytree('code', 'temp_zip/code')

print("Copying doc...")
shutil.copy('Documentation_template.md', 'temp_zip/Documentation_template.md')

print("Zipping...")
shutil.make_archive('Fallback_KingPing_submission', 'zip', 'temp_zip')

print("Cleaning up...")
shutil.rmtree('temp_zip')
print("Done. Fallback_KingPing_submission.zip created.")
