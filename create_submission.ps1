$ErrorActionPreference = "Stop"

Write-Host "Validating submission outputs..."
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Validation failed! You can still submit, but you may want to review the errors above."
} else {
    Write-Host "Validation passed successfully."
}

$zipName = "KingPing_submission.zip"
Write-Host "Creating submission zip: $zipName..."
if (Test-Path $zipName) {
    Remove-Item $zipName
}

# The required structure is:
# <team_name>_submission.zip
# ├── output/
# │   ├── matching_results.tsv
# │   └── candidate_pairs.tsv
# ├── code/
# │   └── business_entity_resolution/
# │       ├── src/
# │       ├── README.md
# │       └── requirements.txt
# └── Documentation_template.md

Compress-Archive -Path output, code, Documentation_template.md -DestinationPath $zipName -Force

Write-Host "Submission package $zipName created successfully!"
