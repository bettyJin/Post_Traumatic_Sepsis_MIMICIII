import os
from src.data import data_utils

def test_data_access(project_id):
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    query = """
            -- total number of hospital admission IDs in MIMIC III
            SELECT COUNT(DISTINCT HADM_ID)
            FROM `physionet-data.mimiciii_clinical.admissions`
            """
    df = data_utils.run_query(query, project_id)

    if df.values[0][0] == 58976:
        print("Successfully accessed the MIMIC III via BigQuery")
    else:
        print("Failed to access the MIMIC III via BigQuery")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test MIMIC III data access.')
    parser.add_argument('project_id', type=str, help='Google Cloud project ID')

    args = parser.parse_args()
    test_data_access(args.project_id)
