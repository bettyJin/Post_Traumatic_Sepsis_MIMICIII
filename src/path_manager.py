import os

class ProjectPaths:
    def __init__(self, base_path):
        self.base_path = base_path
        # data folder: raw data, preprocess data, final dataset
        self.data_path = os.path.join(self.base_path, 'data')
        self.raw_data_path = os.path.join(self.data_path, 'raw')
        self.processed_data_path = os.path.join(self.data_path, 'processed')
        self.final_data_path = os.path.join(self.data_path, 'final')
        # 
        self.scripts_path = os.path.join(self.base_path, 'scripts')
        # self.notebooks_path = os.path.join(self.base_path, 'notebooks')

        # Source code for use in this project.
        self.src_path = os.path.join(self.base_path, 'src')
        # self.docs_path = os.path.join(self.base_path, 'docs')
        # self.tests_path = os.path.join(self.base_path, 'tests')
        self.supplementary_path = os.path.join(self.base_path, 'supplementary')

    def get_file_path(self, *path_segments):
        return os.path.join(self.base_path, *path_segments)

    def get_raw_data_file(self, filename):
        return os.path.join(self.raw_data_path, filename)

    def get_processed_data_file(self, filename):
        return os.path.join(self.processed_data_path, filename)

    def get_final_data_file(self, filename):
        return os.path.join(self.final_data_path, filename)

    def get_script_file(self, filename):
        return os.path.join(self.scripts_path, filename)

    # def get_notebook_file(self, filename):
    #     return os.path.join(self.notebooks_path, filename)

    def get_src_file(self, filename):
        return os.path.join(self.src_path, filename)

    # def get_doc_file(self, filename):
    #     return os.path.join(self.docs_path, filename)

    # def get_test_file(self, filename):
    #     return os.path.join(self.tests_path, filename)

    def get_supplementary_file(self, filename):
        return os.path.join(self.supplementary_path, filename)

# # Example usage
# base_path = '/your/project/base/path'
# project_paths = ProjectPaths(base_path)

# # Access a specific file
# icd_file_path = project_paths.get_supplementary_file('ICD_Nonpoisoning_Cause_Matrix.xlsx')
# print(icd_file_path)
