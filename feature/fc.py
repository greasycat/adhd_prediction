from utils.subject import SubjectDataset

class FC:
    def __init__(self, subject_dataset: SubjectDataset):
        self.subject_dataset = subject_dataset

    def get_fc(self, id):
        return self.subject_dataset.get_fc(id)

    def get_all_fc(self):
        return [self.get_fc(id) for id in self.subject_dataset.get_all_ids()]

    # def get_all_fc_paths(self):
    #     return [self.subject_dataset.get_fc_path(id) for id in self.subject_dataset.get_all_ids()]