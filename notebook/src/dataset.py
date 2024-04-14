import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import random
import torch

DiagnosisStr_to_Int_Mapping={
    'no_AS':0,
    'mild_AS':1,
    'mildtomod_AS':1,
    'moderate_AS':2,
    'severe_AS':2
}
     
class EchoDataset(Dataset):
    def __init__(self, PatientStudy_list, TMED2SummaryTable, ML_DATA_dir, sampling_strategy='first_frame', training_seed=0, transform_fn=None):   
        self.PatientStudy_list = PatientStudy_list 
        self.TMED2SummaryTable = TMED2SummaryTable 
        self.ML_DATA_dir = ML_DATA_dir #'Echo_MIL/AS_Diagnosis/ML_DATA/TMED2Release'
        self.data_root_dir = os.path.join(self.ML_DATA_dir)    
        self.sampling_strategy = sampling_strategy
        self.training_seed=training_seed     
        self.transform_fn = transform_fn  
        self.bag_of_PiatentStudy_images, self.bag_of_PatientStudy_DiagnosisLabels = self._create_bags()

    def _create_bags(self):
        bag_of_PatientStudy_images = []
        bag_of_PatientStudy_DiagnosisLabels = []
        num_studies = len(self.PatientStudy_list)
        print("Processing [{}] studies".format(num_studies))
        #interval = 1
        for study_idx, PatientStudy in enumerate(self.PatientStudy_list):
            #if study_idx % interval == 0:
                #print("Processed [{}%] of the studies".format( (study_idx / num_studies) * 100))
            this_PatientStudy_dir = os.path.join(self.data_root_dir, PatientStudy)       
            # get diagnosis label for this PatientStudy
            this_PatientStudyRecords_from_TMED2SummaryTable = self.TMED2SummaryTable[self.TMED2SummaryTable['patient_study']==PatientStudy]
            assert this_PatientStudyRecords_from_TMED2SummaryTable.shape[0]!=0, 'every PatientStudy from the studylist should be found in TMED2SummaryTable'
            
            this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel = list(set(this_PatientStudyRecords_from_TMED2SummaryTable.diagnosis_label.values)) 
            assert len(this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel)==1, 'every PatientStudy should only have one diagnosis label'
            
            this_PatientStudy_DiagnosisLabel = this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel[0]
            this_PatientStudy_DiagnosisLabel = DiagnosisStr_to_Int_Mapping[this_PatientStudy_DiagnosisLabel]
            
            assert os.path.isfile(this_PatientStudy_dir+"_0.png"), 'ERROR: every PatientStudy from the studylist should be found {}'.format(this_PatientStudy_dir+"_0.png")
            all_TiffFilename_this_PatientStudy = [fname for fname in os.listdir(self.data_root_dir) if PatientStudy in fname and "png" in fname]
            all_TiffFilename_this_PatientStudy.sort()

            # different sampling strategy
            if self.sampling_strategy == 'first_frame':
                bag_of_PatientStudy_DiagnosisLabels.append(this_PatientStudy_DiagnosisLabel)       
                this_PatientStudy_images = []     
                for TiffFilename in all_TiffFilename_this_PatientStudy:
                    img_path = os.path.join(self.data_root_dir, TiffFilename)
                    img = np.array(Image.open(img_path))
                    img = img[:, :, None]
                    img = img[:, :, (0, 0, 0)]
                    assert img.shape == (112, 112, 3), "Image [{}]'s size [{}] != [(112, 112, 3)]".format(TiffFilename, img.shape)
                    this_PatientStudy_images.append(img)        
                # shape: (number of images in study, 112, 112, 3)
                this_PatientStudy_images = np.array(this_PatientStudy_images)     
                bag_of_PatientStudy_images.append(this_PatientStudy_images)   
            else:
                assert False
        return bag_of_PatientStudy_images, bag_of_PatientStudy_DiagnosisLabels
    
    def __len__(self):
        return len(self.bag_of_PiatentStudy_images)
    
    def __getitem__(self, index):   
        bag_image = self.bag_of_PiatentStudy_images[index]
        if self.transform_fn is not None:           
            bag_image = torch.stack([self.transform_fn(Image.fromarray(image)) for image in bag_image])
        DiagnosisLabel = self.bag_of_PatientStudy_DiagnosisLabels[index]
        return bag_image, DiagnosisLabel

