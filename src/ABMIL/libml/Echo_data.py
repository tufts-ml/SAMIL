#Dataset version1: double check what is dataset version1


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




#####################DS1      
class EchoDataset(Dataset):
    '''
    Echo_MIL raw data is structured exactly the same as TMED2Release, and has exactly the same tiff records, except including an additional part: diagnosis_labeled/
    
    --TMED2Release
        --diagnosis_labeled
            --19 studies folder named as PatientIDnocsv/
                                        --all tiff_filename/ for this study
                                            --video_images.npy
            
        --unlabeled
            --5486 studies folder named as PatientIDStudyID/
                                        --all tiff_filename/ for this study
                                            --video_images.npy
            
        --view_and_diagnosis_labeled
            --labeled
                --598 studies folder named as PatientIDStudyID/
                                        --all tiff_filename/ for this study
                                            --video_images.npy
            --unlabeled
                --596 studies folder named as PatientIDStudyID/
                                        --all tiff_filename/ for this study
                                            --video_images.npy
        --view_labeled
            --labeled
                --705 studies folder named as PatientIDStudyID/
                                        --all tiff_filename/ folder this study
                                            --video_images.npy
                                            
            --unlabeled
                --640 studies folder named as PatientIDStudyID/
                                        --all tiff_filename/ folder this study
                                            --video_images.npy
        
    
    '''
    
    def __init__(self, PatientStudy_list, TMED2SummaryTable, ML_DATA_dir, sampling_strategy='first_frame', training_seed=0, transform_fn=None):
        
        self.PatientStudy_list = PatientStudy_list
        self.TMED2SummaryTable = TMED2SummaryTable #note: using the patient_id column in TMED2SummaryTable can uniquely identify a patient_study (there is NO same patient_study belong to different parts: diagnosis_labeled/, unlabeled/, view_and_diagnosis_labeled_set/, view_labeled AT THE SAME TIME)
        
        self.ML_DATA_dir = ML_DATA_dir #'Echo_MIL/AS_Diagnosis/ML_DATA/TMED2Release'
        self.data_root_dir = os.path.join(self.ML_DATA_dir) 
    
        self.sampling_strategy = sampling_strategy

        self.training_seed=training_seed
        
        self.transform_fn = transform_fn
        
        self.bag_of_PiatentStudy_images, self.bag_of_PatientStudy_DiagnosisLabels = self._create_bags()
        
        
    
    def _create_bags(self):
        
        bag_of_PatientStudy_images = []
        bag_of_PatientStudy_DiagnosisLabels = []
#         num_cineloop_with_only_1_frame = 0
        
        for PatientStudy in self.PatientStudy_list:
            this_PatientStudy_dir = os.path.join(self.data_root_dir, PatientStudy)
            
            #get diagnosis label for this PatientStudy
            this_PatientStudyRecords_from_TMED2SummaryTable = self.TMED2SummaryTable[self.TMED2SummaryTable['patient_study']==PatientStudy]
            assert this_PatientStudyRecords_from_TMED2SummaryTable.shape[0]!=0, 'every PatientStudy from the studylist should be found in TMED2SummaryTable'
            
            this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel = list(set(this_PatientStudyRecords_from_TMED2SummaryTable.diagnosis_label.values)) 
            assert len(this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel)==1, 'every PatientStudy should only have one diagnosis label'
            
            this_PatientStudy_DiagnosisLabel = this_PatientStudyRecords_from_TMED2SummaryTable_DiagnosisLabel[0]
            this_PatientStudy_DiagnosisLabel = DiagnosisStr_to_Int_Mapping[this_PatientStudy_DiagnosisLabel]
            
            
            assert os.path.exists(this_PatientStudy_dir), 'every PatientStudy from the studylist should be found {}'.format(this_PatientStudy_dir)
            
            #get all tiff files for thie PatientStudy
            all_TiffFilename_this_PatientStudy = [i for i in os.listdir(this_PatientStudy_dir) if '.ipynb_checkpoints' not in i]
            all_TiffFilename_this_PatientStudy.sort() #just to ensure order of the tiff files are consistent each time
            
            
            #different sampling strategy
            if self.sampling_strategy == 'first_frame':
                bag_of_PatientStudy_DiagnosisLabels.append(this_PatientStudy_DiagnosisLabel)
            
                this_PatientStudy_images = []
                
                for TiffFilename in all_TiffFilename_this_PatientStudy:
                    this_cineloop_frames_path = os.path.join(this_PatientStudy_dir, TiffFilename, 'video_images.npy')
                    this_cineloop_data = np.load(this_cineloop_frames_path, allow_pickle=True).item()
                    this_cineloop_frames = this_cineloop_data['images']
                    assert this_cineloop_frames.shape[0]>=1, 'By construction, ensured each extracted tiff file has more than 1 valid frames, {}, {}'.format(this_cineloop_frames.shape, this_cineloop_frames_path)
                    
#                     if this_cineloop_frames.shape[0]==1:
#                         num_cineloop_with_only_1_frame += 1

                    this_PatientStudy_images.append(this_cineloop_frames[0])
                    
                this_PatientStudy_images = np.array(this_PatientStudy_images)
                
                bag_of_PatientStudy_images.append(this_PatientStudy_images)
                
                
            else: 
                #set random seed is already done in main.py
                
                #sample x times for this PatientStudy, resulting in x bag of correlated but not exactly the same bag of this PatientStudy
                for i in range(int(self.sampling_strategy)):
#                     print('!!!!!!!!!!!!!{} copy{}!!!!!!!!!!!'.format(PatientStudy, i))

                    bag_of_PatientStudy_DiagnosisLabels.append(this_PatientStudy_DiagnosisLabel)
                    
                    this_PatientStudy_images = []

                    for TiffFilename in all_TiffFilename_this_PatientStudy:
                        this_cineloop_frames_path = os.path.join(this_PatientStudy_dir, TiffFilename, 'video_images.npy')
                        this_cineloop_data = np.load(this_cineloop_frames_path, allow_pickle=True).item()
                        this_cineloop_frames = this_cineloop_data['images']
                        assert this_cineloop_frames.shape[0]>=1, 'By construction, ensured each extracted tiff file has more than 1 valid frames'
                        selected_frame_index = random.randint(0, this_cineloop_frames.shape[0]-1) #randint is left and right included
#                         print('{} total frames {}'.format(this_cineloop_frames_path, this_cineloop_frames.shape[0]))
#                         print('selected_frame_index: {}'.format(selected_frame_index))
                        
                        this_PatientStudy_images.append(this_cineloop_frames[selected_frame_index])
                        
                    this_PatientStudy_images = np.array(this_PatientStudy_images)
                    
                    bag_of_PatientStudy_images.append(this_PatientStudy_images)
                
#                 raise NameError('To break out the program')


          

        
#         bag_of_PatientStudy_images = np.array(bag_of_PatientStudy_images)
#         bag_of_PatientStudy_DiagnosisLabels = np.array(bag_of_PatientStudy_DiagnosisLabels)
        
#         print('num_cineloop_with_only_1_frame: {}'.format(num_cineloop_with_only_1_frame))
        
        return bag_of_PatientStudy_images, bag_of_PatientStudy_DiagnosisLabels
    
    
    def __len__(self):
        return len(self.bag_of_PiatentStudy_images)
    
    
    def __getitem__(self, index):
        
        bag_image = self.bag_of_PiatentStudy_images[index]
        
#         print('self.transform_fn(Image.fromarray(image)) shape: {}'.format(self.transform_fn(Image.fromarray(bag_image[0]))))

        if self.transform_fn is not None:           
            bag_image = torch.stack([self.transform_fn(Image.fromarray(image)) for image in bag_image])
        
#         print('Inside Echo_data_DS1Like, bag_image shape: {}'.format(bag_image.shape))
        
        DiagnosisLabel = self.bag_of_PatientStudy_DiagnosisLabels[index]
#         print('DiagnosisLabel: {}'.format(DiagnosisLabel))

#         print('index: {} bag label: {}'.format(index, DiagnosisLabel))

        
        return bag_image, DiagnosisLabel

