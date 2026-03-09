import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_dir = '/home/xulu/code/MetaLearningDynamics/example_data/'

class TrialData:
    def __init__(self, BehaviorTrialTable, IncludeLickTime=False, ImportExistingTable=False):
        '''
        TrialData includes:
        1) Behavioral data table (from alison_spyglass and columns customized by XS for the stem-leaf task).\
        2) Neural data table
        '''

        self.BinsPerSec = 30
        self.TrialDataTable = None # this will be updated at the end of init
        self.BehaviorTableByTrial = None
        self.NeuralTableByTrial = None 
        self.TrialsOfInterest = {}

        if ImportExistingTable:
            # Load behav dataframe from a local folder:
            TrialDataTable = pd.read_pickle(data_dir+f'{BehaviorTrialTable.nwb_file_name.unique()[0][:-5]}_behav_dataframe.pkl')

        else:
            # Define TrialDataTable from spyglass tables loaded from kachery cloud.
            # Trial end time: 
            if IncludeLickTime:
                BehaviorTrialTable['EndTime'] =  BehaviorTrialTable['poke_out_ts'] # poke out time on the current trial (run + lick time)
            else:
                BehaviorTrialTable['EndTime'] =  BehaviorTrialTable['poke_in_ts'] # poke in time on the current trial (running time only)
            # Trial start time: poke out time on the last trial
            BehaviorTrialTable['StartTime'] = \
            np.insert(BehaviorTrialTable['poke_out_ts'][:-1].to_numpy(),0,np.nan) 
            #Duration
            BehaviorTrialTable['Duration'] = BehaviorTrialTable['EndTime'] - BehaviorTrialTable['StartTime']
            
            #Start leaf
            BehaviorTrialTable['EndLeaf'] = BehaviorTrialTable['leaf']
            BehaviorTrialTable['StartLeaf'] = \
            np.insert(BehaviorTrialTable['leaf'][:-1].to_numpy(),0,0)
            
            allRoute = 10*BehaviorTrialTable.StartLeaf.to_numpy() + BehaviorTrialTable.EndLeaf.to_numpy()
            stayRoute = np.array([12,21,34,43,56,65])
            IfStay = list(map(lambda x: x in stayRoute, allRoute))
            
            # Route and stay/switch decisions per trial
            BehaviorTrialTable['Route'] = allRoute
            BehaviorTrialTable['IsSwitch'] = [not x for x in IfStay]
            BehaviorTrialTable['IsSwitchPrevious'] = \
            np.insert(BehaviorTrialTable['IsSwitch'][:-1].to_numpy(),0,0)
            BehaviorTrialTable['IsSwitchPrevPrev'] = \
            np.insert(BehaviorTrialTable['IsSwitch'][:-2].to_numpy(),[0,0],[0,0])        
            
            #Start stem
            BehaviorTrialTable['EndStem'] = BehaviorTrialTable['stem']
            BehaviorTrialTable['StartStem'] = \
            np.insert(BehaviorTrialTable['stem'][:-1].to_numpy(),0,0)
            
            #Reward history
            BehaviorTrialTable['Reward'] = BehaviorTrialTable['reward']
            BehaviorTrialTable['RewardPrevious'] = \
            np.insert(BehaviorTrialTable['reward'][:-1].to_numpy(),0,0) # Whether trial n-1 is rewarded
            BehaviorTrialTable['RewardPrevPrev'] = \
            np.insert(BehaviorTrialTable['reward'][:-2].to_numpy(),[0,0],[0,0]) # Whether trial n-2 is rewarded
                    
            # Build the customized TrialData behav table:
            '''
            Choose only relevant attributes and get rid of the first trial of each epoch (because they don't have a start time)
            '''
            trial_mask = (BehaviorTrialTable.epoch.diff()) == 0
            TrialDataTable = BehaviorTrialTable\
            [['StartTime','EndTime','poke_in_ts','poke_out_ts','Duration',
              'StartLeaf','EndLeaf','StartStem','EndStem','BestStem','MediumStem','Route',\
              'IsSwitch','IsSwitchPrevious','IsSwitchPrevPrev',\
              'ExpectedValue','PastValue','AltLeafValue','ValueDifference',\
              'Reward','RewardPrevious','RewardPrevPrev',\
              'pokes_before_switch','trial_id_before_switch',\
              'nwb_file_name','epoch','trial_number_by_epoch']][trial_mask]
        
        self.TrialDataTable = TrialDataTable

        print(" The recording is " + \
              str(round((self.TrialDataTable.EndTime.iloc[-1] - self.TrialDataTable.EndTime.iloc[0])/60))\
             +" min long")

    def PopulateNeuralDataTable(self,SpikeTimeTable,BinsPerSec=None,AlignedInterval=None,num_bins=None,Alignment="TrialStart"):
        '''
        NeuralDataTable: contains the spike times of each neuron per trial
        One trial per row; one neuron per column. 
        Each trial is aligned to the user-defined interval by user-defined bin size.
        '''
        if BinsPerSec is None:
            BinsPerSec = self.BinsPerSec
        TrialDataTable = self.TrialDataTable.copy(deep=True) # We'll use trial start and end times in the behavior table for alignment
        
        for unit in SpikeTimeTable.index:
            array = [] # array that has different values of Behaviors
            for trial in np.array(TrialDataTable.index):

                if AlignedInterval is None:
                    #If not defined by users, we will do time warping of trials in different lengths with fixed numers of bins
                    if Alignment == "PokeStartToEnd": # XS: Not useable yet because each trial would have different lengths in this case.
                        binHere = np.linspace(TrialDataTable.poke_in_ts[trial], TrialDataTable.poke_out_ts[trial], num_bins + 1)
                        duration_per_bin = (TrialDataTable.poke_out_ts[trial] - TrialDataTable.poke_in_ts[trial])/num_bins
                        # As each trial has different durations, need to normalize spike counts by actual duration of each bin.
                        Fr_one_trial, _ = np.histogram(SpikeTimeTable.spike_times[unit], bins=binHere)/duration_per_bin 
                    TrialLength = num_bins                        
                    self.timevec = np.linspace(0, 1, num_bins + 1)[:-1]
                else:
                    # Neural data alignment <= aligned time point - interval[0] to aligned time point + interval[1]
                    if Alignment == "TrialStart":
                        ##### If we want to cut off timepoints beyond the end of a trial:
                        # binHere = np.arange(TrialDataTable.StartTime[trial]+AlignedInterval[0],\
                        #                     np.min([TrialDataTable.EndTime[trial]+1/BinsPerSec,\
                        #                             TrialDataTable.StartTime[trial]+AlignedInterval[1]]),1/BinsPerSec)
                        ##### Otherwise:
                        binHere = np.arange(TrialDataTable.StartTime[trial]+AlignedInterval[0],\
                                            TrialDataTable.StartTime[trial]+AlignedInterval[1],1/BinsPerSec)                        
                    if Alignment == "TrialEnd":
                        # if trial == TrialDataTable.index[-1]:
                        #     binHere = np.arange(TrialDataTable.EndTime[trial]+AlignedInterval[0],\
                        #                         TrialDataTable.EndTime[trial]+AlignedInterval[1],1/BinsPerSec)
                        # else:
                        #     binHere = np.arange(TrialDataTable.EndTime[trial]+AlignedInterval[0],\
                        #                         np.min([TrialDataTable.EndTime[trial]+AlignedInterval[1],\
                        #                                TrialDataTable.poke_in_ts[trial+1]]),1/BinsPerSec)
                        binHere = np.arange(TrialDataTable.EndTime[trial]+AlignedInterval[0],\
                                            TrialDataTable.EndTime[trial]+AlignedInterval[1],1/BinsPerSec)
                    TrialLength = AlignedInterval[1]-AlignedInterval[0]                        
                    Fr_one_trial = np.nan * np.empty((np.arange(0,TrialLength,1/BinsPerSec)[:-1].shape[0],)) # Not all trials are as long as the defined interval
                    binned_spikes = np.histogram(SpikeTimeTable.spike_times[unit],bins= binHere)[0]*BinsPerSec
                    Fr_one_trial[0:binned_spikes.shape[0]] = binned_spikes
                    self.timevec = np.arange(0,TrialLength,1/BinsPerSec)[:-1] 
                    
                array.append(Fr_one_trial)
                
            TrialDataTable['SpikeTrainUnit_'+ str(unit)] = array
        
            NeuralContent = list(filter(lambda x: "Unit" in x,\
                                      TrialDataTable.keys().to_numpy()))
            BehaviorContent = list(filter(lambda x: "Unit" not in x,\
                                         TrialDataTable.keys().to_numpy()))
            
            TrialTableNeuralData = TrialDataTable[NeuralContent]
            self.NeuralTableByTrial = TrialTableNeuralData # TODO: add a neural table of variable lengths for each trial depending on their trial-specific start/end times.  
            self.BehaviorTableByTrial = TrialDataTable[BehaviorContent]                  
            self.TimeOfInterest = TrialLength
            ###### TODO: decide whether we should update the TrialDataTable with the added neural spike times data ######
            # self.TrialDataTable = TrialDataTable
            
            
    def GroupTrialIDsOfInterest(self,TrialsOfInterest,TrialsTag):
        '''Lists of trial numbers for each condition of interest'''
        self.TrialsOfInterest[TrialsTag] = [x for x in TrialsOfInterest if x in self.NeuralTableByTrial.index.to_numpy()]

        
    def AverageAcrossTrials(self,TrialsOfInterest):
        '''Currently can't deal with neural data of different lengths per trial;
        can consider time-warping those trials in the future'''
        NeuralTableByTrial = self.NeuralTableByTrial.copy(deep=True)
        
        TrialAverageActivity = {}
        for keyindex, unitName in enumerate(NeuralTableByTrial.keys()):
            SingleTrialActivity = np.reshape(np.concatenate(NeuralTableByTrial[unitName][TrialsOfInterest].to_numpy()),\
                       [len(TrialsOfInterest),NeuralTableByTrial[unitName].iloc[1].shape[0]])
            # TrialAverageActivity[unitName] = NeuralTableByTrial[unitName][TrialsOfInterest].mean(skipna=True)
            TrialAverageActivity[unitName] = np.nanmean(SingleTrialActivity,axis=0)
        
        return TrialAverageActivity    


    def DefineTrialTags(self, switch_ids, switch_alignment, min_pokes = 0, max_pokes = 100): 
        ####### TODO: allow value-bin thresholds as an optional input
        '''Script of defining trial tags that include corresponding trials'''
        TrialsTag = "All" 
        TrialsOfInterest= self.TrialDataTable.index
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
    
        for i in switch_ids:
            TrialsTag = f"Switch{i}" 
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        for l in np.arange(1,7,1):
            for i in switch_ids:
                TrialsTag = f"Switch{i}StartLeaf{l}"       
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartLeaf==l)]
                self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)     
            
        for i in switch_ids:
            TrialsTag = f"Switch{i}StartingLeft"           
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.StartLeaf.isin([1,3,5]))]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)        
    
        for i in switch_ids:
            TrialsTag = f"Switch{i}StartingLeftEndingRight"
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.StartLeaf.isin([1,3,5]))&\
             (self.TrialDataTable.EndLeaf.isin([2,4,6]))]
            #Only select switch trials from a left leaf to a right leaf. TODO: make this flexible to user-defined end leaf location.
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)                   
    
        for i in switch_ids:
            TrialsTag = f"Switch{i}StartingRight"    
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.StartLeaf.isin([2,4,6]))]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)  
            
        for i in switch_ids:
            TrialsTag = f"Switch{i}StartingRightEndingLeft"
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.StartLeaf.isin([2,4,6]))&\
             (self.TrialDataTable.EndLeaf.isin([1,3,5]))]            
            #Only select switch trials from a right leaf to a left leaf. TODO: make this flexible to user-defined end leaf location.
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)                  
            
        for i in switch_ids:
            TrialsTag = f"Switch{i}StartingBest" # Stay trials in the high-rew patch and switch trials from the high-rew patch 
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.StartStem == self.TrialDataTable.BestStem)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
    
        for i in switch_ids:
            TrialsTag = f"Switch{i}StartingNonbest" # Stay trials in lower-rew patches and switch trials from lower-rew patches
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.StartStem != self.TrialDataTable.BestStem)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
            
        for i in switch_ids:
            TrialsTag = f"Switch{i}EndingBest" # Stay trials outside of the best stem and switch trials to the best stem
            if i == 0: # Switch trials ending in the best stem
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.EndStem == self.TrialDataTable.BestStem)]
            else:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem != self.TrialDataTable.BestStem)]            
                self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)        
    
        for i in switch_ids:
            TrialsTag = f"Switch{i}StartingA" 
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.StartStem == 'A')]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)            
        for i in switch_ids:
            TrialsTag = f"Switch{i}AtoB"
            if i == 0:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'A')&\
                 (self.TrialDataTable.EndStem == 'B')]
            else:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'A')]            
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)         
        for i in switch_ids:
            TrialsTag = f"Switch{i}AtoC"
            if i == 0:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'A')&\
                 (self.TrialDataTable.EndStem == 'C')]
            else:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'A')]            
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)                    
                
        for i in switch_ids:
            TrialsTag = f"Switch{i}StartingB" 
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.StartStem == 'B')]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        for i in switch_ids:
            TrialsTag = f"Switch{i}BtoA"
            if i == 0:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'B')&\
                 (self.TrialDataTable.EndStem == 'A')]
            else:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'B')]            
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)                
        for i in switch_ids:
            TrialsTag = f"Switch{i}BtoC"
            if i == 0:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'B')&\
                 (self.TrialDataTable.EndStem == 'C')]
            else:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'B')]            
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)      
                
        for i in switch_ids:
            TrialsTag = f"Switch{i}StartingC" 
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.StartStem == 'C')]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        for i in switch_ids:
            TrialsTag = f"Switch{i}CtoA"
            if i == 0:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'C')&\
                 (self.TrialDataTable.EndStem == 'A')]
            else:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'C')]            
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)      
        for i in switch_ids:
            TrialsTag = f"Switch{i}CtoB"
            if i == 0:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'C')&\
                 (self.TrialDataTable.EndStem == 'B')]
            else:
                TrialsOfInterest=\
                self.TrialDataTable.index\
                [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
                 (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
                 (self.TrialDataTable[switch_alignment] == i)&\
                 (self.TrialDataTable.StartStem == 'C')]            
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
                
        for i in switch_ids:
            TrialsTag = f"Switch{i}Rewarded" 
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.Reward == 1)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)  
    
        for i in switch_ids:
            TrialsTag = f"Switch{i}Unrewarded" 
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.pokes_before_switch >= min_pokes) &\
             (self.TrialDataTable.pokes_before_switch <= max_pokes) &\
             (self.TrialDataTable[switch_alignment] == i)&\
             (self.TrialDataTable.Reward == 0)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)       
            
        TrialsTag = "SwitchWithoutReward" 
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.IsSwitchPrevious == False)&\
          (self.TrialDataTable.IsSwitchPrevPrev == False)&\
          (self.TrialDataTable.RewardPrevious == False)&\
          (self.TrialDataTable.RewardPrevPrev == False)]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
      
        TrialsTag = "StayAfterNoReward" 
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == False) &\
         (self.TrialDataTable.IsSwitchPrevious == False)&\
          (self.TrialDataTable.IsSwitchPrevPrev == False)&\
          (self.TrialDataTable.RewardPrevious == False)&\
          (self.TrialDataTable.RewardPrevPrev == False)]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)

        TrialsTag = "SwitchWithReward"  
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.RewardPrevious == True)]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag) 
        
        TrialsTag = "StayAfterReward" 
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == False) &\
         (self.TrialDataTable.RewardPrevious == True)]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag) 
        
        value_thres = np.array(self.TrialDataTable.ValueDifference.quantile([0.25,0.5,0.75]))
        TrialsTag = "HighSwitchValue"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.ValueDifference >= value_thres[2]]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "MedHighSwitchValue"
        TrialsOfInterest=\
        self.TrialDataTable.index[np.logical_and(self.TrialDataTable.ValueDifference >= value_thres[1],self.TrialDataTable.ValueDifference <\
                                               value_thres[2])]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "MedLowSwitchValue"
        TrialsOfInterest=\
        self.TrialDataTable.index[np.logical_and(self.TrialDataTable.ValueDifference >= value_thres[0],self.TrialDataTable.ValueDifference <\
                                               value_thres[1])]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "LowSwitchValue"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.ValueDifference < value_thres[0]]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        for i in range(6):
            TrialsTag = f"HighValueStart{i+1}"
            TrialsOfInterest=\
            self.TrialDataTable.index[(self.TrialDataTable.ValueDifference >= value_thres[2]) &\
                                   (self.TrialDataTable.StartLeaf == i+1)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)    
        
        for i in range(6):
            TrialsTag = f"MedHighValueStart{i+1}"
            TrialsOfInterest=\
            self.TrialDataTable.index[np.logical_and(self.TrialDataTable.ValueDifference >= value_thres[1],self.TrialDataTable.ValueDifference <\
                                                   value_thres[2]) &\
                                   (self.TrialDataTable.StartLeaf == i+1)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        for i in range(6):
            TrialsTag = f"MedLowValueStart{i+1}"
            TrialsOfInterest=\
            self.TrialDataTable.index[np.logical_and(self.TrialDataTable.ValueDifference >= value_thres[0],self.TrialDataTable.ValueDifference <\
                                                   value_thres[1]) &\
                                   (self.TrialDataTable.StartLeaf == i+1)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)      
    
        for i in range(6):
            TrialsTag = f"LowValueStart{i+1}"
            TrialsOfInterest=\
            self.TrialDataTable.index[(self.TrialDataTable.ValueDifference < value_thres[0]) &\
                                   (self.TrialDataTable.StartLeaf == i+1)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)            
        
        value_thres = np.array(self.TrialDataTable.ExpectedValue.quantile([0.25,0.5,0.75]))
        TrialsTag = "HighExpectedValue"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.ExpectedValue >= value_thres[2]]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "MedHighExpectedValue"
        TrialsOfInterest=\
        self.TrialDataTable.index[np.logical_and(self.TrialDataTable.ExpectedValue >= value_thres[1],self.TrialDataTable.ExpectedValue < value_thres[2])]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
    
        TrialsTag = "MedLowExpectedValue"
        TrialsOfInterest=\
        self.TrialDataTable.index[np.logical_and(self.TrialDataTable.ExpectedValue >= value_thres[0],self.TrialDataTable.ExpectedValue < value_thres[1])]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)    
        
        TrialsTag = "LowExpectedValue"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.ExpectedValue < value_thres[0]]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "Reward"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.Reward == True]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "NoReward"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.Reward == False]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "Stay"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.IsSwitch == False]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "Switch"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.IsSwitch == True]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)  
        
        TrialsTag = "StayRewarded"
        TrialsOfInterest=\
        self.TrialDataTable.index[(self.TrialDataTable.Reward == True) & (self.TrialDataTable.IsSwitch == False)]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "StayUnrewarded"
        TrialsOfInterest=\
        self.TrialDataTable.index[(self.TrialDataTable.Reward == False) & (self.TrialDataTable.IsSwitch == False)]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "RewardBefore"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.RewardPrevious == True]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "NoRewardBefore"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.RewardPrevious == False]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "RewardLastLastTrial"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.RewardPrevPrev == True]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "NoRewardLastLastTrial"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.RewardPrevPrev == False]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "StartStemA"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.StartStem == 'A']
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "StartStemB"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.StartStem == 'B']
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "StartStemC"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.StartStem == 'C']
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemA"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.EndStem == 'A']
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemB"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.EndStem == 'B']
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemC"
        TrialsOfInterest=\
        self.TrialDataTable.index[self.TrialDataTable.EndStem == 'C']
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)       
        
        TrialsTag = "EndStemASwitch"
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.EndStem == 'A')]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemBSwitch"
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.EndStem == 'B')]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemCSwitch"
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.EndStem == 'C')]                 
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemBSwitchStartingA"
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.EndStem == 'B') &\
         (self.TrialDataTable.StartStem == 'A')]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemCSwitchStartingA"
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.EndStem == 'C') &\
         (self.TrialDataTable.StartStem == 'A')]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemASwitchStartingB"
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.EndStem == 'A') &\
         (self.TrialDataTable.StartStem == 'B')]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemCSwitchStartingB"
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.EndStem == 'C') &\
         (self.TrialDataTable.StartStem == 'B')]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemASwitchStartingC"
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.EndStem == 'A') &\
         (self.TrialDataTable.StartStem == 'C')]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        TrialsTag = "EndStemBSwitchStartingC"
        TrialsOfInterest=\
        self.TrialDataTable.index\
        [(self.TrialDataTable.IsSwitch == True) &\
         (self.TrialDataTable.EndStem == 'B') &\
         (self.TrialDataTable.StartStem == 'C')]
        self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)
        
        for i in np.arange(1,7,1):
            TrialsTag = f"EndLeaf{i}Switch"
            TrialsOfInterest=\
            self.TrialDataTable.index\
            [(self.TrialDataTable.IsSwitch == True) &\
             (self.TrialDataTable.EndLeaf == i)]
            self.GroupTrialIDsOfInterest(TrialsOfInterest,TrialsTag)