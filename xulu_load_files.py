import os
import numpy as np
import pandas as pd
import scipy as sp
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

results_dir = '/home/xulu/code/results/'

def get_neural_behav_data_for_regression(file,selected_trials,selected_neurons='active_neurons',use_selected_sessions=False,use_updated_value=False,incl_pause_time=False,smooth_neural=0):
    bins_per_seg = 4 # Set as constant for now, might move it to function input when needed in the future.
    data_dir = '/home/xulu/code/data/'+file+'_ValueDecision/'
    
    if smooth_neural == 1:
        smooth_type = "GaussianSmooth"
    else:
        smooth_type = "NoSmooth"
    if incl_pause_time:
        data_df_switch_dir = data_dir+file+f"_data_df_{smooth_type}_mPFC_{int(bins_per_seg*2+3)}bins_WithOutcome+RewardHist+Uncertainty+ExpVal+PatchVal+PastVal+ValueDiff+ValueDiffMax+AltValue+StemStayP+SwitchTrialID+HeadDir+HeadAngle+Velocity.pkl"
    else:
        data_df_switch_dir = data_dir+file+f"_data_df_{smooth_type}_mPFC_{int(bins_per_seg*2)}bins_WithOutcome+RewardHist+ExpVal+PatchVal+PastVal+ValueDiff+ValueDiffMax+AltValue+StemStayP+SwitchTrialID+HeadDir+HeadAngle+Velocity.pkl"    
    if os.path.exists(data_df_switch_dir):
        data_df = pd.read_pickle(data_df_switch_dir)
    else:
        print('dataframe doesn not exist yet, please use the StemLeaf_single_neuron_response notebook to create it')
    # Add a column indicating trial num after a stem switch
    pokes_after_switch = data_df['trial id before switch'] + data_df['pokes before switch']
    pokes_after_switch[data_df['trial id before switch']==0] = 0
    data_df = pd.concat((data_df,pd.DataFrame({'trial id after switch':pokes_after_switch})),axis=1)    
    data_df.columns = data_df.columns.str.replace(" ", "_", regex=False) # replace all spaces in the column names of data_df by underscore

    # Load active neuron indices if pre-computed.
    if os.path.exists(results_dir+file+"_curated_active_neuron_index_0.5thres_everysesh.npy"):
        active_neurons = np.load(results_dir+file+"_curated_active_neuron_index_0.5thres_everysesh.npy")
    else:
        active_neurons = np.load(results_dir+file+"_curated_active_neuron_index_0.5thres.npy",allow_pickle=True)
    print(f'There are {active_neurons.shape[0]} active neurons.')
    # Load value neuron indices if pre-computed.
    value_ind_dir = results_dir+file+"_curated_value+valdiff_neuron_index_8bins.npy"
    if os.path.exists(value_ind_dir):
        value_neurons = np.load(value_ind_dir,allow_pickle=True)
        print(f'There are {value_neurons.shape[0]} value neurons.')

    # First trial of some sessions might have the same poke-in and poke-out times (likely because experimenter started recording after rats had started running), and we need to exclude them
    data_df = data_df[data_df.occupancy > 0]
   
    if incl_pause_time & use_updated_value:# Use the updated value difference estimate for pause times (i.e., the value difference max estimated for the next trial)
        condition = ((data_df['track_segment_id']/10).astype('int')) >=4
        data_df.loc[condition, 'value_difference_max'] = data_df.loc[condition, 'value_difference_next']
        data_df = data_df.loc[data_df['value_difference_max'].notna()]

    if selected_trials == "":
        data_df_selected = data_df # Use all trials
    if selected_trials == "_StayTrials":
        data_df_selected = data_df[data_df['stem_switch']==False]
    if selected_trials == "_EarlyStayTrials":
        data_df_selected = data_df[np.logical_and(data_df['pokes_before_switch']>= 7,data_df['trial_id_before_switch'].isin([-8,-7,-6,-5]))] # Only select early stay trials
    if selected_trials == "_LateStayTrials":
        data_df_selected = data_df[np.logical_and(data_df['pokes_before_switch']>= 7,data_df['trial_id_before_switch'].isin([-4,-3,-2,-1]))] # Only select late stay trials
    if selected_trials == "_StemA":
        data_df_selected = data_df[data_df['start leaf'].isin([1,2])] # Only select trials starting from leaves 3 and 4
        
    sess_selected = data_df_selected['session'].unique()
    if use_selected_sessions:
        sess_selected=find_specific_sessions(file,sess_selected)
    print(f'selected sessions {sess_selected}')
    data_df_selected = data_df_selected[data_df_selected['session'].isin(sess_selected)]    
    print(f"There're {data_df_selected.shape[0]} datapoints for regression.")

    # Position indicators that depends on specific leaves + left vs. right turns.
    if incl_pause_time:
        bins_all_segs = np.array([bins_per_seg,bins_per_seg,3]) # num of bins of start leaf, end leaf, outcome period
    else:
        bins_all_segs = np.array([bins_per_seg,bins_per_seg]) # num of bins of start leaf, end leaf

    pos_indicator_df = convert_pos_and_time_to_indicators(data_df_selected,bins_all_segs)
    # Position indicators (named 'turn_indicator') that only depends on left vs right turns but not specific starting leaves
    first_bin = 1    
    turn_indicator_df = create_turn_indicators_from_pos_and_time_indicators(pos_indicator_df,bins_all_segs,first_bin)

    # Position indicators that depend on progression and patch identity.
    patch_indicator_df = create_patch_indicators_from_pos_and_time_indicators(pos_indicator_df,bins_all_segs,first_bin)

    # Position indicators that only depends on goal progress
    prog_bin_sequence = np.arange(first_bin,first_bin+bins_all_segs.sum())
    progress_indicator_df = create_progress_indicators_from_turn_indicators(turn_indicator_df,prog_bin_sequence)
     
    # Get neural spikes data of value neurons.
    all_units = range(0,data_df_selected.filter(regex='spike').shape[1])
    if selected_neurons == 'value_neurons':
        predictor_neurons = value_neurons # Choose neurons that will be used in the predictor
    if selected_neurons == 'active_neurons': # By default we use all active neurons unless defined in the function input.
        predictor_neurons = active_neurons
    
    spikes_by_posbins = (data_df_selected.filter(regex='spike_train')).iloc[:,np.isin(all_units,predictor_neurons)]
    
    return sess_selected,data_df_selected,pos_indicator_df,turn_indicator_df,patch_indicator_df,progress_indicator_df,spikes_by_posbins,all_units,predictor_neurons

def find_specific_sessions(file,sess_orig):
    if np.isin(file,['j1620210714']):
        sess_sub = np.arange(1,7,1) # j1620210714 had 6 sessions total.
    # Late sessions with low reward rate
    if np.isin(file,['senor20201116','j1620210719']):
        sess_sub = sess_orig[[0,2,3,4]]
    if np.isin(file,['peanut20201205','wilbur20210406','wilbur20210408','wilbur20210409']):
        sess_sub = sess_orig[:-1]
    if np.isin(file,['peanut20201206']):
        sess_sub = sess_orig[1:]
    if np.isin(file, ['senor20201121']):
        sess_sub = sess_orig[[0,2,4]]
    if np.isin(file, ['peanut20201208']):
        sess_sub = sess_orig[[0,1,2,4]]
    if np.isin(file,['peanut20201209','senor20201113','senor20201114']):
        sess_sub = sess_orig[[0,1,3,4]]
    return sess_sub

def convert_pos_and_time_to_indicators(data,num_bins_all_segs):
    ###### num_bins_all_segs is an array sotring number of bins for each leaf segment and during pauses: start leaf, end leaf, pause time
    
    # Arrange the segment ids and head direction into a numpy array of position indicators
    pos_mat = np.zeros((data.shape[0],6*num_bins_all_segs.sum())) # Hard-coded "6" for the stem-leaf/6-arm maze
    column_names = np.empty((0,1))
    for i in range(6): # Hard-coded "6" for the stem-leaf/6-arm maze
        seg_counter = 1
        for j in np.arange(num_bins_all_segs[0]-1,-1,-1): # Start leaf
            bool_ind = np.logical_and(np.array(data['track_segment_id'],dtype='object')==i+10*j,np.array(data['lin_velocity'],dtype='object')<0)
            pos_mat[bool_ind,num_bins_all_segs.sum()*i+seg_counter-1]=1
            column_names = np.vstack((column_names,f"leaf {i+1}-{seg_counter}"))
            seg_counter = seg_counter+1
        for j in range(num_bins_all_segs[1]): # End leaf
            bool_ind = np.logical_and(np.array(data['track_segment_id'],dtype='object')==i+10*j,np.array(data['lin_velocity'],dtype='object')>0)            
            pos_mat[bool_ind,num_bins_all_segs.sum()*i+seg_counter-1]=1
            column_names = np.vstack((column_names,f"leaf {i+1}-{seg_counter}"))
            seg_counter = seg_counter+1
        if num_bins_all_segs.shape[0] == 3:
            for j in np.arange(num_bins_all_segs[1],num_bins_all_segs[1]+num_bins_all_segs[2]): # Pause time
                bool_ind = np.logical_and(np.array(data['track_segment_id'],dtype='object')==i+10*j,np.array(data['lin_velocity'],dtype='object')==0)
                pos_mat[bool_ind,num_bins_all_segs.sum()*i+seg_counter-1]=1
                column_names = np.vstack((column_names,f"leaf {i+1}-{seg_counter}"))
                seg_counter = seg_counter+1
            
    column_names = column_names.flatten()
    pos_ind_df = pd.DataFrame(pos_mat, columns=column_names,index=data.index[:])
    return pos_ind_df


def create_turn_indicators_from_pos_and_time_indicators(pos_indicator,num_bins_per_seg,start_bin):
    # Position indicators (named 'turn_indicator') that only depends on left vs right turns but not specific starting leaves
    left_turn = (pos_indicator.filter(regex = f'-{start_bin}$').iloc[:,[0,2,4]]).sum(axis=1)
    right_turn = (pos_indicator.filter(regex = f'-{start_bin}$').iloc[:,[1,3,5]]).sum(axis=1)
    for i in np.arange(start_bin+1,start_bin+num_bins_per_seg[0]):
        left_turn = pd.concat([left_turn,(pos_indicator.filter(regex = f'-{i}$').iloc[:,[0,2,4]]).sum(axis=1)],axis=1)
        right_turn = pd.concat([right_turn,(pos_indicator.filter(regex = f'-{i}$').iloc[:,[1,3,5]]).sum(axis=1)],axis=1)   
    for i in np.arange(num_bins_per_seg[0]+1,num_bins_per_seg.sum()+1):
        left_turn = pd.concat([left_turn,(pos_indicator.filter(regex = f'-{i}$').iloc[:,[1,3,5]]).sum(axis=1)],axis=1)
        right_turn = pd.concat([right_turn,(pos_indicator.filter(regex = f'-{i}$').iloc[:,[0,2,4]]).sum(axis=1)],axis=1)   
    bin_sequence = np.arange(start_bin,start_bin+num_bins_per_seg.sum())
    left_turn.columns = [f'left turn {i}' for i in bin_sequence]
    right_turn.columns = [f'right turn {i}' for i in bin_sequence]
    return pd.concat([left_turn,right_turn],axis=1)


def create_patch_indicators_from_pos_and_time_indicators(pos_indicator,num_bins_per_seg,start_bin):
    ########## TODO: use a loop to create indicators for the 3 stems rather than repeating the same code three times
    patchA_indicator = pos_indicator.filter(like='leaf 1-1').join(pos_indicator.filter(like='leaf 2-1')).sum(axis=1)
    patchB_indicator = pos_indicator.filter(like='leaf 3-1').join(pos_indicator.filter(like='leaf 4-1')).sum(axis=1)
    patchC_indicator = pos_indicator.filter(like='leaf 5-1').join(pos_indicator.filter(like='leaf 6-1')).sum(axis=1)
    for i in np.arange(start_bin+1,start_bin+num_bins_per_seg.sum()):
        patchA_indicator = pd.concat([patchA_indicator,pos_indicator.filter(like=f'leaf 1-{i}').join(pos_indicator.filter(like=f'leaf 2-{i}')).sum(axis=1)],axis=1)
        patchB_indicator = pd.concat([patchB_indicator,pos_indicator.filter(like=f'leaf 3-{i}').join(pos_indicator.filter(like=f'leaf 4-{i}')).sum(axis=1)],axis=1)
        patchC_indicator = pd.concat([patchC_indicator,pos_indicator.filter(like=f'leaf 5-{i}').join(pos_indicator.filter(like=f'leaf 6-{i}')).sum(axis=1)],axis=1)
    bin_sequence = np.arange(start_bin,start_bin+num_bins_per_seg.sum())
    patchA_indicator.columns = [f'stemA {i}' for i in bin_sequence]
    patchB_indicator.columns = [f'stemB {i}' for i in bin_sequence]
    patchC_indicator.columns = [f'stemC {i}' for i in bin_sequence]
    return pd.concat([patchA_indicator,patchB_indicator,patchC_indicator],axis=1)


def create_progress_indicators_from_turn_indicators(turn_indicator,bin_sequence):
    # Progression indicators that only depend on the fraction of a trial that the rat has completed
    progress_indicator = turn_indicator.filter(regex = f'turn {bin_sequence[0]}$').sum(axis=1)
    for i in bin_sequence[1:]:
        progress_indicator = pd.concat([progress_indicator,turn_indicator.filter(regex = f'turn {i}').sum(axis=1)],axis=1)
    progress_indicator.columns = [f'bin {i}' for i in bin_sequence]
    return progress_indicator