% load barow et al data and save a new file for every subject in python 
% readable format holding the lfp data for all three conditions and the
% meta data 

data_folder = '/Users/Jan/Dropbox/Master/LR_Kuehn/data/dystonia_stim/'; 
file_list = dir([data_folder '*ps_*.mat']);
display(file_list); 
id_list = {};

% find all the patient IDs 
for f = 1:length(file_list)
    file_name = file_list(f).name;
    id = file_name(15:length(file_name)-4);
    id_list(f) = cellstr(id);
end

%% iterate over subject ids 
counter = 1; 
for p = 1:length(id_list)
    % collect all files of that subject 
    subject_files = dir([data_folder '*' cell2mat(id_list(p)) '.mat']);
    
    % iterate over the file 
    for f = 1:length(subject_files)            
        file_name = subject_files(f).name;
        display(file_name);         
        % extract the condition 
        condition = file_name(12:length(file_name)-(5 + length(cell2mat(id_list(p))))); 
        display(condition);
        
        % load the file 
        D = spm_eeg_load([data_folder subject_files(f).name]);
        ind = strmatch('GPi',D.chanlabels); 
        switch condition
            case 's'
                condition_str = 'stim';
            case 'ps'
                condition_str = 'poststim';
            case 'r'
                condition_str = 'rest';
            case 'd'
                condition_str = 'd';
        end
        
        data = D(ind, :, :);
        chanlabels = D.chanlabels(ind);
        time = D.time;
        fsample = D.fsample;
        subject_id = cell2mat(id_list(p));
        
        filename_new = ['spmeeg_' num2str(counter) '_' condition_str];
        % display([data_folder 'for_python/' filename_new]);
        save([data_folder 'for_python/' filename_new], 'fsample', 'time', 'chanlabels', 'data', 'subject_id', 'condition_str');
        condition_str = {};
    end
    counter = counter + 1; 
end
