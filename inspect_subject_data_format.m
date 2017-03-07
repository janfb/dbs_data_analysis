% look in the given directory for filenames to load with spm. then look for
% the number of LFP channels. print a list of filenames and the
% corresponding numebr of channels as an overview 

clear all; close all; 

%% Load data 
data_folder = '/Users/Jan/Dropbox/Master/LR_Kuehn/data/dystonia_rest/'; 
file_list = dir(data_folder);

lfp_channels = [];
filenames = {};
file_outliers = {};  % make a list of all files with other than 6 GPi channels
counter = 1; 
counter2 = 1; 
for i=1:size(file_list, 1); 
    [pathstr,name,ext] = fileparts(file_list(i).name);     
    if strcmp(ext, '.mat') 
        D=spm_eeg_load([data_folder name]);

        % get GPi data into normal matlab format
        ind=strmatch('GPi',D.chanlabels);
        data=D(ind,:,:);
        chans=D.chanlabels(ind);
        time=D.time;
        fsample=D.fsample;
        lfp_channels = [lfp_channels, size(chans, 2)];
        filenames{counter} = name;
        if size(chans, 2) ~= 6  % check the number  GPi channels
            file_outliers{counter2} = name;
            counter2 = counter2 + 1;
        end
        counter = counter + 1; 
    end
end

display(lfp_channels)
display(filenames)
display(file_outliers)
