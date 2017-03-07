clear all; close all; 
% load the data of many subjects and calculate the psd, look at the grand
% mean psd 

data_folder = '/Users/Jan/Dropbox/Master/LR_Kuehn/data/dystonia_rest/'; 
files_to_process = {};
file_list = dir(data_folder);
counter = 1;
for i=1:size(file_list, 1); 
    [pathstr,name,ext] = fileparts(file_list(i).name);
    if strcmp(ext, '.mat') 
        % add data to file list 
        files_to_process{counter} = file_list(i).name; 
        counter = counter + 1;
    end
end


F = zeros(1025, 1); 
P = zeros(1025, 6, size(files_to_process, 2));
for i=1:size(files_to_process, 2); 
    % load data 
    D=spm_eeg_load([data_folder cell2mat(files_to_process(i))]);
    counter = counter + 1;

    % save meta data 
    ind = strmatch('GPi',D.chanlabels); 
    data=D(ind,:,:);
    chans=D.chanlabels(ind);
    time=D.time;
    fsample=D.fsample;

    % zero mean 
    m = mean(data, 2);
    data_centered = data - (m * ones(1, size(data,2)));

    % design the filter to remove hum noise at 50Hz 
    n = 2; 
    nyq = 0.5 * fsample; 
    start = 49 / nyq; 
    stop = 51 / nyq;
    [b, a] = butter(n, [start, stop], 'stop');
    y = filtfilt(b, a, data_centered'); 

    ws=2*fsample;
    window = hanning(ws);   

    for j=1:size(data,1);
        [P(:, j, i), F] = pwelch(y(:, j), window, ws/2, [], fsample);
    end
end

% take the mean over subjects 
p_mean = mean(P, 3);

%% Plotting 
for j=1:6;
    subplot(2,3,j); 
    plot(F, p_mean(:, j)); 
    title(chans{j}); 
    xlabel('Frequency [Hz]');
    ylabel('PSD V^2 / Hz');
end
