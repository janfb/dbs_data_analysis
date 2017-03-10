clear all; close all; 

%% Load data 
data_folder = '/Users/Jan/Dropbox/Master/LR_Kuehn/data/dystonia_rest/'; 
data_files = {'spmeeg_031IU54', 'spmeeg_061ER68', 'spmeeg_seifert'};
D=spm_eeg_load([data_folder cell2mat(data_files(3))]);

% get GPi data into normal matlab format
ind=strmatch('LFP',D.chantype); 
ind2 = strmatch('GP',D.chanlabels); 
data=D(ind,:,:);
chans=D.chanlabels(ind);
time=D.time;
fsample=D.fsample;

%% Initial plot 
figure
plot(time,data); xlabel('Time [s]'); ylabel('mu V');
legend(chans)

%% zero mean the data to remove the DC component 
m = mean(data, 2);
data_centered = data - (m * ones(1, size(data,2)));

%% use pwelch to make a power spectrum
% input the data matrix transposed such that the PSD is calculated on each
% of the six columns, i.e., the 6 lfp channels
% ws=2*fsample;
% window = hanning(ws);
% [P,F] = pwelch(data', window, ws/2, [], fsample);

% make an initial plot of the PSD to see which frequencies are present 
% figure
% for j=1:size(data,1);
%     subplot(2,3,j); plot(F, P(:, j)); title(chans{j}); xlabel('Frequency [Hz]');
%     ylabel('PSD V^2 / Hz');
% end

%% filter more accurately
% design a butterworth band pass filter to restrict to a certain frequency
% band 
n = 2; 
nyq = 0.5 * fsample; 
start = 49 / nyq; 
stop = 51 / nyq;
[b, a] = butter(n, [start, stop], 'stop');
y = filtfilt(b, a, data');
data_filtered = y;

% calculate PSD with filtered data 
ws=2*fsample;
window = hanning(ws);
F = zeros(1025, 1); 
P = zeros(1025, size(data, 1));
for j=1:size(data,1);
    [P(:, j), F] = pwelch(data_filtered(:, j), window, ws/2, [], fsample);
end

% plot the new PSD 
figure
mask = F < 50;  % plot only up to frequ 50
for j=1:size(data,1);
    subplot(2,3,j); 
    plot(F(mask), P(mask, j)); 
    title(chans{j}); xlabel('Frequency [Hz]');
    ylabel('PSD V^2 / Hz');
end

%% plot the filtered data 
figure
plot(time, data_filtered'); xlabel('Time [s]'); ylabel('mu V');
legend(chans)
