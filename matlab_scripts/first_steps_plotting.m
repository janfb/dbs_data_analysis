clear all, close all

% the original data are converted into a format used by the SPM toolbox 
% we won't do much with the toolbox but we need to install it so we can
% load the data. You can get it from http://www.fil.ion.ucl.ac.uk/spm/

% load in some data
data_folder = '/Users/Jan/Dropbox/Master/LR_Kuehn/data/dystonia_rest/'; 
D=spm_eeg_load([data_folder 'spmeeg_031IU54']);

% some useful commands to check the properties of the data file:
D.fsample
D.chanlabels
D.conditions
D.chantype
D.ntrials
D.time;

% there are 6 lfp electrode, 3 in each GPi. order can be differetn between
% subjects.

% get GPi data into normal matlab format
ind=strmatch('LFP',D.chantype); 
data=D(ind,:,:);
chans=D.chanlabels(ind);
time=D.time;
fsample=D.fsample;

% make some plots to check data
% figure
for j=1:size(data,1);
    subplot(2,3,j); plot(time,data(j,:)); title(chans{j}); xlabel('Time [s]');
    ylabel('mu V');
end
% % or all in 1 figure
figure
plot(time,data); xlabel('Time [s]'); ylabel('mu V');
legend(chans)
% 
% units on the y-axis: mu V 

% to use default settings: [P,F] = pwelch(data(1,:),[],[],[],fsample)
% to play with window lengths: e.g. ws=2*fsample; [P,F] = pwelch(data(1,:),hanning(ws),ws/2,[],fsample)
% make some plots

% for filtering use butter + filtfilt
% [b,a] = butter(n,Wn,ftype)
% y = filtfilt(b,a,x)
% ...
 