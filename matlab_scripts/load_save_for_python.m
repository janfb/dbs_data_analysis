% load the spm file of a subject and save it as a matfile that is readable
% in python 

data_folder = '/Users/Jan/Dropbox/Master/LR_Kuehn/data/dystonia_rest/'; 
data_files = {'spmeeg_031IU54','spmeeg_061ER68','spmeeg_066AC76','spmeeg_081WO65','spmeeg_082RO76','spmeeg_110AI58','spmeeg_112AU57','spmeeg_132EUe68','spmeeg_157AI64','spmeeg_175AO64','spmeeg_216NA68','spmeeg_220GA63','spmeeg_250AC59','spmeeg_busch','spmeeg_gaede','spmeeg_holland','spmeeg_hopp','spmeeg_horst','spmeeg_kieslich','spmeeg_koenig','spmeeg_kroke','spmeeg_lorenz','spmeeg_mueller','spmeeg_naumann','spmeeg_schroeter','spmeeg_seifert','spmeeg_spuling'};
counter = 1; 
for f = 1:length(data_files)
    filename = cell2mat(data_files(f));
    D = spm_eeg_load([data_folder filename]);

    % get GPi data into normal matlab format
    ind = strmatch('GP',D.chanlabels); 
    data = D(ind,:,:);
    chanlabels = D.chanlabels(ind);
    time = D.time;
    fsample = D.fsample;


    filename_new = ['spmeeg_' num2str(counter)];
    counter = counter + 1; 
    % display([data_folder 'for_python/' filename_new]);
    save([data_folder 'for_python/' filename_new], 'fsample', 'time', 'chanlabels', 'data')
end