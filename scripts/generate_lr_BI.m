function generate_lr_bic()

up_scale = 4;
mod_scale = 4;
idx = 0;
filepaths = dir('/home/data/djn/Datasets/Vimeo200-Subset/train_sharp/*/*.png');

for i = 1 : length(filepaths)
    [~,imname,ext] = fileparts(filepaths(i).name);
    folder_path = filepaths(i).folder;
    save_lr_folder = strrep(folder_path, 'train_sharp', 'train_bicubic4x_lr')
	  save_bic_folder = strrep(folder_path, 'train_sharp', 'train_bicubic4x_bic')
    if ~exist(save_lr_folder, 'dir')
        mkdir(save_lr_folder);
    end
	if ~exist(save_bic_folder, 'dir')
        mkdir(save_bic_folder);
    end
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_rlt);
        % read image
        img = imread(fullfile(folder_path, [imname, ext]));
        img = im2double(img);
        % modcrop
        img = modcrop(img, mod_scale);
        % LR
        im_LR = imresize(img, 1/up_scale, 'bicubic');
        im_BI = imresize(im_LR, up_scale, 'bicubic');
        if exist('save_lr_folder', 'var')
            imwrite(im_LR, fullfile(save_lr_folder, [imname, '.png']));
        end
        if exist('save_bic_folder', 'var')
            imwrite(im_BI, fullfile(save_bic_folder, [imname, '.png']));
        end
    end
end
end

%% modcrop
function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end
