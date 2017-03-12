% showTestDemo
dtDir = '/home/lili/codes/ssd/caffe-ssd/data/CASIA/test_bb';
%dtDir = '/home/lili/datasets/ICDAR2013/gt/test/txt/word';

CASE = 'test';
MULTI = 1; %0-single, 1-multi
baseDir = '/home/lili/datasets/CASIA/';
imgDir = fullfile(baseDir, 'img', CASE);
images = dir(fullfile(imgDir, '*.jpg'));
nimage = length(images);
%%
for i=1: nimage
    imageName = fullfile(imgDir, images(i).name);
    img = imread(imageName);
    tempName = images(i).name;
    % dt
    dtFile = fullfile(dtDir, ['res_', tempName(1:end-3), 'txt']);
    box = importdata(dtFile);
    imshow(img);
    
    
    if ~isempty(box)
        box0 = box;
%         nbox = size(box,1);
%         box0 = box;
%         for j = 1: nbox
%             box(j,1) = min([box0(j,1), box0(j,3)]);
%             box(j,2) = min([box0(j,2), box0(j,4)]);
%             box(j,3) = max([box0(j,1), box0(j,3)]);
%             box(j,4) = max([box0(j,2), box0(j,4)]);
%         end
        box(:,3) = box(:,3) - box(:,1);
        box(:,4) = box(:,4) - box(:,2);
        % delete those negative box
     
    end
    if MULTI
        newbox = myNms2(box, 1.3);
    else
        newbox = myNms(box, 0.25);
    end

    % gt
%     dtFile = fullfile(dtDir, ['gt_', tempName(1:end-3), 'txt']);
%     [box, tag, word] = loadGTFromTxtFile( dtFile );
    
%     box(:,4) = box(:,6) - box(:,2);
%     box(:,3) = box(:,5) - box(:,1);
%     rightIdx = (box(:,4)>0)&(box(:,3) > 0);
%     box = box(rightIdx,:);
    fprintf('%d: %s\n', i, tempName);
    displayBox(newbox,'g', 'u');
    title(tempName);
end
