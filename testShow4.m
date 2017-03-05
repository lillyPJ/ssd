% testShowDemo
dtDir = '/home/lili/codes/ssd/caffe-ssd/data/shopSign/test_bb/';
%dtDir = '/home/lili/datasets/ICDAR2013/gt/test/txt/word';

CASE = 'test';
LEVEL = 'word';
baseDir = '/home/lili/datasets/shopSign/';
imgDir = fullfile(baseDir, 'img', CASE);
images = dir(fullfile(imgDir, '*.jpg'));
nimage = length(images);
%%
for i=1: nimage
%     if i < 33
%         continue
%     end
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
%         box = box(((box(:,4)>0)&(box(:,3) > 0)), :);
%         
%         h_w = box(:,4)./box(:,3);
%         hwIdx = h_w < 2;
%         rightIdx = hwIdx;
%         box = box(rightIdx,:);
%         oldbox1 = box;
        
        displayBox(box, 'r');
        %box = bbNms(box,'type','cover','overlap', 0.5);
        box = myNms(box, 0.25); 
    end
    
    oldbox2 = box;
    if ~isempty(box)
        box = box((box(:,5)> 0),:);
    end
    % gt
%     dtFile = fullfile(dtDir, ['gt_', tempName(1:end-3), 'txt']);
%     [box, tag, word] = loadGTFromTxtFile( dtFile );
    
%     box(:,4) = box(:,6) - box(:,2);
%     box(:,3) = box(:,5) - box(:,1);
%     rightIdx = (box(:,4)>0)&(box(:,3) > 0);
%     box = box(rightIdx,:);
    fprintf('%d: %s\n', i, tempName);
    displayBox(box,'g');
    title(tempName);
end
