% testShowDemo
dtDir = '/home/lili/codes/ssd/caffe-ssd/data/TextBoxes/test_bb_multi_scale/';
%dtDir = '/home/lili/datasets/ICDAR2013/gt/test/txt/word';

CASE = 'test';
LEVEL = 'word';
baseDir = '/home/lili/datasets/ICDAR2013/';
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
    %box = importdata(dtFile);
    [x1,y1,x2,y2,x3,y3,x4,y4,score,size_num]=textread(dtFile,'%d %d %d %d %d %d %d %d %f %d');
    bbs_dt = [x1,y1,x2,y2,x3,y3,x4,y4];

%     nms_flag=nms(bbs_dt',score,'overlap',0.25);
%     bbs_dt=bbs_dt(nms_flag==true,:);
%     score=score(nms_flag==true,:);
%     bbs_dt=bbs_dt(score>0.9,:);
    box = [bbs_dt(:,1:2), bbs_dt(:,5:6), score];
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
        box = box(((box(:,4)>0)&(box(:,3) > 0)), :);
        
        h_w = box(:,4)./box(:,3);
        hwIdx = h_w < 2;
        rightIdx = hwIdx;
        box = box(rightIdx,:);
        oldbox1 = box;
        %displayBox(box, 'r');
        box = bbNms(box,'type','cover','overlap', 0.5);
        %box = myNms(box, 0.25); 
    end
    
    oldbox2 = box;
    if ~isempty(box)
    box = box((box(:,5)> 1.3),:);
    %box = box((box(:,5)> 0.8),:);
    box = bbNms(box,'type','maxg','overlap', 0.25);
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
