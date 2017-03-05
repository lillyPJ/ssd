clear all;
icdar_2013_dataset_root = '/home/lili/datasets/VOC/VOCdevkit/CASIA/';
test_list_file=[icdar_2013_dataset_root, 'ImageSets/Main/test.txt'];
img_dir=[icdar_2013_dataset_root,'JPEGImages/'];
gt_dir= '/home/lili/datasets/CASIA/gt/test/txt/';
% multi-scale detection results directory
dt_dir = '/home/lili/codes/ssd/caffe-ssd/data/CASIA/test_bb/';
%dt_dir = '/home/lili/codes/ssd/caffe-ssd/data/TextBoxes/test_bb_multi_scale/';
% save results to upload to ICDAR 2015 website
icdar_test_dir='./icdar_submit_results/';
mkdir(icdar_test_dir)
[img_name]=textread(test_list_file,'%s');

%settings
info.iou_thr = 0.5;
nImg = length(img_name);
gt = cell(nImg,1);
dt = cell(nImg,1);
box_num=0;
time = 0;
for ii=1:nImg
    name = img_name(ii);
    name=char(name);
    img=imread([img_dir,name, '.jpg']);
    [img_height,img_width,~]=size(img);
    gt_path = [ gt_dir,name,'.txt'];
    gtbox = [];
    dtbox = [];
    [x1,y1,x3,y3,str]=textread(gt_path,'%d,%d,%d,%d,%s%*[^\n]');
    x2=x3;
    y2=y1;
    x4=x1;
    y4=y3;
    bbs_gt = [x1,y1,x2,y2,x3,y3,x4,y4];
    gtbox = [gtbox; x1,y1,(x3-x1+1),(y3-y1+1)];
    
    gt{ii}=bbs_gt;
    dt_path = [dt_dir,'res_', name,'.txt'];
    %% input file
    %--------multi: dt
     
%     [x1,y1,x2,y2,x3,y3,x4,y4,score,size_num]=textread(dt_path,'%d %d %d %d %d %d %d %d %f %d');
%     bbs_dt = [x1,y1,x2,y2,x3,y3,x4,y4];
%     dtbox = [bbs_dt(:,1:2), bbs_dt(:,5:6), score];
%     dtbox(:,3) = dtbox(:,3) - dtbox(:,1);
%     dtbox(:,4) = dtbox(:,4) - dtbox(:,2);
    %----------single:
    [x1,y1,x3,y3,score]=textread(dt_path,'%d,%d,%d,%d,%f');
    x2=x3;
    y2=y1;
    x4=x1;
    y4=y3;
    bbs_dt = [x1,y1,x2,y2,x3,y3,x4,y4];
    dtbox = [dtbox; x1,y1,(x3-x1+1),(y3-y1+1), score];
    %% nms
    dtbox = myNms2(dtbox, 1.3);
%    dtbox = sortrows( dtbox, -5);
%      dtbox = bbNms(dtbox,'type','cover','overlap', 0.5);
%      dtbox = dtbox((dtbox(:,5)> 1.3),:);
%      dtbox = bbNms(dtbox,'type','maxg','overlap', 0.25);
%      %dtbox = bbNms(dtbox,'type','maxg','overlap', 0.4, 'ovrDnm', 'min');
%      %dtbox = bbNms(dtbox,'type','max','overlap', 0.6, 'ovrDnm', 'min');
%      if ~isempty(dtbox)
%          dtbox = dtbox(:,1:4);
%      end
    %%
%     if ~isempty(dtbox)
%         h_w = dtbox(:,4)./dtbox(:,3);
%         hwIdx = h_w < 2;
%         %rightIdx = (dtbox(:,4)>0)&(dtbox(:,3) > 0)& hwIdx;
%         rightIdx = (dtbox(:,4)>0)&(dtbox(:,3) > 0);
%         %dtbox = dtbox(rightIdx,:);
%         dtbox = myNms(dtbox, 0.25);
%         if ~isempty(dtbox)
%           dtbox = dtbox(:,1:4);
%         end
%     end
    tic;
    %dtbox = dtbox((dtbox(:,5)> 0.7),1:4);
   
    [recall, precision, fscore, evalInfo(ii) ] = evalDetBox03( dtbox, gtbox );
    time = time + toc;
    score = ones(size(bbs_dt,1),1);
    nms_flag=nms(bbs_dt',score,'overlap',0.25);
    bbs_dt=bbs_dt(nms_flag==true,:);
    
    dt{ii}=double(bbs_dt);
end
%
%computation p,r,f-measure
tic;
detection_count = 0;
gt_count = 0;
hit_recall = 0;
hit_precision=0;
for ii=1:nImg
    bbs_dt = dt{ii};
    if(~isempty(bbs_dt))
        flag_strick = zeros(size(bbs_dt,1), 1);
        for j=1:size(gt{ii},1)
            for i=1:size(bbs_dt,1)
                x_union = [bbs_dt(i,1:2:8),gt{ii}(j,1:2:8)];
                y_union = [bbs_dt(i,2:2:8),gt{ii}(j,2:2:8)];
                union_poly_ind = convhull(x_union, y_union);
                union_area = polyarea(x_union(union_poly_ind), y_union(union_poly_ind));
                insect_area = polygon_intersect(bbs_dt(i,1:2:8),bbs_dt(i,2:2:8), ...
                    gt{ii}(j,1:2:8), gt{ii}(j,2:2:8));
                
                flag_strick(i) = ((insect_area / union_area) > info.iou_thr);
            end
            if(sum(flag_strick) > 0)
                hit_recall = hit_recall + 1;
                hit_precision=hit_precision+1;
            end
        end
        detection_count = detection_count + size(bbs_dt,1);
        gt_count = gt_count + size(gt{ii},1);
    else
        gt_count = gt_count + size(gt{ii},1);
    end
end
recall =  hit_recall / gt_count
precision = hit_precision / detection_count
f_measure = 2 * recall * precision / (recall + precision)
disp('over!');
time0 = toc;
%}
tic;
recall2 =  sum( [evalInfo.tr] ) / sum( [evalInfo.nG] )
precision2 = sum( [evalInfo.tp] ) / sum( [evalInfo.nD] )
f_measure2 = 2 * recall2 * precision2 / (recall2 + precision2)
time = time + toc;
fprintf('time0 = %.3f, time = %.3f\n',time0, time);
fprintf('my:recall = %.3f, precision = %.3f, fmeasure = %.3f\n', recall2, precision2, f_measure2);
fprintf('bx:recall = %.3f, precision = %.3f, fmeasure = %.3f\n', recall, precision, f_measure);
