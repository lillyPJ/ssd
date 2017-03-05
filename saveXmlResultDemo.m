% saveXmlResultDemo
dtDir = '/home/lili/Desktop/到 ICDAR2013 的链接/test_bb';
%dtDir = '/home/lili/datasets/ICDAR2013/gt/test/txt/word';

CASE = 'test';
LEVEL = 'word';
baseDir = '/home/lili/datasets/ICDAR2013';
imgDir = fullfile(baseDir, 'img', CASE);
images = dir(fullfile(imgDir, '*.jpg'));
nImg = length(images);
saveBase = fileparts(dtDir);
saveXMLDir= fullfile(saveBase, 'test_xml');
mkdir(saveXMLDir);
saveXMLPath = fullfile(saveXMLDir, 'test.xml');
%%
resultBox = cell(nImg,2);
for i=1: nImg
    imageName = fullfile(imgDir, images(i).name);
    tempName = images(i).name;
    % dt
    dtFile = fullfile(dtDir, ['res_', tempName(1:end-3), 'txt']);
    box = importdata(dtFile);
    resultBox{i,1} = [CASE, '/', images(i).name];
    if ~isempty(box)
        box(:,3) = box(:,3) - box(:,1);
        box(:,4) = box(:,4) - box(:,2);
        %box = myNms(box, 0.25);
        box = myNms2(box, 1.3);
        resultBox{i,2} = floor(box(:,1:4));
%         imshow(imread(imageName));
%         displayBox(resultBox{i,2});
%         title(tempName);
    end
end
saveXMLFiles(saveXMLPath, resultBox);