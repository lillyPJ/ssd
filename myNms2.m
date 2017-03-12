function box = myNms2( box, scoreThreshold, overlap1Threshold, overlap2Threshold )
 if nargin < 3
     overlap1Threshold = 0.5;
     overlap2Threshold = 0.25;
 end
 if nargin < 4
     overlap2Threshold = 0.25;
 end
box = sortrows( box, -5);
box = bbNms(box,'type','cover','overlap', overlap1Threshold);

if ~isempty(box)
    box = box((box(:,5)> scoreThreshold),:);
    box = bbNms(box,'type','maxg','overlap', overlap2Threshold);
    %box = box(:,1:4);
end