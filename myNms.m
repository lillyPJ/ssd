function newbox = myNms( box, overlapThreshold )
if nargin < 2
    overlapThreshold = 0.75;
end
if isempty( box )
    newbox = [];
    return;
end

if( size( box, 2) < 5 )
        box(:, 5) = box(:,3) .* box( :, 4);
end
nbox = size( box, 1 );
box = sortrows( box, 5);
if  nbox > 5000
    % num of box is large
    newNbox = min( ceil( nbox.*0.5 ), 5000 );
    box = box( (end - newNbox) : end, : );
end
newbox=bbNms(box,'type','maxg','overlap', overlapThreshold);
%newbox = newbox(: ,1:4 );
    
        