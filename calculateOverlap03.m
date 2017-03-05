function [overlap, interArea, unionArea ] = calculateOverlap03( detectbox, gtbox )
xDt          =  detectbox(1);
yDt          =  detectbox(2);
widthDt	= detectbox(3);
heightDt = detectbox(4);

xGt          = gtbox(1);
yGt          = gtbox(2);
widthGt  = gtbox(3);
heightGt = gtbox(4);

unionStartX    = min(xDt, xGt);
unionEndX      = max(xDt + widthDt, xGt + widthGt);
interWidth      = widthDt + widthGt - ( unionEndX - unionStartX );
unionStartY     = min(yDt, yGt);
unionEndY      = max(yDt + heightDt, yGt + heightGt);
interHeight     = heightDt + heightGt - ( unionEndY - unionStartY);
interArea = 0;
overlap = 0;
unionArea  = 0;
if interWidth > 0 && interHeight > 0  
    interArea	=  interWidth * interHeight;
    %---------------ICDAR03-standard ( hit = 0.5 + unionArea )---------------
    unionArea = ( unionEndY - unionStartY ) * ( unionEndX - unionStartX );
    %--------------ICDAR03-detval ( nonhit = 0 + meanArea )---------------
    %unionArea = ( widthDt * heightDt + widthGt * heightGt) / 2;
    overlap = interArea / unionArea;
end