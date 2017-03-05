function [recall, precision, fscore , evalInfo] = evalDetBox03( detectBox, gtBox, varargin )
% To evaluate the detection results
%
% USAGE
% [recall, precision, fscore] = evalDetBox( detectBox, gtBox, varargin )
%
% INPUTS
% - detectBox
%  	- detection results with each line as a 4-d box [x, y, w, h], the total dimension is nD*4
% - gtBox
%  	- groundTruth boxes with each line as a 4-d box [x, y, w, h], the total dimension is nG*4
% - varargin
%  	- 'overlapThr'[0,1], default = 0.5(HIT criteria)
%       - the threshold of the overlap area. When IOU ( intersection / union
%       of the test and the gt box ) is greater than the threshold, it
%       counts. (HIT criteria)
%       - Especially, when  overlapThr = 0, use the 'NOHIT' criteria.
%
% OUTPUTS
% - recall          - recall of the results
% - precision       - precision of the results
% - fscore
%   - fscore of the results ( alpha = 0.5 ) :
%      fscore = 2 / (1/recall + 1/precision)
% - others
%   - others.nD  = nD, num of detection box
%   - others.nG  = nG, num of gt box
%   - others.tr    = tr, sum of score of gt
%   - others.tp   = tp, sum of score of detection
%   - others.dt   = dt [x y w h matchScore], column vector with the same size of the detectBox.
%   - others.gt   = gt [x y w h matchScore], column vector with the same size of the gtBox.

%
% EXAMPLE
%  dfs = { 'overlapThr', 0.5 };
%  detectBox = [ 11, 62, 392, 55; 8, 128, 584, 186];
%  gtBox = [ 9, 130, 581, 180; 12, 65, 309, 50; 349, 68, 238, 47];
%  prm = evalDetBox( detectBox, gtBox, dfs )

%% get parameters
dfs={'overlapThr', 0.5};
params = getPrmDflt( varargin, dfs);
overlapThr = params.overlapThr;
assert( overlapThr >= 0 );
assert( overlapThr <= 1 );
% check input
if( nargin < 2 )
    error( 'Input must include at least detectBox and gtBox !' );
end
%% initialization
recall = 0;
precision = 0;
fscore = 0;
nD = size( detectBox, 1 );
nG = size( gtBox, 1 );
evalInfo.empty = true;
evalInfo.nD = nD;
evalInfo.nG = nG;
evalInfo.tr = 0;
evalInfo.tp = 0;
evalInfo.dt = [];
evalInfo.gt = [];
evalInfo.scoreMatrix = [];
if( nD == 0 || nG == 0 )
    return;
end
assert( size(detectBox,2) == 4 );
assert( size(gtBox,2) == 4 );

%% sort
% gtBox = sortrows( gtBox, 1);
% detectBox = sortrows( detectBox, 1 );
dt = horzcat( detectBox, zeros( nD, 1 ) );
gt = horzcat( gtBox, zeros( nG, 1 ) );
%%  calculate scoreMatrix
scoreMatrix = zeros( nD, nG );
for i=1:nD
    for j=1:nG
        scoreMatrix(i, j) = calculateOverlap03( detectBox(i,:), gtBox(j,:) );
    end
end
%% assign detection
for i=1:nD
    dt(i, 5) = max( scoreMatrix(i, :) );
end
%% assign gt
for j=1:nG
    gt(j, 5) = max( scoreMatrix(:, j) );
end
%% calculate the criterion
if overlapThr > 0  % HIT
    ind = ( dt(:, 5) > overlapThr );
    dt(ind, 5) = 1;
    dt(~ind, 5) = 0;
    ind = ( gt(:, 5) > overlapThr );
    gt(ind, 5) = 1;
    gt(~ind, 5) = 0;
end
tr = sum( gt(:, 5) );
tp = sum( gt(:, 5) );
recall = tr / nG * 100;
precision = tp / nD * 100;
fscore = 2/( 1/recall + 1/precision);
% all results;
if( nargout > 3 )
    evalInfo.empty = false;
    evalInfo.nD  = nD;
    evalInfo.nG  = nG;
    evalInfo.tr    = tr;
    evalInfo.tp   = tp;
    evalInfo.dt   = dt;
    evalInfo.gt   = gt;
    evalInfo.scoreMatrix = scoreMatrix;
end
