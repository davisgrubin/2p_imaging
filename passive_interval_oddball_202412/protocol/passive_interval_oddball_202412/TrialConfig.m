classdef TrialConfig
    methods


% input numbers to set session parameters
function [DefaultConfig, FixJitterMode, StandardMode, OddballMode, OptoMode] = GetConfig(obj)
    fprintf('\n\nInitiating passive protocol \n\n')
    fprintf('default 1: 3331Random \n')
    fprintf('default 2: 1451ShortLong \n')
    fprintf('default 3: 4131FixJitterOdd \n')
    fprintf('default 4: Extended3331Random \n')
    fprintf('default 5: costomized \n')
    DefaultConfig = input('Input number to set default parameters >> ');
    if (DefaultConfig==5)
        fprintf('\nFixJitterMode 1: fix \n')
        fprintf('FixJitterMode 2: jitter \n')
        fprintf('FixJitterMode 3: random \n')
        fprintf('FixJitterMode 4: block \n')
        FixJitterMode = input('Input number to set fix or jitter >> ');
        fprintf('\nStandardMode 1: short \n')
        fprintf('StandardMode 2: long \n')
        fprintf('StandardMode 3: random \n')
        fprintf('StandardMode 4: block \n')
        StandardMode = input('Input number to set standard mode >> ');
        fprintf('\nOddballMode 1: short \n')
        fprintf('OddballMode 2: long \n')
        fprintf('OddballMode 3: random \n')
        fprintf('OddballMode 4: block \n')
        fprintf('OddballMode 5: reverse \n')
        OddballMode = input('Input number to set oddball mode >> ');
        fprintf('\nOptoMode 1: off \n')
        fprintf('OptoMode 2: on \n')
        fprintf('OptoMode 3: default \n')
        fprintf('OptoMode 4: block \n')
        fprintf('OptoMode 5: Random \n')
        OptoMode = input('Input number to set opto mode >> ');
    else
        FixJitterMode = -1;
        StandardMode = -1;
        OddballMode = -1;
        OptoMode = -1;
    end
end


% extend type sequence to prevent crash
function [ExtendedSeq] = ExtendSeq( ...
        obj, Seq)
    ExtendedSeq = repmat(Seq, 1, 43);
end


% generate a sequence with random flag.
function [RandomTypes] = GenRandomTypes( ...
        obj, S)
    RandomTypes = [ones(1,S.GUI.RandomImg) zeros(1,143143)];
end


% insert random image sequence into TrialTypes or ImgSeqLabel.
function [TrialTypes, ImgSeqLabel] = InsertRandom( ...
        obj, S, possibleTargetImg, TrialTypes, ImgSeqLabel)
    RandomImgs = [];
    while length(RandomImgs) < S.GUI.RandomImg
        RandomImgs = [RandomImgs, possibleTargetImg(randi(length(possibleTargetImg)))];
    end
    TrialTypes = [RandomImgs, TrialTypes];
    ImgSeqLabel = [RandomImgs, ImgSeqLabel];
end


% generate a sequence with orientation mini blocks and oddballs
function [TrialTypes, ImgSeqLabel] = GenTrialTypesSeq( ...
        obj, S)
    TrialTypes = [];
    ImgSeqLabel = [];
    possibleTargetImg = [2, 3, 4, 5];
    for b = 1:S.GUI.NumBlocks
        TrialBlock = [];
        LabelBlock = [];
        remaining = S.GUI.MaxImg / S.GUI.NumBlocks;
        while remaining > 0
            % find image TrialBlock length
            seqLen = randi([S.GUI.OrienBlockNumMin, min(S.GUI.OrienBlockNumMax, remaining)]);
            if remaining - seqLen < S.GUI.OrienBlockNumMin
                seqLen = remaining;
            end
            % find image for the image TrialBlock
            TargetImg = possibleTargetImg(randi(length(possibleTargetImg)));
            if (isempty(TrialBlock)) && (~isempty(TrialTypes))
                TargetImg = TrialTypes(end);
            end
            if (~isempty(TrialBlock))
                while (TargetImg == TrialBlock(end))
                    TargetImg = possibleTargetImg(randi(length(possibleTargetImg)));
                end
            end
            SeqBlock = TargetImg * ones(1, seqLen);
            % create oddball
            if (S.GUI.OddProb>0)
                oddIdx = rand(size(SeqBlock)) < S.GUI.OddProb;
                oddIdx(1:S.GUI.OddAvoidFrameStart) = false;
                oddIdx(end-S.GUI.OddAvoidFrameEnd+1:end) = false;
                for i = S.GUI.OddAvoidFrameStart+1:length(oddIdx)-S.GUI.OddAvoidFrameEnd
                    if oddIdx(i)
                        oddIdx(i+1:i+S.GUI.OddAvoidFrameBetween) = false;
                    end
                end
                SeqBlock(oddIdx) = 1;
            end
            % generate blocks
            TrialBlock = [TrialBlock, SeqBlock];
            if (~isempty(LabelBlock) && LabelBlock(end) ~= SeqBlock(1))
                SeqBlock(1) = -SeqBlock(1);
            end
            LabelBlock = [LabelBlock, SeqBlock];
            remaining = remaining - seqLen;
        end
        TrialTypes = [TrialTypes, TrialBlock];
        ImgSeqLabel = [ImgSeqLabel LabelBlock];
    end
    ImgSeqLabel(ImgSeqLabel==1) = -1;
    TrialTypes = TrialTypes(1:S.GUI.MaxImg);
    ImgSeqLabel = ImgSeqLabel(1:S.GUI.MaxImg);
    TrialTypes = ExtendSeq(obj, TrialTypes);
    ImgSeqLabel = ExtendSeq(obj, ImgSeqLabel);
    [TrialTypes, ImgSeqLabel] = InsertRandom(obj, S, possibleTargetImg, TrialTypes, ImgSeqLabel);
end


% generate a sequence with short long baseline block
function [StandardTypes] = GenStandardTypes( ...
        obj, S)
    switch S.GUI.StandardMode
        case 1
            StandardTypes = zeros(1, S.GUI.MaxImg);
        case 2
            StandardTypes = ones(1, S.GUI.MaxImg);
        case 3
            StandardTypes = randi([0, 1], 1, S.GUI.MaxImg);
        case 4
            b1 = zeros(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            b2 = ones(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            StandardTypes = repmat([b1 b2], 1, S.GUI.NumBlocks+1);
            StandardTypes = StandardTypes(1:S.GUI.MaxImg);
    end
    StandardTypes = [zeros(1, S.GUI.RandomImg) ExtendSeq(obj, StandardTypes)];
end


% generaate a sequence with fix jitter block
function [FixJitterTypes] = GenFixJitterTypes( ...
        obj, S)
    switch S.GUI.FixJitterMode
        case 1
            FixJitterTypes = zeros(1, S.GUI.MaxImg);
        case 2
            FixJitterTypes = ones(1, S.GUI.MaxImg);
        case 3
            FixJitterTypes = randi([0, 1], 1, S.GUI.MaxImg);
        case 4
            b1 = zeros(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            b2 = ones(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            FixJitterTypes = repmat([b1 b2], 1, S.GUI.NumBlocks+1);
            FixJitterTypes = FixJitterTypes(1:S.GUI.MaxImg);
    end
    FixJitterTypes = [ones(1, S.GUI.RandomImg) ExtendSeq(obj, FixJitterTypes)];
end


% generate a sequence with fix jitter block
function [OddballTypes] = GenOddballTypes( ...
        obj, S)
    switch S.GUI.OddballMode
        case 1
            OddballTypes = zeros(1, S.GUI.MaxImg);
        case 2
            OddballTypes = ones(1, S.GUI.MaxImg);
        case 3
            OddballTypes = randi([0, 1], 1, S.GUI.MaxImg);
        case 4
            b1 = zeros(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            b2 = ones(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            OddballTypes = repmat([b1 b2], 1, S.GUI.NumBlocks+1);
            OddballTypes = OddballTypes(1:S.GUI.MaxImg);
        case 5
            b1 = ones(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            b2 = zeros(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            OddballTypes = repmat([b1 b2], 1, S.GUI.NumBlocks+1);
            OddballTypes = OddballTypes(1:S.GUI.MaxImg);
    end
    OddballTypes = [zeros(1, S.GUI.RandomImg) ExtendSeq(obj, OddballTypes)];
end


% generate a sequence with short long baseline block
function [OptoTypes] = GenOptoTypes( ...
        obj, S, TrialTypes)
    switch S.GUI.OptoMode
        case 1
            OptoTypes = zeros(1, S.GUI.MaxImg);
        case 2
            OptoTypes = ones(1, S.GUI.MaxImg);
        case 3
            OptoTypes = zeros(size(TrialTypes));
            % oddball
            IdxOdd = find(TrialTypes == 1);
            NumPicked = round(S.GUI.OptoProb * length(IdxOdd));
            IdxOddPicked = randsample(IdxOdd, NumPicked);
            OptoTypes(IdxOddPicked) = 1;
            % post oddball
            IdxPostOdd = IdxOdd + 1;
            IdxPostOdd = IdxPostOdd(1:end-1);
            ValidIdxPostOdd = IdxPostOdd(ismember(TrialTypes(IdxPostOdd), [2, 3, 4, 5]));
            NotIdxOddPicked = setdiff(ValidIdxPostOdd, IdxOddPicked + 1);
            NumPicked = round(S.GUI.OptoProb * length(NotIdxOddPicked));
            IdxPostOddPicked = randsample(NotIdxOddPicked, NumPicked);
            OptoTypes(IdxPostOddPicked) = 2;
            % normal
            IdxStandard = find(TrialTypes == 2 | TrialTypes == 3 | TrialTypes == 4 | TrialTypes == 5);
            notAfter1 = IdxStandard(~ismember(IdxStandard, IdxOdd + 1));
            NumPicked = round(S.GUI.OptoProb * length(notAfter1));
            IdxStandardPicked = randsample(notAfter1, NumPicked);
            OptoTypes(IdxStandardPicked) = 3;
            % interval
            nonZeroIndices = find(OptoTypes ~= 0);
            for i = 2:length(nonZeroIndices)
                if nonZeroIndices(i) - nonZeroIndices(i-1) <= S.GUI.OptoAvoidFrameBetween
                    OptoTypes(nonZeroIndices(i)) = 0;
                end
            end
        case 4
            OptoTypes = zeros(1, S.GUI.MaxImg);
            i = randsample(S.GUI.MaxImg, round(S.GUI.OptoProb * S.GUI.MaxImg));
            OptoTypes(i) = 1;
        case 5
            b1 = zeros(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            b2 = ones(1, S.GUI.MaxImg/S.GUI.NumBlocks);
            if (rand() < 0.5)
                OptoTypes = repmat([b1 b2], 1, S.GUI.NumBlocks+1);
            else
                OptoTypes = repmat([b2 b1], 1, S.GUI.NumBlocks+1);
            end
            OptoTypes = OptoTypes(1:S.GUI.MaxImg);
    end
    OptoTypes = ExtendSeq(obj, OptoTypes);
    OptoTypes(1:S.GUI.OptoAvoidFrameStart) = 0;
    OptoTypes(1:S.GUI.RandomImg) = 0;
end


    end
end