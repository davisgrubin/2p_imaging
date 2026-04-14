classdef OptoConfig
    methods

% get pre and post isi for the current stim
function [PreISI, OddISI, PostISI] = GetPrePostISIOdd( ...
        obj, ISIseq, currentImg)
    paddedISIseq = [ISIseq(1), ISIseq, ISIseq(end)];
    PreISI  = paddedISIseq(currentImg);
    OddISI  = paddedISIseq(currentImg + 1);
    PostISI = paddedISIseq(currentImg + 2);
    PreISI  = PreISI / 2;
    PostISI = PostISI / 2;
end

% get pre and post isi for the current stim
function [PreISI, PostISI] = GetPrePostISI( ...
        obj, ISIseq, currentImg)
    paddedISIseq = [ISIseq(1), ISIseq, ISIseq(end)];
    PreISI  = paddedISIseq(currentImg);
    PostISI = paddedISIseq(currentImg + 1);
    PreISI  = PreISI / 2;
    PostISI = PostISI / 2;
end

function [OptoSeq] = GetOptoSeq(obj, S, TrialTypes, OptoTypes, ISIseq)
    OptoSeq = [];
    currentImg = 1;
    while currentImg <= S.GUI.MaxImg
        if TrialTypes(currentImg) == 1
            [PreISI, OddISI, ~] = GetPrePostISIOdd(obj, ISIseq, currentImg);
            % oddball
            if OptoTypes(currentImg) == 1
                OptoSetup.Type = 1;
                OptoSetup.TimeOn = PreISI + S.GUI.GratingDur_s + S.GUI.OptoIntervalOdd;
                OptoSetup.TimeDur = OddISI - 2*S.GUI.OptoIntervalOdd;
                OptoSeq = [OptoSeq OptoSetup];
            % post oddball
            elseif OptoTypes(currentImg+1) == 2
                OptoSetup.Type = 2;
                OptoSetup.TimeOn = PreISI + S.GUI.GratingDur_s + OddISI - S.GUI.OptoOnPreStim;
                OptoSetup.TimeDur = S.GUI.OptoOnPreStim + S.GUI.GratingDur_s + S.GUI.OptoOffPostStim;
                OptoSeq = [OptoSeq OptoSetup];
            % none
            else
                OptoSetup.Type = 0;
                OptoSetup.TimeOn = 0;
                OptoSetup.TimeDur = 0;
                OptoSeq = [OptoSeq OptoSetup];
            end
            currentImg = currentImg + 2;
        else
            % normal
            if OptoTypes(currentImg) == 3
                OptoSetup.Type = 3;
                [PreISI, ~] = GetPrePostISI(obj, ISIseq, currentImg);
                OptoSetup.TimeOn = PreISI - S.GUI.OptoOnPreStim;
                OptoSetup.TimeDur = S.GUI.OptoOnPreStim + S.GUI.GratingDur_s + S.GUI.OptoOffPostStim;
                OptoSeq = [OptoSeq OptoSetup];
            % none
            else
                OptoSetup.Type = 0;
                OptoSetup.TimeOn = 0;
                OptoSetup.TimeDur = 0;
                OptoSeq = [OptoSeq OptoSetup];
            end
            currentImg = currentImg + 1;
        end
    end
end


function [sma] = SetOpto(obj, S, sma, OptoSeq, currentTrial)
    OptoSetup = OptoSeq(currentTrial);
    PMTStartCloseDelay = 0.010;
    PMTCloseTransferDelay = 0.0121;
    PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;          
    PMTMin5VSignalDur = PMTCloseDelay + 0.003;
    PMTStartOpenDelay = 0;
    PMTOpenTransferDelay = 0.0125;
    PMTExtendDur = 0.002;
    LEDOnsetDelay = OptoSetup.TimeOn; 
    PMTOnsetDelay = LEDOnsetDelay - PMTCloseDelay - PMTExtendDur;
    LEDOnDur = S.GUI.LEDOnPulseDur;
    LEDOffDur = S.GUI.LEDOffPulseDur;
    T = LEDOnDur + LEDOffDur;
    PMT5VDur = max(PMTCloseDelay + LEDOnDur, PMTMin5VSignalDur) + PMTExtendDur;
    PMT0VDur =  T - PMT5VDur;
    numPMTLEDCycles = floor(OptoSetup.TimeDur/T);
    sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', LEDOnDur, 'OnsetDelay', LEDOnsetDelay,...
        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
        'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
        'GlobalTimerEvents', 0, 'OffsetValue', 0);
    sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', PMT5VDur, 'OnsetDelay', PMTOnsetDelay,...
        'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
        'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMT0VDur,...
        'GlobalTimerEvents', 0, 'OffsetValue', 0);
end


    end
end