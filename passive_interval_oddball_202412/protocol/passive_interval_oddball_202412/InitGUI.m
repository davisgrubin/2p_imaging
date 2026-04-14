classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings;
    if isempty(fieldnames(S))

        % stim params
        S.GUI.AudioStimEnable = 1;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 1.0;
        S.GUI.AudioStimFreq_Hz = 11025;
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.2;
        S.GUIPanels.AudVisStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz', 'VisStimEnable', 'GratingDur_s'};
        S.GUI.MaxImg = 2000;
        S.GUI.RandomImg = 100;
        S.GUI.NumBlocks = 20;
        S.GUI.OrienBlockNumMin = 15;
        S.GUI.OrienBlockNumMax = 20;
        S.GUI.SpontSilenceTimeSess = 0;
        S.GUIPanels.Passive = {'MaxImg', 'RandomImg', 'NumBlocks', 'OrienBlockNumMin', 'OrienBlockNumMax', 'SpontSilenceTimeSess'};
        
        % oddball
        S.GUI.OddballMode = 1;
        S.GUIMeta.OddballMode.Style = 'popupmenu';
        S.GUIMeta.OddballMode.String = {'Short', 'Long', 'Random', 'Block', 'reverse'};
        S.GUI.OddAvoidFrameStart = 3;
        S.GUI.OddAvoidFrameEnd = 3;
        S.GUI.OddAvoidFrameBetween = 3;
        S.GUI.ShortStandardShortOdd = 0.5;
        S.GUI.ShortStandardLongOdd = 2.5;
        S.GUI.LongStandardShortOdd = 2.5;
        S.GUI.LongStandardLongOdd = 4.5;
        S.GUI.OddProb = 0.15;
        S.GUIPanels.Oddball = {'OddballMode', 'OddAvoidFrameStart', 'OddAvoidFrameEnd', 'OddAvoidFrameBetween', 'ShortStandardShortOdd', 'ShortStandardLongOdd', 'LongStandardShortOdd', 'LongStandardLongOdd', 'OddProb'};

        % ISI
        S.GUI.StandardMode = 1;
        S.GUIMeta.StandardMode.Style = 'popupmenu';
        S.GUIMeta.StandardMode.String = {'Short', 'Long', 'Random', 'Block'};
        S.GUI.ShortStandardISI = 1.0;
        S.GUI.LongStandardISI = 4.0;
        S.GUI.FixJitterMode = 1;
        S.GUIMeta.FixJitterMode.Style = 'popupmenu';
        S.GUIMeta.FixJitterMode.String = {'Fix', 'Jitter', 'Random', 'Block'};
        S.GUI.ShortRandomMin = 0.5;
        S.GUI.ShortRandomMax = 2.5;
        S.GUI.LongRandomMin = 2.5;
        S.GUI.LongRandomMax = 4.5;
        S.GUI.NumRandomBins = 1106;
        S.GUIPanels.FixJitterISI = {'StandardMode', 'ShortStandardISI', 'LongStandardISI', 'FixJitterMode', 'ShortRandomMin', 'ShortRandomMax', 'LongRandomMin', 'LongRandomMax', 'NumRandomBins'};

        % Optogentics
        S.GUI.OptoMode = 1;
        S.GUIMeta.OptoMode.Style = 'popupmenu';
        S.GUIMeta.OptoMode.String = {'off', 'on', 'default', 'Random', 'Block'};
        S.GUI.OptoProb = 0.5;
        S.GUI.OptoOnPreStim = 0.2;
        S.GUI.OptoOffPostStim = 0.4;
        S.GUI.OptoIntervalOdd = 0.2;
        S.GUI.LEDOnPulseDur = 0.015;
        S.GUI.LEDOffPulseDur = 0.085;
        S.GUI.OptoAvoidFrameStart = 2;
        S.GUI.OptoAvoidFrameBetween = 1;
        S.GUIPanels.Opto = {'OptoMode', 'OptoProb', 'OptoOnPreStim', 'OptoOffPostStim', 'OptoIntervalOdd', 'LEDOnPulseDur', 'LEDOffPulseDur', 'OptoAvoidFrameStart', 'OptoAvoidFrameBetween'};    
    end
end


function [S] = UpdateConfig( ...
        obj, S, DefaultConfig, FixJitterMode, StandardMode, OddballMode, OptoMode)
    switch DefaultConfig
        % 3331Random: 3 mins for 100 main images
        case 1
            S.GUI.FixJitterMode = 3;
            S.GUI.StandardMode  = 3;
            S.GUI.OddballMode   = 3;
            S.GUI.OptoMode      = 1;
            S.GUI.MaxImg = 2000;
            S.GUI.RandomImg = 2000;
            S.GUI.NumBlocks = 10;
            S.GUI.ShortRandomMin = 0.5;
            S.GUI.ShortRandomMax = 2.5;
        % 1451ShortLong: 3 mins for 100 main images
        case 2
            S.GUI.FixJitterMode = 1;
            S.GUI.StandardMode  = 4;
            S.GUI.OddballMode   = 5;
            S.GUI.OptoMode      = 1;
            S.GUI.MaxImg = 2000;
            S.GUI.RandomImg = 100;
            S.GUI.NumBlocks = 40;
            S.GUI.ShortStandardISI = 1.0;
            S.GUI.LongStandardISI = 2.0;
            S.GUI.ShortStandardLongOdd = 2.0;
            S.GUI.LongStandardShortOdd = 1.0;
            S.GUI.OddProb = 0.05;
            S.GUI.OddAvoidFrameStart = 5;
            S.GUI.OddAvoidFrameEnd = 5;
            S.GUI.OddAvoidFrameBetween = 5;
        % 4131FixJitterOdd: 3 mins for 100 main images
        case 3
            S.GUI.FixJitterMode = 4;
            S.GUI.StandardMode  = 1;
            S.GUI.OddballMode   = 3;
            S.GUI.OptoMode      = 1;
            S.GUI.MaxImg = 2000;
            S.GUI.RandomImg = 100;
            S.GUI.NumBlocks = 10;
            S.GUI.ShortStandardISI = 1.5;
            S.GUI.ShortRandomMin = 0.5;
            S.GUI.ShortRandomMax = 2.5;
            S.GUI.ShortStandardShortOdd = 0.5;
            S.GUI.ShortStandardLongOdd = 2.5;
        % extended 3331Random:
        case 4
            S.GUI.FixJitterMode = 3;
            S.GUI.StandardMode  = 3;
            S.GUI.OddballMode   = 3;
            S.GUI.OptoMode      = 1;
            S.GUI.MaxImg = 600;
            S.GUI.RandomImg = 600;
            S.GUI.NumBlocks = 10;
            S.GUI.ShortRandomMin = 2.5;
            S.GUI.ShortRandomMax = 7.5;
            S.GUI.NumRandomBins = 10;
        % no predefined parameters
        case 5
            S.GUI.FixJitterMode = FixJitterMode;
            S.GUI.StandardMode  = StandardMode;
            S.GUI.OddballMode   = OddballMode;
            S.GUI.OptoMode      = OptoMode;
    end
end

    end
end
