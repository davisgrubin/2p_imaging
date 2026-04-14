function passive_interval_oddball_202412
    global BpodSystem
    MonitorID  = 2;
    trialManager = BpodTrialManager;

    %% Import scripts

    m_InitGUI      = InitGUI;
    m_TrialConfig  = TrialConfig;
    m_AVstimConfig = AVstimConfig;
    m_OptoConfig   = OptoConfig;
    
    
    %% Turn off Bpod LEDs
    
    BpodSystem.setStatusLED(0);
    
    %% Assert HiFi module is present + USB-paired (via USB button on console GUI)
    
    BpodSystem.assertModule('HiFi', 1);
    H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);
    
    
    %% Define parameters

    [DefaultConfig, FixJitterMode, StandardMode, OddballMode, OptoMode] = m_TrialConfig.GetConfig();
    
    global S
    [S] = m_InitGUI.SetParams(BpodSystem);
    [S] = m_InitGUI.UpdateConfig( ...
        S, DefaultConfig, FixJitterMode, StandardMode, OddballMode, OptoMode);

    BpodParameterGUI('init', S);


    %% Define stimuli and send to analog module
    
    SF = 44100; % Use lower sampling rate (samples/sec) to allow for longer duration audio file (max length limited by HiFi)
    H.SamplingRate = SF;
    Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset
    H.DigitalAttenuation_dB = -35; % Set a comfortable listening level for most headphones (useful during protocol dev).


    %% Setup video
    
    BpodSystem.PluginObjects.V = [];
    BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(MonitorID, 0, [0 0], [180 180], 0); % Assumes Sync patch = 180x180 pixels
    BpodSystem.PluginObjects.V.SyncPatchIntensity = 255;
    Xsize = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
    Ysize = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
    FPS   = BpodSystem.PluginObjects.V.DetectedFrameRate;
    
    % load frame images
    
    [VideoGrayFixed] = m_AVstimConfig.GenGreyImg(Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(1, VideoGrayFixed);
    
    ImgParams.spatialFreq = 0.005;
    ImgParams.contrast    = 1;
    ImgParams.phase       = 0.5;
    
    ImgParams.orientation = 0;
    [VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(2, VideoGrating);
    
    ImgParams.orientation = 45;
    [VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(3, VideoGrating);
    
    ImgParams.orientation = 90;
    [VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(4, VideoGrating);
    
    ImgParams.orientation = 135;
    [VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
    BpodSystem.PluginObjects.V.loadVideo(5, VideoGrating);
    
    VisStim.orientation = [0 45 90 135];
    VisStim.GratingIdx = [2 3 4 5];
    VisStim.OddballFlag = 0;
    VisStim.OddballISI = 0.19961106;

    [VisStim] = m_AVstimConfig.GetVisStimImg(S, BpodSystem, FPS, VisStim, 2);
    GrayInitBNCSync = [repmat(VisStim.Img.GrayFrame_SyncW, 1, 120) VisStim.Img.GrayFrame_SyncBlk];
    BpodSystem.PluginObjects.V.Videos{6} = struct;
    BpodSystem.PluginObjects.V.Videos{6}.nFrames = 121;
    BpodSystem.PluginObjects.V.Videos{6}.Data = GrayInitBNCSync;

    pause(5.0);
    BpodSystem.PluginObjects.V.TimerMode = 2;
    BpodSystem.PluginObjects.V.play(0);
    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler';    
    BpodSystem.PluginObjects.V.play(6);

    input('Set parameters and press enter to continue >> ', 's'); 
    S = BpodParameterGUI('sync', S);

    %% sequence configuration and visualization

    % random type
    % 0: deterministic
    % 1: random
    [RandomTypes] = m_TrialConfig.GenRandomTypes(S);

    % trial type
    % 1: oddball
    % 2: 0 orien stim
    % 3: 45 orien stim
    % 4: 90 orien stim
    % 5: 135 orien stim
    [TrialTypes, ImgSeqLabel] = m_TrialConfig.GenTrialTypesSeq(S);
    
    % baseline type
    % 0: short ISI
    % 1: long ISI
    [StandardTypes] = m_TrialConfig.GenStandardTypes(S);
    
    % fix jitter type
    % 0: fix
    % 1: jitter
    [FixJitterTypes] = m_TrialConfig.GenFixJitterTypes(S);
    
    % oddball type
    % 0: short oddball
    % 1: long oddball
    [OddballTypes] = m_TrialConfig.GenOddballTypes(S);

    % opto type
    % 0: opto off
    % 1: opto on for oddball
    % 2: opto on for post oddball
    % 3: opto on for normal
    [OptoTypes] = m_TrialConfig.GenOptoTypes(S, TrialTypes);

    % isi
    [ISIseq] = m_AVstimConfig.GetISIseq( ...
        S, TrialTypes, RandomTypes, StandardTypes, FixJitterTypes, OddballTypes);

    % video sequence
    [VisStimSeq] = m_AVstimConfig.GetVisStimSeq( ...
        S, BpodSystem, FPS, VisStim, TrialTypes, ISIseq);
    VisStimAll = [m_AVstimConfig.GetUnitVideo(VisStim.Img.GrayFrame_SyncBlk, 520);];
    for i = 1:length(VisStimSeq)
        VisStimAll = [VisStimAll VisStimSeq(i).Data.Full];
    end
    % audio
    [AudStim] = m_AVstimConfig.GetAudStim(S, SF, Envelope);
    H.load(5, AudStim);
    % opto sequence
    [OptoSeq] = m_OptoConfig.GetOptoSeq(S, TrialTypes, OptoTypes, ISIseq);
    
    % save into session data
    BpodSystem.Data.RandomTypes = RandomTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.ImgSeqLabel = ImgSeqLabel(1:S.GUI.MaxImg);
    BpodSystem.Data.StandardTypes = StandardTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.FixJitterTypes = FixJitterTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.OddballTypes = OddballTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.OptoTypes = OptoTypes(1:S.GUI.MaxImg);
    BpodSystem.Data.ISIseq = ISIseq;
    SaveBpodSessionData


    %% Start video

    pause(S.GUI.SpontSilenceTimeSess)
    BpodSystem.PluginObjects.V.Videos{25} = struct;
    BpodSystem.PluginObjects.V.Videos{25}.nFrames = length(VisStimAll); 
    BpodSystem.PluginObjects.V.Videos{25}.Data = VisStimAll;
    BpodSystem.PluginObjects.V.play(25);


    %% Main trial loop

    sma = NewStateMatrix();
    sma = AddState(sma, 'Name', 'Start', ...
        'Timer', 0,...
        'StateChangeConditions', {'BNC1High', 'AudVisStim'},...
        'OutputActions', {'HiFi1','*'});
    sma = AddState(sma, 'Name', 'AudVisStim', ...
        'Timer', 0,...
        'StateChangeConditions', {'BNC1Low', 'End'},...
        'OutputActions', {'HiFi1', ['P', 4]});
    sma = AddState(sma, 'Name', 'End', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', '>exit'},...
        'OutputActions', {'HiFi1', 'X'});
    trialManager.startTrial(sma);

    wb = waitbar(0, 'Starting', 'Name','Playing stimulus videos');
    for ImgIdx = 1:S.GUI.MaxImg        
        waitbar(ImgIdx/S.GUI.MaxImg, wb, sprintf('If experiments go shit say I LOVE YICONG FOREVER! \n %d/%d', ImgIdx, S.GUI.MaxImg));        
        trialManager.getCurrentEvents({'End'});

        %% construct state matrix
        
        sma = NewStateMatrix();
        sma = AddState(sma, 'Name', 'Start', ...
            'Timer', 10,...
            'StateChangeConditions', {'BNC1High', 'AudVisStim', 'Tup', 'End'},...
            'OutputActions', {'HiFi1','*'});
        sma = AddState(sma, 'Name', 'AudVisStim', ...
            'Timer', 0,...
            'StateChangeConditions', {'BNC1Low', 'End'},...
            'OutputActions', {'HiFi1', ['P', 4]});
        sma = AddState(sma, 'Name', 'End', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', '>exit'},...
            'OutputActions', {'HiFi1', 'X'});
        
        SendStateMachine(sma, 'RunASAP');
        trialManager.getTrialData;
        trialManager.startTrial(sma);


        %% save data

        HandlePauseCondition;
        if BpodSystem.Status.BeingUsed == 0
            clear global M;
            BpodSystem.PluginObjects.V = [];
            BpodSystem.setStatusLED(1);
            return
        end


    end
    
    delete(wb)
    input('Session successfully ended. press enter to exit >> ', 's'); 
    
    clear global M;
    BpodSystem.PluginObjects.V = [];
end
