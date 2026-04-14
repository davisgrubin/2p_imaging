function SoftCodeHandler(code)
    HandleVideo(code)
end


function HandleVideo(code)
    global BpodSystem
    switch true
        case code == 255 % stop video
            BpodSystem.PluginObjects.V.stop;
        case code == 254 % stop video
            BpodSystem.PluginObjects.V.stop;     
        case code >= 0 && code <= 253 % play video
            BpodSystem.PluginObjects.V.play(code);
    end
end
