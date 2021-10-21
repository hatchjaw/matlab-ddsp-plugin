classdef MorphablePlugin < audioPlugin
% MORPHABLEPLUGIN A plugin running a morphable DDSP decoder.
%
%  DDSP plugins extract pitch and loudness information from their input
%  in real-time and output audio generated by a pretrained DDSP decoder.
    
    properties (Access = private, Constant)
        BufSize = 20000; % size of the internal input and output buffers
    end
    
    properties (Access = private)
        Dec;             % the DDSP decoder
        Synth;           % the spectral modeling synthesizer
        InBuf;           % input buffer
        OutBuf;          % output buffer
        
        CurrFrameSize;   % save the current value of the FrameSize property
                         % to be able to tell when it changes
        
        PrevFrame;       % save previous input frame for overlapping
                         % pitch detection windows
        L = 2;           % order of the harmonic summation for pitch detection
    end
    
    properties
        % Ratio (100-Morph):Morph of weights in model 1 to weights in model 2 
        % (see MorphableDecoder).
        Morph = 50;
        % octave scaling of the detected input frequency
        FreqScale = 0;
        % input gain in dB
        InGain = 0;      
        % output gain in dB
        OutGain = 0;     
        % Number of samples generated based on a single run of the decoder.
        % The models included here were trained on a frame rate of 250, 
        % corresponding to a frame size of 176 at a sample rate of 44.1 kHz.
        % However, invoking the decoder at this rate turned out too
        % computationally heavy for real-time audio synthesis. For this reason
        % the frame size is  an adjustable parameter: Higher frame sizes invoke
        % the decoder less often and enable real-time, but decrease the accuracy
        % of the generated synthesizer parameters.
        FrameSize = 512; 
        % scaling for the number of FFT points for the 
        % harmonic summation
        FftResol = 5;    
    end
    
    properties (Constant)
        LDMIN = -120;    % loudness range for scaling
        LDMAX = 0;

        F0MIN = 60;      % search window for pitch detection
        F0MAX = 5000;
        
        PluginInterface = audioPluginInterface( ...
            'PluginName', 'MATLAB Morphable DDSP', ...
            'UniqueId', 'smcd', ...
            'VendorName', 'SMC7 2021', ...
            'VendorVersion', '1.0.0', ...
            'InputChannels', 1, ...
            'OutputChannels', 1, ...
            audioPluginGridLayout( ...
                'RowHeight', [100 100 100 100 100],...
                'ColumnWidth', [150 150 150] ...
            ), ...
            audioPluginParameter('InGain', ...
                'DisplayName', 'Input Gain', ...
                'DisplayNameLocation', 'above',...
                'Label', 'dB', ...
                'Mapping', {'lin', -10, 10},...
                'Style', 'rotaryknob',...
                'Layout', [2 1] ...
            ),...
            audioPluginParameter('FreqScale', ...
                'DisplayName', 'Octave Shift', ...
                'DisplayNameLocation', 'above',...
                'Mapping', {'int', -2, 2},...
                'Style', 'rotaryknob',...
                'Layout', [2 2] ...
            ),...
            audioPluginParameter('OutGain', ...
                'DisplayName', 'Output Gain', ...
                'DisplayNameLocation', 'above',...
                'Label', 'dB', ...
                'Mapping', {'lin', -10, 10},...
                'Style', 'rotaryknob',...
                'Layout', [2 3] ...
            ),...
            audioPluginParameter('Morph', ...
                'DisplayName', 'Morph', ...
                'DisplayNameLocation', 'left', ...
                'Mapping', {'lin', 0., 100.}, ...
                'Layout', [3 2; 3 3] ...
            ), ...
            audioPluginParameter('FftResol',...
                'DisplayName', 'Pitch Resolution', ...
                'DisplayNameLocation', 'left', ...
                'Mapping', {'int', 1, 6},...
                'Layout', [4 2; 4 3] ...
            ),...
            audioPluginParameter('FrameSize', ...
                'DisplayName', 'Frame Size', ...
                'DisplayNameLocation', 'left', ...
                'Mapping', {'int', 300, 2048},...
                'Layout', [5 2; 5 3] ...
            )...
        );
    end
    
    methods
        function plugin = MorphablePlugin
            % initialize properties
            
            plugin.Dec = MorphableDecoder();
            plugin.Synth = SpectralModelingSynth;
            plugin.InBuf = CircularBuffer(plugin.BufSize);
            
            % The plugin introduces a latency of one frame for two reasons:
            %   * slightly more accurate pitch detection
            %   * avoid running out of audio when the external input buffer
            %     size is smaller than the current decoder frame size
            
            % so initialize the output buffer with FrameSize zeros.
            plugin.OutBuf = CircularBuffer(plugin.BufSize, plugin.FrameSize);
            
            plugin.CurrFrameSize = plugin.FrameSize;
            
            % pitch detection buffers are downsampled for speed
            plugin.PrevFrame = zeros(ceil(plugin.FrameSize/2), 1);
            
            % to make sure. still need to initialize properties in the
            % constructor for code generation
            plugin.reset;
        end
      
        function out = process(plugin, in)
            
            % if the user changed the decoder frame size, we need to adjust
            % the plugin latency
            if (plugin.FrameSize ~= plugin.CurrFrameSize)
                plugin.reset;
            end
            
            % The high-level algorithm for the `process` method:
            %
            %    1. Write all input into an internal buffer
            %    
            %    2. While there are more than FrameSize samples in the buffer:
            %       2.1 Read a frame of input
            %       2.2 Calculate f0 and loudness for that frame
            %       2.3 Call the decoder and generate audio
            %       2.4 Write audio into the output buffer
            %    
            %    3. Return length(in) samples of audio from the output
            %       buffer
            
            plugin.InBuf.write(in);
            plugin.generateAudio();
            out = plugin.OutBuf.read(length(in));
        end

        function reset(plugin)
            % reset buffers and decoder and adjust latency to new FrameSize
            
            plugin.CurrFrameSize = plugin.FrameSize;
%             if (~verLessThan('matlab', '9.9'))
%                 %requires MATLAB >= R2020b (9.9)
%                 plugin.setLatencyInSamples(plugin.FrameSize);
%             end
            plugin.PrevFrame = zeros(ceil(plugin.FrameSize/2), 1);
            plugin.Dec.reset;
            plugin.InBuf.reset;
            plugin.OutBuf.reset(plugin.FrameSize);
        end
        
        function generateAudio(plugin)
            sampleRate = plugin.getSampleRate;
            
            % Read in frame by frame from the input buffer, if there are any
            while plugin.InBuf.nElems >= plugin.FrameSize
                in = plugin.InBuf.read(plugin.FrameSize);
                
                %%%%%%%%% Loudness calculation %%%%%%%%%%
                
                power = sum(in.^2) / plugin.FrameSize;
                
                % momentary loudness in LUFS + input gain
                % see https://mathworks.com/help/audio/ref/integratedloudness.html#bvb_vd6
                ld = -0.691 + 10*log10(power) + plugin.InGain;
                
                % normalize for decoder input
                ldScaled = ld / (plugin.LDMAX - plugin.LDMIN) + 1;
                
                
                %%%%%%%%%% Pitch detection %%%%%%%%%%%%%
                
                
                % downsampling by 2: slight accuracy tradeoff for better
                % speed
                downsampled = downsample(in, 2);
                pitchFrameSize = length(downsampled);
                
                % concatenate with previous frame and apply hann window
                % to get analysis window for pitch detection
                
                % this corresponds to a window size of two frames and 50%
                % overlap
                
                pitchFrame = [plugin.PrevFrame; downsampled] .* ...
                    hann(pitchFrameSize*2, 'periodic');
                
                % save current frame for next time
                plugin.PrevFrame = downsampled(1:pitchFrameSize,1);
                

                
                % Perform pitch detection by harmonic summation: 
                
                % 1. Get the magnitude spectrum of the analysis window
                % 2. For a range of candidate frequencies, sum the
                % magnitude of the candidate frequency and its L integer
                % multiples
                % 3. Choose the candidate frequency with the highest sum
                
                
                
                % the size of the FFT is dependent on the size of the
                % analysis window, the model order L and a scaling factor.
                
                % increasing FftResol increases the resolution of the pitch
                % detection but also the computational cost.
                nFft = round(plugin.FftResol*pitchFrameSize*2*plugin.L);
                
                % zero pad input frame and calculate magnitude spectrum
                spec = abs(fft([pitchFrame; zeros(nFft - pitchFrameSize*2, 1)])).^2;
                
                % Choose the bins of the FFT as candidate frequencies.
                
                % adjust sampleRate to account for the downsampling.
                % we can't simply divide it by 2 because the length of the
                % input might not be even!
                
                kstart = ceil(nFft * plugin.F0MIN / ...
                    (sampleRate * pitchFrameSize / plugin.FrameSize));
                
                kstop = floor(nFft * plugin.F0MAX / ...
                    (sampleRate * pitchFrameSize / plugin.FrameSize));

                % integer multiples
                ls = 1:plugin.L;
                
                % search for the best candidate
                bestk = 0;
                bestval = 0;
                for i=kstart:kstop
                    % this might happen at very low input sizes, which are
                    % tested by validateAudioPlugin. Simply reduce the
                    % order of the model if it happens
                    if (i+1) * ls(end) > nFft
                        ls = ls(1:end-1);
                    end
                    
                    % calculate the actual summation
                    val = sum(spec((i+1)*ls));
                    
                    % update best candidate
                    if val > bestval
                        bestk = i;
                        bestval = val;
                    end
                end
                
                % convert the best candidate back to Hz 
                f0 = sampleRate * (pitchFrameSize / plugin.FrameSize) * bestk / nFft;
                
                % octave scaling
                f0 = f0 * 2^plugin.FreqScale;
                
                % normalize for decoder input
                f0Scaled = utils.hzToMidi(f0) / 127;
                
                % get features from decoder
                [amp, harmDist, noiseMags] = plugin.Dec.call(ldScaled, f0Scaled);

                % synthesize audio
                frame = plugin.Synth.getAudio(f0, amp, harmDist, ...
                    noiseMags, sampleRate, plugin.FrameSize); 
                
                % apply output gain and write to internal buffer
                plugin.OutBuf.write(frame * 10^(plugin.OutGain/20));
            end
        end
        
        function set.Morph(plugin, morph)
            plugin.Morph = morph;
            plugin.updateDecoder();
        end
    end
    
    methods (Access = private)
        function updateDecoder(plugin)
            plugin.Dec.updateLayers(plugin.Morph);
        end
    end
end