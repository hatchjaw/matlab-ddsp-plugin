function midi = hzToMidi(hz)
    % Convert a frequency in Hz to a (continuous) Midi note value

    % 440 Hz = Midi note 69
    midi = 12 * (log2(hz) - log2(440)) + 69;
    midi(midi < 0) = 0;
end