from midi2audio import FluidSynth

#FluidSynth().play_midi('best.midi')
fs = FluidSynth()
fs.midi_to_audio('best.midi', 'output.wav')
