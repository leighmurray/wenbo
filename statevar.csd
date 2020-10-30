<CsoundSynthesizer>
<CsOptions>

</CsOptions>

<CsInstruments>
sr 	= 	44100
ksmps 	= 	32
nchnls 	= 	1

instr	1 ; play audio from disk
kSpeed  init     1           ; playback speed
iSkip   init     0           ; inskip into file (in seconds)
iLoop   init     0           ; looping switch (0=off 1=on)
SFilename  chnget "filename"
kCutoffFrequency chnget "cutoff"
kResonance init 0.6
; read audio from disk using diskin2 opcode
a1      diskin2  SFilename, kSpeed, iSkip, iLoop

ahp, a2, abp, abr statevar a1, kCutoffFrequency, 4
      out      a2          ; send audio to outputs
endin
</CsInstruments>

<CsScore>
</CsScore>
</CsoundSynthesizer>
