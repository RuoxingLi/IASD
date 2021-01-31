# IASDF 
This IASDF (Impairment-aware Spectrum Defragmentation) repo is mainly based on the optical network model in [1], while some parts of the settings are based on other relevant papers. 

The ia_mbb_re function is a simple reactive channel reconfiguration algorithm based on the Make-before-Break reconfiguration scheme proposed in [2], while the ia_pp_re function is a reactive channel reconfiguration algorithm based on the Push-Pull technique in [3]. Currently, these two algorithms are logically simple but can also be much computationally intensive when the network is heavily loaded. 

Some parts of the functions can be further improved later to accelerate the computation speed.   
