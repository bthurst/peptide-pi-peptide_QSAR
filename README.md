# peptide-pi-peptide_QSAR
The data and code used to predict peptide free energies and alignments in https://doi.org/10.1080/08927022.2018.1469754

Three csv files are included: data on the free energies of aggregation, data on peptide alignment, and a subset of the features used in high-throughput screening of peptide chemistries (the full csv is too large).

The python code will implement the model requested by parameters in the python file by fitting it to the alignment and free energy data contained in their respective csv files. It will then use this model and the features provided in the features csv file to predict which chemistries will optimize the peptide alignment. For more information, read the comments in the python file.

PaDEL-Descriptor was used to generate descriptors: http://www.yapcwsoft.com/dd/padeldescriptor/
