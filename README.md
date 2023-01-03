# ehd_exsitu
Image analysis for ex-situ characterization of ehd-printed patterns

## Workflow:
1. Use align_pattern.py to set the offset and angle for a mosaic image so that the EHD toolpath pattern lines up with the printed pattern
2. Use the GUI in place_patches.py to place an image patch over each isolated print pattern
3. Run parse_patches.py to run image analysis on each patch, extracting metrics.

## Requirements:
Section is incomplete
 - sklearn
 
# ehd_dataset
The EHD_Loader object holds multiple training datasets in the loader.datasets array, each a dataframe containing waveforms and measurements from a single experiment. 
