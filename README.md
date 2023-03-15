This code base was archived on March 14, 2023. This represents a mid-stream snapshot of the development cycle of a machine learning approach to predicting the line widths resulting from an EHD printing process. Development of the image analysis and dataset construction routines continues in onakanob/ehd-dataset. Machine learning models and evaluation continues in onakanob/ehd-ml.

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
The EHD_Loader object holds multiple training datasets in the loader.datasets array, each a dataframe containing waveforms and measurements from a single experiment. When returning a dataset, the "xtype" and "ytype" arguments control how the X and Y variables (input and supervised output, respectively) will be formatted. The following options are available:

## xtype
 - vector
 - wave
 - last_wave
 - last_vector
 - normed_squares
 - v_normed_squares
 
## ytype
 - area
 - print_length
 - max_width
 - mean_width
 - obj_count
 - jetted
 - jetted_selectors

# ehd_model

## Regressors:
 - MLE
 - cold_RF
 - cold_MLP
 - only_pretrained_RF
 - only_pretrained_MLP

## Classifiers:
 - MLE_class
 - cold_RF_class
 - cold_MLP_class
 - only_pretrained_RF_class
 - only_pretrained_MLP_class
