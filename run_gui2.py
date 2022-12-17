from pic_parser.patches_gui import run_patches_gui


if __name__ == "__main__":
    run_patches_gui(im_path="./tests/patch_test_files/2-may-22 run1.bmp",
                    pattern_path="./tests/patch_test_files/pattern.txt",
                    params_file="./tests/patch_test_files/pattern_params.json")
